# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from matplotlib import pyplot as plt

BASE_PATH = os.path.join(os.path.dirname(__file__), "../")


# =========================
# Config
# =========================
class Config:
    def __init__(self):
        self.lr = 8e-4
        self.eps = 1e-8
        self.weight_decay = 0.0
        self.epochs = 200
        self.batch_size = 128
        self.device = "cpu"

        self.in_dim = 6      # err(t,t-1,t-2), qd(t,t-1,t-2)
        self.units = 32
        self.layers = 2
        self.out_dim = 1     # Δq
        self.act = "softsign"

        self.dt = 0.02
        self.max_delta_q = 0.5   # rad


# =========================
# Dataset
# =========================
class ActuatorDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return {
            "x": self.xs[idx],
            "y": self.ys[idx],
        }


# =========================
# Activation
# =========================
class Act(nn.Module):
    def __init__(self, act):
        super().__init__()
        self.act = act

    def forward(self, x):
        if self.act == "relu":
            return F.relu(x)
        elif self.act == "tanh":
            return torch.tanh(x)
        elif self.act == "softsign":
            return F.softsign(x)
        else:
            raise RuntimeError(f"Unknown activation {self.act}")


# =========================
# Network
# =========================
def build_mlp(config):
    layers = [nn.Linear(config.in_dim, config.units), Act(config.act)]
    for _ in range(config.layers - 1):
        layers += [nn.Linear(config.units, config.units), Act(config.act)]
    layers += [nn.Linear(config.units, config.out_dim)]
    return nn.Sequential(*layers)


# =========================
# Data loading
# =========================
def load_data(csv_path):
    data = pd.read_csv(csv_path)

    joint_pos_cols = [
        c for c in data.columns
        if c.startswith("joint_pos_") and not c.startswith("joint_pos_target_")
    ]

    motor_ids = sorted(int(c.split("_")[-1]) for c in joint_pos_cols)
    num_motors = max(motor_ids) + 1

    print(f"Detected {num_motors} motors")

    data_dict = {}
    for key in ["joint_pos_", "joint_pos_target_", "joint_vel_"]:
        cols = [f"{key}{i}" for i in range(num_motors)]
        data_dict[key] = data[cols].values

    return data_dict, num_motors


# =========================
# Feature & Label
# =========================
def process_data(data_dict, num_motors, step):
    q = torch.tensor(data_dict["joint_pos_"], dtype=torch.float)
    q_rl = torch.tensor(data_dict["joint_pos_target_"], dtype=torch.float)
    qd = torch.tensor(data_dict["joint_vel_"], dtype=torch.float)

    delta_q = q_rl - q

    xs, ys = [], []

    for i in range(num_motors):
        x_i = torch.cat([
            delta_q[step:    , i:i+1],
            delta_q[step-1:-1, i:i+1],
            delta_q[step-2:-2, i:i+1],
            qd[step:    , i:i+1],
            qd[step-1:-1, i:i+1],
            qd[step-2:-2, i:i+1],
        ], dim=1)

        y_i = delta_q[step:, i:i+1]

        xs.append(x_i)
        ys.append(y_i)

    xs = torch.cat(xs, dim=0)
    ys = torch.cat(ys, dim=0)

    return xs, ys, q.numpy(), q_rl.numpy()


# =========================
# Training
# =========================
def train_network(xs, ys, save_path, config):
    dataset = ActuatorDataset(xs, ys)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train

    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size)

    model = build_mlp(config).to(config.device)
    opt = Adam(model.parameters(), lr=config.lr, eps=config.eps)

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x = batch["x"].to(config.device)
            y = batch["y"].to(config.device)

            pred = model(x)
            loss = ((pred - y) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        if epoch % 5 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["x"].to(config.device)
                    y = batch["y"].to(config.device)
                    val_loss += ((model(x) - y) ** 2).mean().item()
            val_loss /= len(val_loader)

            print(f"[{epoch:03d}] train={train_loss:.4f} val={val_loss:.4f}")

        torch.jit.script(model).save(save_path)

    return model


# =========================
# Train / Play
# =========================
def train_and_plot(data_path, model_path, load_model, config):
    data_dict, num_motors = load_data(data_path)
    step = 2

    xs, ys, q, q_rl = process_data(data_dict, num_motors, step)

    if load_model:
        model = torch.jit.load(model_path)
    else:
        model = train_network(xs, ys, model_path, config)

    with torch.no_grad():
        dq_pred = model(xs).reshape(num_motors, -1).T.numpy()

    dq_pred = np.clip(dq_pred, -config.max_delta_q, config.max_delta_q)

    q = q[step:step + len(dq_pred)]
    q_rl = q_rl[step:step + len(dq_pred)]
    q_final = q_rl + dq_pred

    t = np.arange(len(q_final)) * config.dt

    fig, axs = plt.subplots(6, 2, figsize=(14, 8))
    axs = axs.flatten()

    for i in range(num_motors):
        axs[i].plot(t, q_rl[:, i], label="q_rl (RL)")
        axs[i].plot(t, q_final[:, i], "--", label="q_final")
        axs[i].plot(t, q[:, i], label="q_real")

    axs[0].legend()
    plt.tight_layout()
    plt.show()


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "play"], required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data_path = args.data
    model_path = args.output

    config = Config()

    train_and_plot(
        data_path,
        model_path,
        load_model=(args.mode == "play"),
        config=config,
    )


if __name__ == "__main__":
    main()
