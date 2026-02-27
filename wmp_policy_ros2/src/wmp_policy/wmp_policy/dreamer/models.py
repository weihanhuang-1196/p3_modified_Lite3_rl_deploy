# MIT License

# Copyright (c) 2023 NM512

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
# All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

import copy
import torch
from torch import nn
import numpy as np

from . import tools
from . import networks

to_np = lambda x: x.detach().cpu().numpy()

class BYOLAugmentor:
    def __init__(self, prob_mask=0.3, prob_blur=0.5, prob_noise=0.5):
        self.prob_mask = prob_mask
        self.prob_blur = prob_blur
        self.prob_noise = prob_noise

    def __call__(self, depth):
        # depth: [B, T, H, W, C]，假设输入为归一化后的深度图像（0~1）
        depth = depth.clone()
        B, T, H, W, C = depth.shape

        # 遮挡噪声
        if torch.rand(1) < self.prob_mask:
            for b in range(B):
                x0, y0 = np.random.randint(0, W // 2), np.random.randint(0, H // 2)
                w, h = np.random.randint(5, 10), np.random.randint(5, 10)
                depth[b, :, y0:y0 + h, x0:x0 + w, :] = 0.0

        # 高斯模糊
        if torch.rand(1) < self.prob_blur:
            from torchvision.transforms.functional import gaussian_blur
            depth = depth.view(-1, H, W)
            depth = gaussian_blur(depth, kernel_size=3)
            depth = depth.view(B, T, H, W, 1)

        # 白噪声扰动
        if torch.rand(1) < self.prob_noise:
            noise = 0.02 * torch.randn_like(depth)
            depth = torch.clamp(depth + noise, -0.5, 0.5)

        return depth


class WorldModel(nn.Module):
    def __init__(self, config, obs_shape, use_camera):
        super(WorldModel, self).__init__()
        # self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self.device = self._config.device
        
        self.encoder = networks.MultiEncoder(obs_shape, **config.encoder, use_camera=use_camera)
        self.embed_size = self.encoder.outdim
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
        )
        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, obs_shape, **config.decoder, use_camera=use_camera
        )
        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )
        # self.heads["cont"] = networks.MLP(
        #     feat_size,
        #     (),
        #     config.cont_head["layers"],
        #     config.units,
        #     config.act,
        #     config.norm,
        #     dist="binary",
        #     outscale=config.cont_head["outscale"],
        #     device=config.device,
        #     name="Cont",
        # )
        for name in config.grad_heads:
            assert name in self.heads, name
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        # other losses are scaled by 1.0.
        # can set different scale for terms in decoder here
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            image = 1.0,
            # clean_prop = 0,
            # cont=config.cont_head["loss_scale"],
        )
        self.byol_augmentor = BYOLAugmentor()

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)

                # === BYOL 增加部分：生成两种扰动后的图像视图 ===
        # image_clean = data['image']  # 原始深度图像
        # image_aug1 = self.byol_augmentor(image_clean)  # 增强视图1
        # image_aug2 = self.byol_augmentor(image_clean)  # 增强视图2
        # data_aug1 = data.copy()
        # data_aug2 = data.copy()
        # data_aug1["image"] = image_aug1 #clean input
        # data["image"] = image_aug1
        
        # data_aug2["image"] = image_aug2
            #--------------------------
        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                # === BYOL 增加部分：编码两个扰动视图并计算损失 ===
                # embed1 = self.encoder(data_aug1)
                # embed1 = self.encoder(data)
                # embed2 = self.encoder(data_aug2)
                # byol_loss = 2 - 2 * torch.nn.functional.cosine_similarity(embed1, embed2, dim=-1).mean()
                #---------------------

                embed = self.encoder(data)
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                # # ---------------------
                # lambda_byol = self._config.lambda_byol if hasattr(self._config, 'lambda_byol') else 1.0
                # model_loss = sum(scaled.values()) + kl_loss + lambda_byol * byol_loss
                # ---------------------
                
                model_loss = sum(scaled.values()) + kl_loss 
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        # metrics["byol_loss"] = to_np(byol_loss)  # === BYOL 增加部分：记录损失 ===
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    # this function is called during both rollout and training
    def preprocess(self, obs):
        # obs = obs.copy()
        # obs["image"] = torch.Tensor(obs["image"]) / 255.0

        # discount in obs seems useless
        # if "discount" in obs:
        #     obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            # obs["discount"] = torch.Tensor(obs["discount"]).unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        # assert "is_terminal" in obs
        # obs["cont"] = torch.Tensor(1.0 - obs["is_terminal"]).unsqueeze(-1)
        obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6]
        model = model
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model], 2)
