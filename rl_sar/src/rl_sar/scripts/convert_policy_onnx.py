import torch
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", type=str, required=True)
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    # 1. 加载 TorchScript
    model = torch.jit.load(args.pt, map_location="cpu")
    model.eval()

    # 2. dummy input: (batch, 6)
    dummy_input = torch.randn(args.batch, 6)

    # 3. 导出 ONNX（⚠️ 关闭 dynamo）
    torch.onnx.export(
        model,
        dummy_input,
        args.onnx,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["joint_state"],
        output_names=["tau_est"],
        dynamic_axes={
            "joint_state": {0: "batch"},
            "tau_est": {0: "batch"},
        },
        dynamo=False,   # ✅ 关键
    )

    print(f"✅ Export success: {args.onnx}")

if __name__ == "__main__":
    main()
