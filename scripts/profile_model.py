import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.dmcnet_polyp import DMCNetPolyp2d
from src.models.mobilenetv2_unet import MobileNetV2UNet
from src.models.polyp_mfc_net import PolypMFCNet
from src.models.pranet import PraNet
from src.models.pure_unet_2d import PureUNet2D
from src.models.unet import UNet
from src.models.unetpp import UNetPlusPlus2D


MODEL_DEFAULTS = {
    "unet_light": {
        "class": UNet,
        "config": "configs/unet_kvasir_cvc.yaml",
        "checkpoint": "checkpoints/unet_best.pth",
        "output": "results/tables/unet_light_profile.csv",
    },
    "pureunet": {
        "class": PureUNet2D,
        "config": "configs/b01_pureunet_kvasir_cvc.yaml",
        "checkpoint": "checkpoints/pureunet_best.pth",
        "output": "results/tables/pureunet_profile.csv",
    },
    "unetpp": {
        "class": UNetPlusPlus2D,
        "config": "configs/b02_unetpp_kvasir_cvc.yaml",
        "checkpoint": "checkpoints/unetpp_best.pth",
        "output": "results/tables/unetpp_profile.csv",
    },
    "dmcnetfull": {
        "class": DMCNetPolyp2d,
        "config": "configs/b10a_dmcnetfull_kvasir_cvc.yaml",
        "checkpoint": "checkpoints/dmcnetfull_best.pth",
        "output": "results/tables/dmcnetfull_profile.csv",
    },
    "pranet": {
        "class": PraNet,
        "config": "configs/b04_pranet_kvasir_cvc.yaml",
        "checkpoint": "checkpoints/pranet_best.pth",
        "output": "results/tables/pranet_profile.csv",
    },
    "mobilenetv2unet": {
        "class": MobileNetV2UNet,
        "config": "configs/b05_mobilenetv2unet_kvasir_cvc.yaml",
        "checkpoint": "checkpoints/b05_mobilenetv2unet_kvasircvc_jointtrain_best.pth",
        "output": "results/tables/mobilenetv2unet_profile.csv",
    },
    "polypmfc": {
        "class": PolypMFCNet,
        "config": "configs/a3_polypmfc_msm_bcf_kvasir_cvc.yaml",
        "checkpoint": "checkpoints/a3_polypmfc_msm_bcf_kvasircvc_jointtrain_best.pth",
        "output": "results/tables/polypmfc_profile.csv",
    },
}


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_profile_model(model_name, config):
    model_cfg = config.get("model", {})
    common = {
        "in_channels": model_cfg.get("in_channels", 3),
        "out_channels": model_cfg.get("out_channels", 1),
    }
    if model_name == "unet_light":
        return UNet(**common)
    if model_name == "pureunet":
        return PureUNet2D(**common, features=tuple(model_cfg.get("features", [32, 64, 128, 256, 512, 512])))
    if model_name == "unetpp":
        return UNetPlusPlus2D(**common, features=tuple(model_cfg.get("features", [32, 64, 128, 256, 512, 512])))
    if model_name == "dmcnetfull":
        return DMCNetPolyp2d(
            **common,
            features=tuple(model_cfg.get("features", [32, 64, 128, 256, 512, 512])),
            conv5_mode=model_cfg.get("conv5_mode", "depthwise"),
        )
    if model_name == "pranet":
        return PraNet(
            **common,
            channel=model_cfg.get("channel", 32),
            pretrained=model_cfg.get("pretrained", False),
        )
    if model_name == "mobilenetv2unet":
        return MobileNetV2UNet(
            **common,
            pretrained=model_cfg.get("pretrained", True),
            decoder_channels=tuple(model_cfg.get("decoder_channels", [256, 128, 64, 32])),
        )
    if model_name == "polypmfc":
        return PolypMFCNet(
            **common,
            pretrained=model_cfg.get("pretrained", True),
            use_msm=model_cfg.get("use_msm", True),
            use_bcf=model_cfg.get("use_bcf", True),
            decoder_channels=tuple(model_cfg.get("decoder_channels", [256, 128, 64, 64])),
        )
    raise ValueError(f"Unknown model '{model_name}'. Use one of: {list(MODEL_DEFAULTS)}")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def measure_fps(model, dummy_input, device, warmup=20, repeats=100):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return dummy_input.size(0) * repeats / elapsed


def compute_macs(model, dummy_input):
    try:
        from thop import profile
    except ImportError as exc:
        raise ImportError("thop is required for MACs/FLOPs. Install it with: pip install thop") from exc
    macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
    return macs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_DEFAULTS.keys()))
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    defaults = MODEL_DEFAULTS[args.model]
    config_path = ROOT / (args.config or defaults["config"])
    checkpoint_path = ROOT / (args.checkpoint or defaults["checkpoint"])
    output_path = ROOT / (args.output or defaults["output"])

    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = config.get("image_size", 352)

    model = build_profile_model(args.model, config).to(device)
    checkpoint_used = "not found; profiled initialized model"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        checkpoint_used = str(checkpoint_path.relative_to(ROOT))

    dummy_input = torch.randn(args.batch_size, 3, image_size, image_size, device=device)
    params = count_params(model)
    macs = compute_macs(model, dummy_input)
    flops = macs * 2
    fps = measure_fps(model, dummy_input, device, warmup=args.warmup, repeats=args.repeats)

    row = {
        "model": args.model,
        "checkpoint": checkpoint_used,
        "device": str(device),
        "image_size": image_size,
        "batch_size": args.batch_size,
        "params": params,
        "params_m": params / 1e6,
        "macs": macs,
        "macs_g": macs / 1e9,
        "flops": flops,
        "flops_g": flops / 1e9,
        "fps": fps,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    print(f"Model: {row['model']}")
    print(f"Checkpoint: {row['checkpoint']}")
    print(f"Device: {row['device']}")
    print(f"Input size: {image_size}x{image_size}, batch_size={args.batch_size}")
    print(f"Params: {row['params_m']:.3f} M")
    print(f"MACs: {row['macs_g']:.3f} G")
    print(f"FLOPs: {row['flops_g']:.3f} G")
    print(f"FPS: {row['fps']:.2f}")
    print(f"Saved profile: {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
