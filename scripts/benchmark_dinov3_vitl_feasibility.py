"""Resource-bounded ViT-L DINOv3 feasibility smoke test.

This script compares official web-pretrained and satellite-pretrained DINOv3
ViT-L/16 checkpoints with the same architecture size. It runs only forward-pass
and optional one-step backward smoke tests; it does not train or update weights.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from transformers import AutoConfig, AutoImageProcessor, AutoModel

from forestpersons.dinov3_detector import (
    DINOv3FCOSHead,
    choose_device,
    image_to_tensor,
    resize_to_patch_grid,
)
from forestpersons.dinov3_lora import LoRAConfig, apply_lora_adapters, trainable_parameter_summary


MODELS = {
    "web_vitl16_lvd1689m": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "sat_vitl16_sat493m": "facebook/dinov3-vitl16-pretrain-sat493m",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("data/images"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--initial-short-side", type=int, default=224)
    parser.add_argument("--benchmark-short-side", type=int, default=336)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--run-backward", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser.parse_args()


def load_manifest(path: Path, limit: int) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["images"] if isinstance(data, dict) else data
    return items[:limit]


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def mps_memory() -> dict[str, int] | None:
    if not torch.backends.mps.is_available():
        return None
    return {
        "current_allocated": int(torch.mps.current_allocated_memory()),
        "driver_allocated": int(torch.mps.driver_allocated_memory()),
    }


def clear_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def extract_features(
    image: Image.Image,
    processor: Any,
    model: nn.Module,
    device: torch.device,
    short_side: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    resized, info = resize_to_patch_grid(image, short_side=short_side)
    pixel_values = image_to_tensor(resized, processor, device)
    outputs = model(pixel_values=pixel_values)
    tokens = outputs.last_hidden_state[0]
    num_patches = info.grid_height * info.grid_width
    patch_tokens = tokens[-num_patches:].reshape(info.grid_height, info.grid_width, -1)
    features = F.normalize(patch_tokens.float(), dim=-1)
    return features, {
        "original_size": [info.original_width, info.original_height],
        "resized_size": [info.resized_width, info.resized_height],
        "grid_size": [info.grid_width, info.grid_height],
        "feature_shape": list(features.shape),
    }


def model_card_estimate(config: Any) -> dict[str, Any]:
    hidden = int(config.hidden_size)
    layers = int(config.num_hidden_layers)
    lora_rank = 4
    lora_qv_params = layers * 2 * 2 * hidden * lora_rank
    head = DINOv3FCOSHead(input_dim=hidden)
    return {
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "num_attention_heads": int(config.num_attention_heads),
        "patch_size": int(config.patch_size),
        "fcos_head_parameters": int(sum(p.numel() for p in head.parameters())),
        "rank4_qv_lora_parameter_estimate": int(lora_qv_params),
    }


def run_forward_rows(
    items: list[dict[str, Any]],
    image_root: Path,
    processor: Any,
    model: nn.Module,
    device: torch.device,
    short_side: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    for index, item in enumerate(items, 1):
        image = Image.open(image_root / item["file_name"]).convert("RGB")
        start = time.perf_counter()
        with torch.inference_mode():
            features, info = extract_features(image, processor, model, device, short_side)
        synchronize(device)
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "index": index,
                "file_name": item["file_name"],
                "seconds": elapsed,
                "mps_memory_bytes": mps_memory(),
                **info,
            }
        )
        print(f"{index:03d}/{len(items)} {elapsed:.3f}s {item['file_name']}")
        del features
    return rows


def run_backward_smoke(
    item: dict[str, Any],
    image_root: Path,
    model_id: str,
    device: torch.device,
    short_side: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    clear_memory(device)
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    config = LoRAConfig(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_suffixes=("attention.q_proj", "attention.v_proj"),
    )
    targets = apply_lora_adapters(model, config)
    model.to(device)
    model.train()
    head = DINOv3FCOSHead(input_dim=int(model.config.hidden_size)).to(device)
    head.train()
    image = Image.open(image_root / item["file_name"]).convert("RGB")
    start = time.perf_counter()
    features, info = extract_features(image, processor, model, device, short_side)
    class_logits, raw_box, centerness_logits = head(features)
    loss = (
        class_logits.float().square().mean()
        + 0.01 * raw_box.float().square().mean()
        + centerness_logits.float().square().mean()
    )
    loss.backward()
    synchronize(device)
    elapsed = time.perf_counter() - start
    lora_grads = [
        parameter.grad.detach().abs().max().item()
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    head_grads = [
        parameter.grad.detach().abs().max().item()
        for parameter in head.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    result = {
        "short_side": short_side,
        "seconds": elapsed,
        "loss": float(loss.detach().cpu()),
        "feature_shape": info["feature_shape"],
        "lora_target_modules": targets,
        "parameter_summary": trainable_parameter_summary(model, head),
        "lora_grad_tensors": len(lora_grads),
        "lora_nonzero_grad_tensors": int(sum(value > 0 for value in lora_grads)),
        "head_grad_tensors": len(head_grads),
        "head_nonzero_grad_tensors": int(sum(value > 0 for value in head_grads)),
        "mps_memory_bytes": mps_memory(),
    }
    del model, head, features, loss
    clear_memory(device)
    return result


def run_model(
    name: str,
    model_id: str,
    items: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    started = time.perf_counter()
    clear_memory(device)
    config = AutoConfig.from_pretrained(model_id)
    estimate = model_card_estimate(config)
    processor = AutoImageProcessor.from_pretrained(model_id)
    load_start = time.perf_counter()
    model = AutoModel.from_pretrained(model_id)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    synchronize(device)
    load_seconds = time.perf_counter() - load_start
    print(f"Loaded {name} in {load_seconds:.2f}s")

    initial_rows = run_forward_rows(
        items[:1],
        args.image_root,
        processor,
        model,
        device,
        args.initial_short_side,
    )
    benchmark_rows = run_forward_rows(
        items,
        args.image_root,
        processor,
        model,
        device,
        args.benchmark_short_side,
    )
    del model
    clear_memory(device)
    backward = None
    if args.run_backward:
        backward = run_backward_smoke(
            items[0],
            args.image_root,
            model_id,
            device,
            args.initial_short_side,
            args,
        )

    return {
        "name": name,
        "model_id": model_id,
        "config": estimate,
        "load_seconds": load_seconds,
        "initial_forward": initial_rows,
        "benchmark_forward": benchmark_rows,
        "benchmark_seconds_total": float(sum(row["seconds"] for row in benchmark_rows)),
        "benchmark_seconds_per_image": float(
            sum(row["seconds"] for row in benchmark_rows) / len(benchmark_rows)
        ),
        "backward_smoke": backward,
        "seconds_total": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    items = load_manifest(args.manifest, args.limit)
    summary = {
        "settings": {
            "manifest": str(args.manifest),
            "image_root": str(args.image_root),
            "output_dir": str(args.output_dir),
            "device": str(device),
            "initial_short_side": args.initial_short_side,
            "benchmark_short_side": args.benchmark_short_side,
            "limit": len(items),
            "run_backward": args.run_backward,
            "model_weight_bytes_each": 1212559808,
            "model_weight_gib_each": 1212559808 / (1024**3),
            "both_model_weight_gib": 2 * 1212559808 / (1024**3),
        },
        "models": {},
    }
    for name, model_id in MODELS.items():
        try:
            result = run_model(name, model_id, items, args, device)
            summary["models"][name] = result
        except Exception as exc:  # noqa: BLE001 - report feasibility failures
            summary["models"][name] = {
                "name": name,
                "model_id": model_id,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
        finally:
            (args.output_dir / "summary.json").write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )
    print(json.dumps(summary, indent=2))
    print(f"Saved summary: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
