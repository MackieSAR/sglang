#!/usr/bin/env python3
"""Inspect MXFP8 weights and UE8M0 scales in a Hugging Face checkpoint.

Example:
  python scripts/inspect_hf_mxfp8_scales.py /path/to/Qwen3-8B-MXFP8 \
    --match model.layers.0.mlp.down_proj --preview
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

torch = None
safe_open = None


SCALE_SUFFIXES = (
    ".weight_scale",
    ".weight_scale_inv",
    ".w13_weight_scale",
    ".w2_weight_scale",
    ".w13_weight_scale_inv",
    ".w2_weight_scale_inv",
)


def find_safetensor_files(model_path: Path) -> List[Path]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        return sorted({model_path / name for name in index["weight_map"].values()})

    files = sorted(model_path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors files found under {model_path}")
    return files


def build_key_to_file(files: Iterable[Path]) -> Dict[str, Path]:
    key_to_file: Dict[str, Path] = {}
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as f:
            for key in f.keys():
                key_to_file[key] = file
    return key_to_file


def load_tensor(key_to_file: Dict[str, Path], key: str) -> torch.Tensor:
    file = key_to_file[key]
    with safe_open(file, framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def ue8m0_u8_to_fp32(scale_u8: torch.Tensor) -> torch.Tensor:
    """Decode UE8M0 raw exponent bytes to float32 scales.

    UE8M0 stores the 8-bit exponent field used by FP32. So 127 -> 1.0,
    126 -> 0.5, 128 -> 2.0.
    """
    if scale_u8.dtype != torch.uint8:
        raise TypeError(f"expected uint8, got {scale_u8.dtype}")
    bits = (scale_u8.to(torch.int32) << 23).contiguous()
    return bits.view(torch.float32)


def unpack_int32_to_u8(packed: torch.Tensor) -> torch.Tensor:
    """Unpack little-endian int32 scale packing into four uint8 values."""
    if packed.dtype != torch.int32:
        raise TypeError(f"expected int32, got {packed.dtype}")
    return packed.contiguous().view(torch.uint8).view(*packed.shape[:-1], packed.shape[-1] * 4)


def decode_scale(scale: torch.Tensor) -> Tuple[torch.Tensor, str]:
    if scale.dtype == torch.uint8:
        return ue8m0_u8_to_fp32(scale), "uint8 UE8M0"
    if scale.dtype == torch.int32:
        scale_u8 = unpack_int32_to_u8(scale)
        return ue8m0_u8_to_fp32(scale_u8), "int32 packed UE8M0"
    if scale.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return scale.float(), "float scale"
    raise TypeError(f"unsupported scale dtype {scale.dtype}")


def tensor_stats(x: torch.Tensor) -> str:
    if x.numel() == 0:
        return "empty"
    if x.dtype.is_floating_point:
        xf = x.float()
        return (
            f"min={xf.min().item():.8g} max={xf.max().item():.8g} "
            f"mean={xf.mean().item():.8g}"
        )
    xi = x.to(torch.int64)
    return f"min={xi.min().item()} max={xi.max().item()}"


def scale_suffix_base(scale_key: str) -> Optional[Tuple[str, str]]:
    for suffix in SCALE_SUFFIXES:
        if scale_key.endswith(suffix):
            return scale_key[: -len(suffix)], suffix
    return None


def find_weight_for_scale(keys: set[str], scale_key: str) -> Optional[str]:
    parsed = scale_suffix_base(scale_key)
    if parsed is None:
        return None
    base, suffix = parsed

    candidates = [base + ".weight"]
    if suffix.startswith(".w13_"):
        candidates.append(base + ".w13_weight")
        candidates.append(base + ".gate_up_proj.weight")
    if suffix.startswith(".w2_"):
        candidates.append(base + ".w2_weight")
        candidates.append(base + ".down_proj.weight")

    for candidate in candidates:
        if candidate in keys:
            return candidate
    return None


def preview_dequant(weight: torch.Tensor, scale_fp32: torch.Tensor, group_size: int) -> None:
    if weight.dim() != 2:
        print(f"  preview: skip, expected 2D weight, got {tuple(weight.shape)}")
        return
    if weight.shape[1] % group_size != 0:
        print(
            f"  preview: skip, K={weight.shape[1]} is not divisible by group_size={group_size}"
        )
        return

    rows = min(2, weight.shape[0])
    groups = min(4, weight.shape[1] // group_size)
    cols = groups * group_size

    w = weight[:rows, :cols].float().view(rows, groups, group_size)

    # Common MXFP8 saved layout is (N, K//32). Some runtimes pack or align scales;
    # the first dimensions are enough for a preview when they match.
    if scale_fp32.dim() != 2 or scale_fp32.shape[0] < rows or scale_fp32.shape[1] < groups:
        print(
            "  preview: skip, scale shape does not look like "
            f"(N, K//{group_size}); decoded_scale_shape={tuple(scale_fp32.shape)}"
        )
        return

    s = scale_fp32[:rows, :groups].view(rows, groups, 1)
    dq = (w * s).reshape(rows, cols)
    print(f"  preview dequant[{rows}x{cols}]: {tensor_stats(dq)}")
    print(f"  first row first {min(16, cols)} dq values:")
    print("   ", dq[0, : min(16, cols)].tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path, help="HF checkpoint directory")
    parser.add_argument("--match", default="", help="Only show tensor names containing this text")
    parser.add_argument("--limit", type=int, default=20, help="Max number of scales to print")
    parser.add_argument("--group-size", type=int, default=32, help="MXFP8 group size")
    parser.add_argument("--preview", action="store_true", help="Preview dequantized values")
    parser.add_argument("--list-all", action="store_true", help="List all tensor keys and exit")
    args = parser.parse_args()

    global safe_open, torch
    try:
        import torch as torch_import
        from safetensors import safe_open as safe_open_import
    except ImportError as exc:
        raise SystemExit(
            "This script needs torch and safetensors. Run it inside the same env "
            "you use for llmcompressor, or install safetensors there."
        ) from exc

    torch = torch_import
    safe_open = safe_open_import

    files = find_safetensor_files(args.model_path)
    key_to_file = build_key_to_file(files)
    keys = set(key_to_file)

    if args.list_all:
        for key in sorted(keys):
            print(key)
        return

    scale_keys = [
        key
        for key in sorted(keys)
        if any(key.endswith(suffix) for suffix in SCALE_SUFFIXES)
        and (not args.match or args.match in key)
    ]

    print(f"model_path: {args.model_path}")
    print(f"safetensors: {len(files)} files, tensors: {len(keys)}")
    print(f"scale tensors matched: {len(scale_keys)}")

    for i, scale_key in enumerate(scale_keys[: args.limit], start=1):
        scale = load_tensor(key_to_file, scale_key)
        scale_fp32, scale_kind = decode_scale(scale)
        weight_key = find_weight_for_scale(keys, scale_key)

        print(f"\n[{i}] {scale_key}")
        print(f"  file: {key_to_file[scale_key].name}")
        print(f"  raw scale: shape={tuple(scale.shape)} dtype={scale.dtype} {tensor_stats(scale)}")
        print(
            f"  decoded scale: kind={scale_kind} shape={tuple(scale_fp32.shape)} "
            f"dtype={scale_fp32.dtype} {tensor_stats(scale_fp32)}"
        )

        if weight_key is None:
            print("  weight: not found by suffix heuristic")
            continue

        weight = load_tensor(key_to_file, weight_key)
        print(
            f"  weight: {weight_key} shape={tuple(weight.shape)} dtype={weight.dtype} "
            f"{tensor_stats(weight)}"
        )
        if weight.dim() == 2:
            expected = (weight.shape[0], weight.shape[1] // args.group_size)
            print(f"  expected MXFP8 scale shape for group_size={args.group_size}: {expected}")

        if args.preview:
            preview_dequant(weight, scale_fp32, args.group_size)

    if len(scale_keys) > args.limit:
        print(f"\n... omitted {len(scale_keys) - args.limit} more scale tensors")


if __name__ == "__main__":
    main()
