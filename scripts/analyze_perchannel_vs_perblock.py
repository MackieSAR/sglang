#!/usr/bin/env python3
"""Analyze whether a model layer leans toward per-channel or per-block quantization.

This script does three things for each Linear layer:
1. Weight-only statistics:
   - how different row scales are from each other
   - how uneven block scales are inside the same matrix
   - how "spiky" each row looks
2. Optional activation statistics from a small calibration set.
3. Fake quantization comparison:
   - per-channel FP8 E4M3 fake quantization:
     weight per output channel, input activation per token
   - per-block FP8 E4M3 fake quantization:
     weight per block, input activation per 1x128 block
   - compare weight error and layer output error

The most important signal is output error from fake quantization. The other
metrics help explain why a layer leans one way or the other.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


EPS = 1.0e-12
DEFAULT_PROMPTS = [
    "Explain in three sentences why quantization can change model accuracy.",
    "Write a short summary of how attention works in a transformer.",
    "If x + 2y = 9 and 3x - y = 8, solve for x and y.",
]


@dataclass
class LayerReport:
    name: str
    shape: str
    params_m: float
    samples: int
    between_channel_scale_spread: float
    within_block_scale_spread: float
    channel_peakiness_p95: float
    activation_channel_spread: Optional[float]
    activation_outlier_frac: Optional[float]
    weight_rel_mse_per_channel: float
    weight_rel_mse_per_block: float
    weight_only_output_rel_mse_per_channel: Optional[float]
    weight_only_output_rel_mse_per_block: Optional[float]
    output_rel_mse_per_channel: Optional[float]
    output_rel_mse_per_block: Optional[float]
    output_cosine_per_channel: Optional[float]
    output_cosine_per_block: Optional[float]
    winner: str
    confidence: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Layer-wise analysis for per-channel vs per-block quantization."
    )
    parser.add_argument("--model-path", required=True, help="HF model path.")
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Tokenizer path. Defaults to --model-path.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Calibration prompt. Can be repeated.",
    )
    parser.add_argument(
        "--prompts-file",
        default=None,
        help="JSONL file for calibration prompts.",
    )
    parser.add_argument(
        "--skip-activations",
        dest="skip_activations",
        action="store_true",
        default=False,
        help="Only run weight analysis and fake quant on weights.",
    )
    parser.add_argument(
        "--with-activations",
        dest="skip_activations",
        action="store_false",
        help="Collect activation samples and include layer output error analysis. This is the default.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=8,
        help="Use at most this many calibration prompts.",
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=256,
        help="Max tokens per prompt for calibration forward passes.",
    )
    parser.add_argument(
        "--samples-per-layer",
        type=int,
        default=96,
        help="How many activation vectors to keep per Linear layer.",
    )
    parser.add_argument(
        "--block-size",
        nargs=2,
        type=int,
        default=[128, 128],
        metavar=("ROWS", "COLS"),
        help="2D block shape for per-block quantization.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Regex of layer names to include. Repeatable.",
    )
    parser.add_argument(
        "--ignore",
        action="append",
        default=["lm_head"],
        help="Substring/glob or regex prefixed with re: to skip.",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Analyze at most this many matching layers.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for loading the model.",
    )
    parser.add_argument(
        "--analysis-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help=(
            "Device for per-layer fake-quant analysis. Defaults to cuda when available, "
            "while the full model stays on --device."
        ),
    )
    parser.add_argument(
        "--activation-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help=(
            "Device for calibration forward passes. Activation samples are copied to CPU "
            "inside hooks after they are collected."
        ),
    )
    parser.add_argument(
        "--keep-model-on-device",
        action="store_true",
        help="Keep the full model on --device during per-layer analysis.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model loading dtype.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trust remote model code. Enabled by default; use --no-trust-remote-code to disable.",
    )
    parser.add_argument(
        "--fp8-format",
        default="e4m3",
        choices=["e4m3"],
        help="FP8 format used for fake quantization.",
    )
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--min-output-mse",
        type=float,
        default=1.0e-4,
        help=(
            "Minimum max(w8a8_pc, w8a8_pb) to treat an output-error "
            "difference as meaningful."
        ),
    )
    return parser.parse_args()


def matches_any(name: str, patterns: Iterable[str]) -> bool:
    for pattern in patterns:
        if pattern.startswith("re:"):
            if re.search(pattern[3:], name):
                return True
        elif pattern in name or fnmatch.fnmatch(name, pattern):
            return True
    return False


def should_keep_layer(name: str, module: nn.Module, args: argparse.Namespace) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    if args.include and not any(re.search(pattern, name) for pattern in args.include):
        return False
    if matches_any(name, args.ignore):
        return False
    return True


def resolve_dtype(name: str) -> Optional[torch.dtype]:
    if name == "auto":
        return None
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def percentile(x: torch.Tensor, q: float) -> float:
    if x.numel() == 0:
        return 0.0
    q = min(max(q, 0.0), 1.0)
    return float(torch.quantile(x.float(), q).item())


def safe_ratio(a: float, b: float) -> float:
    return float(a / max(b, EPS))


def relative_mse(ref: torch.Tensor, test: torch.Tensor) -> float:
    ref = ref.float()
    test = test.float()
    num = torch.mean((ref - test) ** 2)
    den = torch.mean(ref**2) + EPS
    return float((num / den).item())


def mean_cosine_similarity(ref: torch.Tensor, test: torch.Tensor) -> float:
    ref = ref.float()
    test = test.float()
    ref_norm = ref.norm(dim=-1)
    test_norm = test.norm(dim=-1)
    mask = (ref_norm > 0) & (test_norm > 0)
    if not torch.any(mask):
        return 1.0
    cos = torch.sum(ref[mask] * test[mask], dim=-1) / (ref_norm[mask] * test_norm[mask] + EPS)
    return float(cos.mean().item())


def get_fp8_dtype() -> torch.dtype:
    if getattr(torch.version, "hip", None):
        if hasattr(torch, "float8_e4m3fnuz"):
            return torch.float8_e4m3fnuz
    if hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn
    raise RuntimeError("Current PyTorch build does not expose torch.float8_e4m3fn/fnuz.")


def is_cuda_device(device: str) -> bool:
    return device == "cuda" or device.startswith("cuda:")


def scaled_fp8_e4m3_fake_quant(weight: torch.Tensor, scales: torch.Tensor, fp8_dtype: torch.dtype) -> torch.Tensor:
    weight = weight.float()
    scales = scales.float().clamp_min(EPS)
    finfo = torch.finfo(fp8_dtype)
    scaled = torch.clamp(weight / scales, min=finfo.min, max=finfo.max)
    return scaled.to(fp8_dtype).to(torch.float32) * scales


def fake_quant_per_channel(weight: torch.Tensor, fp8_dtype: torch.dtype) -> torch.Tensor:
    fp8_max = torch.finfo(fp8_dtype).max
    scales = weight.abs().amax(dim=1, keepdim=True).clamp_min(EPS) / fp8_max
    return scaled_fp8_e4m3_fake_quant(weight, scales, fp8_dtype)


def fake_quant_activation_per_token(x: torch.Tensor, fp8_dtype: torch.dtype) -> torch.Tensor:
    fp8_max = torch.finfo(fp8_dtype).max
    scales = x.abs().amax(dim=1, keepdim=True).clamp_min(EPS) / fp8_max
    return scaled_fp8_e4m3_fake_quant(x, scales, fp8_dtype)


def fake_quant_per_block(
    weight: torch.Tensor, fp8_dtype: torch.dtype, block_rows: int, block_cols: int
) -> torch.Tensor:
    weight = weight.float()
    rows, cols = weight.shape
    pad_rows = math.ceil(rows / block_rows) * block_rows
    pad_cols = math.ceil(cols / block_cols) * block_cols
    padded = torch.zeros((pad_rows, pad_cols), dtype=weight.dtype, device=weight.device)
    padded[:rows, :cols] = weight

    view = padded.view(pad_rows // block_rows, block_rows, pad_cols // block_cols, block_cols)
    view = view.permute(0, 2, 1, 3).contiguous()
    fp8_max = torch.finfo(fp8_dtype).max
    scales = view.abs().amax(dim=(2, 3), keepdim=True).clamp_min(EPS) / fp8_max
    dequant = scaled_fp8_e4m3_fake_quant(view, scales, fp8_dtype)
    dequant = dequant.permute(0, 2, 1, 3).contiguous().view(pad_rows, pad_cols)
    return dequant[:rows, :cols]


def iter_prompts(args: argparse.Namespace) -> List[str]:
    prompts = list(args.prompt)
    if args.prompts_file:
        path = Path(args.prompts_file)
        for line in path.read_text().splitlines():
            if line.strip():
                prompts.append(extract_prompt(json.loads(line)))

    if not prompts:
        prompts = list(DEFAULT_PROMPTS)
    return prompts[: args.max_prompts]


def extract_prompt(item) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("text", "prompt", "input", "query"):
            if key in item and isinstance(item[key], str):
                return item[key]
        if "messages" in item:
            parts = []
            for msg in item["messages"]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(f"{role}: {content}")
            return "\n".join(parts)
    raise ValueError(f"Cannot extract prompt from {item!r}")


def collect_activation_samples(
    model: nn.Module,
    tokenizer,
    layer_names: List[str],
    layers: Dict[str, nn.Linear],
    args: argparse.Namespace,
    activation_device: str,
) -> Dict[str, torch.Tensor]:
    samples: Dict[str, List[torch.Tensor]] = {name: [] for name in layer_names}
    sample_counts: Dict[str, int] = {name: 0 for name in layer_names}
    handles = []

    def make_hook(layer_name: str):
        def hook(_module, module_inputs):
            if sample_counts[layer_name] >= args.samples_per_layer:
                return
            if not module_inputs:
                return
            x = module_inputs[0]
            if not torch.is_tensor(x):
                return
            if x.ndim < 2:
                return
            x = x.detach()
            x = x.reshape(-1, x.shape[-1])
            if x.shape[-1] != layers[layer_name].weight.shape[1]:
                return

            remain = args.samples_per_layer - sample_counts[layer_name]
            if x.shape[0] > remain:
                indices = torch.randperm(x.shape[0], device=x.device)[:remain]
                x = x.index_select(0, indices)
            samples[layer_name].append(x.float().cpu())
            sample_counts[layer_name] += x.shape[0]

        return hook

    for name in layer_names:
        handles.append(layers[name].register_forward_pre_hook(make_hook(name)))

    prompts = iter_prompts(args)
    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_length,
            )
            encoded = {k: v.to(activation_device) for k, v in encoded.items()}
            model(**encoded)

    for handle in handles:
        handle.remove()

    merged: Dict[str, torch.Tensor] = {}
    for name, chunks in samples.items():
        if chunks:
            merged[name] = torch.cat(chunks, dim=0)[: args.samples_per_layer]
    return merged


def block_scales(weight: torch.Tensor, block_rows: int, block_cols: int) -> torch.Tensor:
    weight = weight.float()
    rows, cols = weight.shape
    pad_rows = math.ceil(rows / block_rows) * block_rows
    pad_cols = math.ceil(cols / block_cols) * block_cols
    padded = torch.zeros((pad_rows, pad_cols), dtype=weight.dtype, device=weight.device)
    padded[:rows, :cols] = weight
    view = padded.view(pad_rows // block_rows, block_rows, pad_cols // block_cols, block_cols)
    view = view.permute(0, 2, 1, 3).contiguous()
    return view.abs().amax(dim=(2, 3)).flatten()


def weight_metrics(weight: torch.Tensor, block_rows: int, block_cols: int) -> Tuple[float, float, float]:
    weight = weight.float()
    row_rms = torch.sqrt(torch.mean(weight**2, dim=1) + EPS)
    row_max = weight.abs().amax(dim=1)
    between_channel_scale_spread = safe_ratio(percentile(row_rms, 0.90), percentile(row_rms, 0.10))
    channel_peakiness = (row_max / row_rms.clamp_min(EPS)).float()
    channel_peakiness_p95 = percentile(channel_peakiness, 0.95)
    blk_scales = block_scales(weight, block_rows, block_cols)
    within_block_scale_spread = safe_ratio(percentile(blk_scales, 0.90), percentile(blk_scales, 0.10))
    return between_channel_scale_spread, within_block_scale_spread, channel_peakiness_p95


def activation_metrics(samples: Optional[torch.Tensor]) -> Tuple[Optional[float], Optional[float]]:
    if samples is None or samples.numel() == 0:
        return None, None
    samples = samples.float()
    feat_rms = torch.sqrt(torch.mean(samples**2, dim=0) + EPS)
    channel_spread = safe_ratio(percentile(feat_rms, 0.90), percentile(feat_rms, 0.10))
    global_rms = float(torch.sqrt(torch.mean(samples**2) + EPS).item())
    outlier_frac = float((samples.abs() > 6.0 * global_rms).float().mean().item())
    return channel_spread, outlier_frac


def decide_winner(
    name: str,
    between_channel_scale_spread: float,
    within_block_scale_spread: float,
    channel_peakiness_p95: float,
    weight_rel_mse_per_channel: float,
    weight_rel_mse_per_block: float,
    output_rel_mse_per_channel: Optional[float],
    output_rel_mse_per_block: Optional[float],
    min_output_mse: float,
) -> Tuple[str, str, str]:
    if output_rel_mse_per_channel is not None and output_rel_mse_per_block is not None:
        if max(output_rel_mse_per_channel, output_rel_mse_per_block) < min_output_mse:
            return (
                "close",
                "low",
                "Both FP8 fake-quant output errors are below the reporting threshold, so the observed difference is too small to treat as a meaningful preference.",
            )
        ratio = safe_ratio(output_rel_mse_per_block, output_rel_mse_per_channel)
        if ratio >= 1.35:
            return (
                "per-channel",
                "high",
                "Per-block output error is clearly larger, so this layer is more sensitive to sharing one scale inside a block.",
            )
        if ratio <= 0.74:
            return (
                "per-block",
                "high",
                "Per-block output error is lower, so this layer benefits from finer local scaling inside the matrix.",
            )
        return (
            "close",
            "medium",
            "The two FP8 fake-quant output errors are close. This layer alone does not show a decisive winner, so kernel support and end-to-end validation matter more.",
        )

    weight_ratio = safe_ratio(weight_rel_mse_per_block, weight_rel_mse_per_channel)
    if weight_ratio >= 1.35 or between_channel_scale_spread >= 2.5 or channel_peakiness_p95 >= 8.0:
        return (
            "per-channel",
            "medium",
            "Weight-only signals suggest this layer has uneven channels or sharper row-level spikes, so sharing one scale per block is riskier.",
        )
    if weight_ratio <= 0.74 and within_block_scale_spread >= 1.6:
        return (
            "per-block",
            "medium",
            "Weight-only signals suggest local blocks have more distinct scales, so per-block is more likely to fit the matrix structure.",
        )
    return (
        "close",
        "low",
        "Weight-only signals are not decisive. A small calibration run is recommended before drawing a conclusion.",
    )


def winner_label(value: str) -> str:
    return {
        "per-channel": "leans per-channel",
        "per-block": "leans per-block",
        "close": "close",
    }.get(value, value)


def confidence_label(value: str) -> str:
    return {
        "high": "high",
        "medium": "medium",
        "low": "low",
    }.get(value, value)


def print_metric_legend() -> None:
    print("Metric legend:")
    print("  between_channel_scale_spread: p90(row RMS) / p10(row RMS).")
    print("    Plain meaning: how different the output-channel scales are. Bigger means channels are more uneven, so per-channel often helps.")
    print("  within_block_scale_spread: p90(block max) / p10(block max).")
    print("    Plain meaning: how different local block scales are. Bigger means stronger local structure, so per-block may help.")
    print("  channel_peakiness_p95: p95(row max / row RMS).")
    print("    Plain meaning: whether rows contain sharp spikes. Bigger means a few large values are more likely to hurt shared scales.")
    print("  output_rel_mse_*: FP8 fake-quant output error divided by original output energy.")
    print("    Plain meaning: smaller is better; this is the most important metric.")
    print("    per-channel path uses per-token input activation quantization.")
    print("    per-block path uses 1x128 input activation block quantization.")
    print("    The table also prints wonly_* for the same output error with original activations and quantized weights only.")
    print("  output_cosine_*: average cosine similarity between pre- and post-fake-quant outputs.")
    print("    Plain meaning: closer to 1 is better.")
    print()
    print("Decision logic:")
    print("  When activation samples are available, output error is the primary signal:")
    print("    If both output errors are below --min-output-mse, classify as close.")
    print("    If w8a8_pb / w8a8_pc >= 1.35, classify as leans per-channel.")
    print("    If w8a8_pb / w8a8_pc <= 0.74, classify as leans per-block.")
    print("    Otherwise classify as close, meaning the gap is not large enough to be decisive.")
    print("  When activation samples are not available, fall back to weight-only signals:")
    print("    leans per-channel: w_mse_pb / w_mse_pc >= 1.35, or between_spread >= 2.5, or channel_peakiness_p95 >= 8.0.")
    print("    leans per-block: w_mse_pb / w_mse_pc <= 0.74 and block_spread >= 1.6.")
    print("    Otherwise classify as close.")
    print()


def format_optional(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}g}"


def row_advantage_value(row: LayerReport) -> float:
    left = (
        row.output_rel_mse_per_block
        if row.output_rel_mse_per_block is not None
        else row.weight_rel_mse_per_block
    )
    right = (
        row.output_rel_mse_per_channel
        if row.output_rel_mse_per_channel is not None
        else row.weight_rel_mse_per_channel
    )
    return left - right


def row_max_output_mse(row: LayerReport) -> Optional[float]:
    if row.output_rel_mse_per_channel is None or row.output_rel_mse_per_block is None:
        return None
    return max(row.output_rel_mse_per_channel, row.output_rel_mse_per_block)


def print_report_rows(rows: List[LayerReport], top_k: int) -> None:
    print(
        "  "
        f"{'layer':<42} "
        f"{'decision':<18} "
        f"{'conf':<8} "
        f"{'wonly_pc':>12} "
        f"{'wonly_pb':>12} "
        f"{'w8a8_pc':>12} "
        f"{'w8a8_pb':>12} "
        f"{'w_mse_pc':>12} "
        f"{'w_mse_pb':>12} "
        f"{'ch_spread':>10} "
        f"{'blk_spread':>10}"
    )
    for row in rows[:top_k]:
        print(
            "  "
            f"{row.name:<42.42} "
            f"{winner_label(row.winner):<18.18} "
            f"{confidence_label(row.confidence):<8.8} "
            f"{format_optional(row.weight_only_output_rel_mse_per_channel):>12} "
            f"{format_optional(row.weight_only_output_rel_mse_per_block):>12} "
            f"{format_optional(row.output_rel_mse_per_channel):>12} "
            f"{format_optional(row.output_rel_mse_per_block):>12} "
            f"{row.weight_rel_mse_per_channel:>12.4g} "
            f"{row.weight_rel_mse_per_block:>12.4g} "
            f"{row.between_channel_scale_spread:>10.3g} "
            f"{row.within_block_scale_spread:>10.3g}"
        )


def print_top_table(reports: List[LayerReport], top_k: int, min_output_mse: float) -> None:
    ranked_pc = sorted(
        [row for row in reports if row.winner == "per-channel"],
        key=lambda row: (-row_advantage_value(row), row.name),
    )
    ranked_pb = sorted(
        [row for row in reports if row.winner == "per-block"],
        key=lambda row: (row_advantage_value(row), row.name),
    )
    ranked_close = sorted(
        [
            row
            for row in reports
            if row.winner == "close"
            and (
                row_max_output_mse(row) is None
                or row_max_output_mse(row) >= min_output_mse
            )
        ],
        key=lambda row: (-abs(row_advantage_value(row)), row.name),
    )

    if ranked_pc:
        print("Top layers that lean per-channel:")
        print("  Note: rows are sorted by (w8a8_pb - w8a8_pc) in descending order.")
        print("  Column guide: wonly_* = weight-only output relative MSE, w8a8_* = W8A8 output relative MSE, w_mse_* = weight relative MSE, ch_spread = channel-to-channel scale spread, blk_spread = block-to-block scale spread.")
        print_report_rows(ranked_pc, top_k)
        print()

    if ranked_pb:
        print("Top layers that lean per-block:")
        print("  Note: rows are sorted by (w8a8_pc - w8a8_pb) in descending order.")
        print("  Column guide: wonly_* = weight-only output relative MSE, w8a8_* = W8A8 output relative MSE, w_mse_* = weight relative MSE, ch_spread = channel-to-channel scale spread, blk_spread = block-to-block scale spread.")
        print_report_rows(ranked_pb, top_k)
        print()

    if ranked_close:
        print("Representative close layers:")
        print("  Note: these layers do not cross the decisive threshold; rows are sorted by absolute gap size.")
        print("  Column guide: wonly_* = weight-only output relative MSE, w8a8_* = W8A8 output relative MSE, w_mse_* = weight relative MSE, ch_spread = channel-to-channel scale spread, blk_spread = block-to-block scale spread.")
        print_report_rows(ranked_close, top_k)


def print_table_metric_explanation_zh() -> None:
    print()
    print("表格指标中文说明：")
    print("  w8a8_pc: per-channel weight + per-token activation 下的层输出相对 MSE。越小越好。")
    print("    该路径量化当前 Linear 的输入激活，activation 粒度为 per-token。")
    print("  w8a8_pb: per-block weight + 1x128 activation 下的层输出相对 MSE。越小越好。")
    print("    该路径量化当前 Linear 的输入激活，activation 粒度为 1x128 block。")
    print("  wonly_pc: 只量化 per-channel weight、不量化 activation 的层输出相对 MSE。")
    print("  wonly_pb: 只量化 per-block weight、不量化 activation 的层输出相对 MSE。")
    print("  w_mse_pc: per-channel 下的权重相对 MSE。越小越好。")
    print("  w_mse_pb: per-block 下的权重相对 MSE。越小越好。")
    print("  ch_spread: 通道间尺度离散度，p90(row RMS) / p10(row RMS)。越大表示不同输出通道差异越大。")
    print("  blk_spread: block 间尺度离散度，p90(block max) / p10(block max)。越大表示不同局部 block 差异越大。")
    print("  decision: 当前层更偏向 per-channel、per-block，或者 close。")
    print("  conf: 当前判定的置信度。")
    print()


def main() -> None:
    args = parse_args()
    if args.block_size[0] <= 0 or args.block_size[1] <= 0:
        raise ValueError("--block-size values must be positive.")
    analysis_device = args.analysis_device or args.device
    activation_device = args.activation_device or args.device
    torch.manual_seed(0)
    fp8_dtype = get_fp8_dtype()

    print_metric_legend()
    print(f"Loading model from {args.model_path} on {args.device} ...")
    print(f"Activation collection will run on {activation_device}.")
    print(f"Per-layer fake-quant analysis will run on {analysis_device}.")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
        torch_dtype=resolve_dtype(args.dtype),
    )
    model.to(args.device)
    model.eval()

    named_layers = {
        name: module
        for name, module in model.named_modules()
        if should_keep_layer(name, module, args)
    }
    layer_names = list(named_layers.keys())
    if args.max_layers is not None:
        layer_names = layer_names[: args.max_layers]
    named_layers = {name: named_layers[name] for name in layer_names}

    if not named_layers:
        raise ValueError("No matching Linear layers found.")

    print(f"Matched {len(named_layers)} Linear layers.")

    activation_samples: Dict[str, torch.Tensor] = {}
    if not args.skip_activations:
        tokenizer_path = args.tokenizer_path or args.model_path
        print(f"Collecting activation samples using tokenizer={tokenizer_path} ...")
        if activation_device != args.device:
            print(f"Moving model to {activation_device} for activation collection ...")
            model.to(activation_device)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=args.trust_remote_code,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        activation_samples = collect_activation_samples(
            model=model,
            tokenizer=tokenizer,
            layer_names=layer_names,
            layers=named_layers,
            args=args,
            activation_device=activation_device,
        )
        print(f"Collected activation samples for {len(activation_samples)} layers.")

    if (
        is_cuda_device(args.device)
        or (not args.skip_activations and is_cuda_device(activation_device))
    ) and not args.keep_model_on_device:
        print("Moving model to CPU before per-layer analysis so full weights do not stay on GPU ...")
        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    block_rows, block_cols = args.block_size
    reports: List[LayerReport] = []
    for index, (name, module) in enumerate(named_layers.items(), start=1):
        weight = module.weight.detach().float().to(analysis_device)
        between_spread, block_spread, peakiness = weight_metrics(weight, block_rows, block_cols)

        w_pc = fake_quant_per_channel(weight, fp8_dtype)
        w_pb = fake_quant_per_block(weight, fp8_dtype, block_rows, block_cols)
        weight_rel_mse_pc = relative_mse(weight, w_pc)
        weight_rel_mse_pb = relative_mse(weight, w_pb)

        samples = activation_samples.get(name)
        act_spread, act_outlier_frac = activation_metrics(samples)
        weight_only_out_mse_pc = None
        weight_only_out_mse_pb = None
        out_mse_pc = None
        out_mse_pb = None
        out_cos_pc = None
        out_cos_pb = None

        if samples is not None and samples.numel() > 0:
            x = samples.to(analysis_device).float()
            ref = torch.matmul(x, weight.t())
            weight_only_out_pc = torch.matmul(x, w_pc.t())
            weight_only_out_pb = torch.matmul(x, w_pb.t())
            weight_only_out_mse_pc = relative_mse(ref, weight_only_out_pc)
            weight_only_out_mse_pb = relative_mse(ref, weight_only_out_pb)

            x_pc = fake_quant_activation_per_token(x, fp8_dtype)
            x_pb = fake_quant_per_block(x, fp8_dtype, 1, block_cols)
            out_pc = torch.matmul(x_pc, w_pc.t())
            out_pb = torch.matmul(x_pb, w_pb.t())
            out_mse_pc = relative_mse(ref, out_pc)
            out_mse_pb = relative_mse(ref, out_pb)
            out_cos_pc = mean_cosine_similarity(ref, out_pc)
            out_cos_pb = mean_cosine_similarity(ref, out_pb)
            del x, x_pc, x_pb, ref, weight_only_out_pc, weight_only_out_pb, out_pc, out_pb

        winner, confidence, reason = decide_winner(
            name=name,
            between_channel_scale_spread=between_spread,
            within_block_scale_spread=block_spread,
            channel_peakiness_p95=peakiness,
            weight_rel_mse_per_channel=weight_rel_mse_pc,
            weight_rel_mse_per_block=weight_rel_mse_pb,
            output_rel_mse_per_channel=out_mse_pc,
            output_rel_mse_per_block=out_mse_pb,
            min_output_mse=args.min_output_mse,
        )

        report = LayerReport(
            name=name,
            shape=f"{weight.shape[0]}x{weight.shape[1]}",
            params_m=weight.numel() / 1.0e6,
            samples=0 if samples is None else int(samples.shape[0]),
            between_channel_scale_spread=between_spread,
            within_block_scale_spread=block_spread,
            channel_peakiness_p95=peakiness,
            activation_channel_spread=act_spread,
            activation_outlier_frac=act_outlier_frac,
            weight_rel_mse_per_channel=weight_rel_mse_pc,
            weight_rel_mse_per_block=weight_rel_mse_pb,
            weight_only_output_rel_mse_per_channel=weight_only_out_mse_pc,
            weight_only_output_rel_mse_per_block=weight_only_out_mse_pb,
            output_rel_mse_per_channel=out_mse_pc,
            output_rel_mse_per_block=out_mse_pb,
            output_cosine_per_channel=out_cos_pc,
            output_cosine_per_block=out_cos_pb,
            winner=winner,
            confidence=confidence,
            reason=reason,
        )
        reports.append(report)
        print(
            f"[{index:03d}/{len(named_layers):03d}] {name}: "
            f"decision={winner_label(winner)}, confidence={confidence_label(confidence)}, "
            f"wonly_pc={format_optional(weight_only_out_mse_pc)}, "
            f"wonly_pb={format_optional(weight_only_out_mse_pb)}, "
            f"w8a8_pc={format_optional(out_mse_pc)}, "
            f"w8a8_pb={format_optional(out_mse_pb)}, reason={reason}"
        )
        del weight, w_pc, w_pb
        if is_cuda_device(analysis_device):
            torch.cuda.empty_cache()

    summary = {
        "per-channel": sum(row.winner == "per-channel" for row in reports),
        "per-block": sum(row.winner == "per-block" for row in reports),
        "close": sum(row.winner == "close" for row in reports),
    }
    print()
    print("Summary:")
    print(f"  leans per-channel: {summary['per-channel']}")
    print(f"  leans per-block:   {summary['per-block']}")
    print(f"  close:             {summary['close']}")
    print()
    print_top_table(reports, args.top_k, args.min_output_mse)

    if args.output_csv or args.output_json:
        print_table_metric_explanation_zh()

    if args.output_csv:
        with Path(args.output_csv).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(reports[0]).keys()))
            writer.writeheader()
            for row in reports:
                writer.writerow(asdict(row))
        print(f"\nWrote CSV to {args.output_csv}")

    if args.output_json:
        payload = {
            "summary": summary,
            "block_size": [block_rows, block_cols],
            "fp8_dtype": str(fp8_dtype),
            "reports": [asdict(row) for row in reports],
        }
        Path(args.output_json).write_text(json.dumps(payload, indent=2))
        print(f"Wrote JSON to {args.output_json}")


if __name__ == "__main__":
    main()
