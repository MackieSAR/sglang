#!/usr/bin/env python3
"""Standalone FP8 per-channel vs MXFP8 CUTLASS GEMM micro-benchmark.

This does not construct a vLLM engine.  It prepares random tensors, quantizes
them outside the measured region, then times only the GEMM entry points:

* fp8-pertensor:   vLLM cutlass_scaled_mm, one scale for A and one for B
* fp8-perchannel: vLLM cutlass_scaled_mm, one B scale per output channel
* fp8-perblock:   vLLM cutlass_scaled_mm, A blocks [1, 128], B blocks [128, 128]
* mxfp8:          FlashInfer mm_mxfp8 with backend="cutlass"

Example:
  PYTHONPATH=/data/home/cgxu2/vllm_0190:$PYTHONPATH \
  python scripts/bench_fp8_mxfp8_cutlass_gemm.py \
      --mode both --m 128 --n 8192 --k 8192 --warmup 20 --iters 100

  PYTHONPATH=/data/home/cgxu2/vllm_0190:$PYTHONPATH \
  python scripts/bench_fp8_mxfp8_cutlass_gemm.py \
      --profile qwen3-14b --mode both --warmup 20 --iters 100
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F


FP8_MAX = 448.0
FP8_PER_TENSOR = "fp8-pertensor"
FP8_PER_CHANNEL = "fp8-perchannel"
FP8_PER_BLOCK = "fp8-perblock"
MXFP8 = "mxfp8"
ALL_MODES = (FP8_PER_TENSOR, FP8_PER_CHANNEL, FP8_PER_BLOCK, MXFP8)
QWEN3_14B_HIDDEN_SIZE = 5120
QWEN3_14B_INTERMEDIATE_SIZE = 17408
QWEN3_14B_NUM_LAYERS = 40
QWEN3_14B_NUM_ATTN_HEADS = 40
QWEN3_14B_NUM_KV_HEADS = 8
QWEN3_14B_HEAD_DIM = 128
QWEN3_14B_VOCAB_SIZE = 151936

QWEN3_14B_GEMMS = (
    (
        "qkv_proj",
        QWEN3_14B_HIDDEN_SIZE,
        (
            QWEN3_14B_NUM_ATTN_HEADS
            + 2 * QWEN3_14B_NUM_KV_HEADS
        )
        * QWEN3_14B_HEAD_DIM,
    ),
    ("o_proj", QWEN3_14B_HIDDEN_SIZE, QWEN3_14B_HIDDEN_SIZE),
    ("gate_up_proj", QWEN3_14B_HIDDEN_SIZE, 2 * QWEN3_14B_INTERMEDIATE_SIZE),
    ("down_proj", QWEN3_14B_INTERMEDIATE_SIZE, QWEN3_14B_HIDDEN_SIZE),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark standalone CUTLASS GEMM calls for FP8 and MXFP8."
    )
    parser.add_argument(
        "--mode",
        choices=[
            "fp8",
            FP8_PER_TENSOR,
            FP8_PER_CHANNEL,
            FP8_PER_BLOCK,
            MXFP8,
            "both",
            "all",
        ],
        default="both",
        help=(
            "fp8 is an alias of fp8-perchannel. both keeps old behavior: "
            "fp8-perchannel + mxfp8. all runs every path."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=["single", "qwen3-14b"],
        default="single",
        help="Run one custom GEMM shape, or the Qwen3-14B layer GEMM sequence.",
    )
    parser.add_argument("--m", type=int, default=128, help="GEMM M dimension.")
    parser.add_argument("--n", type=int, default=8192, help="GEMM N dimension.")
    parser.add_argument("--k", type=int, default=8192, help="GEMM K dimension.")
    parser.add_argument(
        "--stage",
        choices=["prefill", "decode", "both"],
        default="both",
        help="Qwen3-14B profile stage. Prefill M defaults to 2048, decode M to 128.",
    )
    parser.add_argument("--prefill-m", type=int, default=2048)
    parser.add_argument("--decode-m", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=QWEN3_14B_NUM_LAYERS)
    parser.add_argument(
        "--include-lm-head",
        action="store_true",
        help="Also time the final hidden_size x vocab_size logits GEMM.",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dtype",
        choices=["bf16", "fp16"],
        default="bf16",
        help="Output dtype for both kernels.",
    )
    parser.add_argument(
        "--vllm-path",
        type=str,
        default=os.environ.get("VLLM_PATH"),
        help="Optional local vLLM checkout to prepend to PYTHONPATH.",
    )
    return parser.parse_args()


def maybe_add_vllm_path(path: str | None) -> None:
    if path:
        sys.path.insert(0, path)


def dtype_from_name(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bf16" else torch.float16


def selected_modes(mode: str) -> tuple[str, ...]:
    if mode == "fp8":
        return (FP8_PER_CHANNEL,)
    if mode == "both":
        return (FP8_PER_CHANNEL, MXFP8)
    if mode == "all":
        return ALL_MODES
    return (mode,)


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")


def rand_bf16(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)


def quantize_fp8_per_tensor(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = x.abs().float().amax().clamp(min=1.0e-12) / FP8_MAX
    x_q = (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return x_q, scale.reshape(1, 1).float()


def quantize_fp8_per_channel_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # weight is [N, K].  cutlass_scaled_mm sees weight.T as [K, N], so the
    # dequant scale broadcasts as [1, N].
    scale = weight.abs().float().amax(dim=1).clamp(min=1.0e-12) / FP8_MAX
    weight_q = (
        (weight.float() / scale[:, None])
        .clamp(-FP8_MAX, FP8_MAX)
        .to(torch.float8_e4m3fn)
    )
    return weight_q, scale.reshape(1, -1).float()


def quantize_fp8_per_block_matrix(
    x: torch.Tensor,
    block_m: int,
    block_n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = x.shape
    if rows % block_m != 0 or cols % block_n != 0:
        raise ValueError(
            "FP8 block quantization requires divisible dimensions: "
            f"shape={tuple(x.shape)}, block=({block_m}, {block_n})"
        )

    x_fp32 = x.float()
    row_blocks = rows // block_m
    col_blocks = cols // block_n
    x_blocked = x_fp32.view(row_blocks, block_m, col_blocks, block_n)
    scale = x_blocked.abs().amax(dim=(1, 3)).clamp(min=1.0e-12) / FP8_MAX
    x_q = (
        (x_blocked / scale[:, None, :, None])
        .clamp(-FP8_MAX, FP8_MAX)
        .to(torch.float8_e4m3fn)
        .reshape(rows, cols)
    )
    return x_q, scale.float()


def pad_mxfp8_m_to_128(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    m = x.shape[0]
    padded_m = ((m + 127) // 128) * 128
    if padded_m == m:
        return x, padded_m
    return F.pad(x, (0, 0, 0, padded_m - m)), padded_m


def time_cuda(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
    torch.cuda.synchronize()

    elapsed_s = time.perf_counter() - start
    avg_us = elapsed_s * 1.0e6 / iters
    print(f"  avg latency: {avg_us:9.3f} us")
    print(f"  last output: shape={tuple(out.shape)}, dtype={out.dtype}")
    return avg_us


def time_cuda_quiet(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
    torch.cuda.synchronize()

    # Touch the tensor metadata so the variable is clearly live through timing.
    _ = out.shape
    return (time.perf_counter() - start) * 1.0e6 / iters


def bench_fp8_per_channel(
    x: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> float:
    from vllm import _custom_ops as ops

    x_q, x_scale = quantize_fp8_per_tensor(x)
    weight_q, weight_scale = quantize_fp8_per_channel_weight(weight)

    print("fp8-perchannel: vllm._custom_ops.cutlass_scaled_mm")
    print(
        "  A={}, B={}, scale_a={}, scale_b={}".format(
            tuple(x_q.shape),
            tuple(weight_q.t().shape),
            tuple(x_scale.shape),
            tuple(weight_scale.shape),
        )
    )
    return time_cuda(
        lambda: ops.cutlass_scaled_mm(
            x_q,
            weight_q.t(),
            x_scale,
            weight_scale,
            out_dtype=out_dtype,
        ),
        warmup,
        iters,
    )


def prepare_cutlass_fp8_tensors(
    scheme: str,
    x: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if scheme == FP8_PER_TENSOR:
        x_q, x_scale = quantize_fp8_per_tensor(x)
        weight_q, weight_scale = quantize_fp8_per_tensor(weight)
        return x_q, weight_q.t(), x_scale, weight_scale

    if scheme == FP8_PER_CHANNEL:
        x_q, x_scale = quantize_fp8_per_tensor(x)
        weight_q, weight_scale = quantize_fp8_per_channel_weight(weight)
        return x_q, weight_q.t(), x_scale, weight_scale

    if scheme == FP8_PER_BLOCK:
        x_q, x_scale = quantize_fp8_per_block_matrix(x, 1, 128)
        weight_q, weight_scale = quantize_fp8_per_block_matrix(weight, 128, 128)
        return x_q, weight_q.t(), x_scale, weight_scale.t().contiguous()

    raise ValueError(f"Unsupported FP8 scheme: {scheme}")


def bench_cutlass_fp8_scheme(
    scheme: str,
    x: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> float:
    from vllm import _custom_ops as ops

    x_q, b_q, x_scale, b_scale = prepare_cutlass_fp8_tensors(scheme, x, weight)

    print(f"{scheme}: vllm._custom_ops.cutlass_scaled_mm")
    print(
        "  A={}, B={}, scale_a={}, scale_b={}".format(
            tuple(x_q.shape),
            tuple(b_q.shape),
            tuple(x_scale.shape),
            tuple(b_scale.shape),
        )
    )
    return time_cuda(
        lambda: ops.cutlass_scaled_mm(
            x_q,
            b_q,
            x_scale,
            b_scale,
            out_dtype=out_dtype,
        ),
        warmup,
        iters,
    )


def bench_cutlass_fp8_scheme_us(
    scheme: str,
    m: int,
    n: int,
    k: int,
    out_dtype: torch.dtype,
    warmup: int,
    iters: int,
    seed: int,
) -> float:
    from vllm import _custom_ops as ops

    x = rand_bf16((m, k), seed)
    weight = rand_bf16((n, k), seed + 1)
    x_q, b_q, x_scale, b_scale = prepare_cutlass_fp8_tensors(scheme, x, weight)

    avg_us = time_cuda_quiet(
        lambda: ops.cutlass_scaled_mm(
            x_q,
            b_q,
            x_scale,
            b_scale,
            out_dtype=out_dtype,
        ),
        warmup,
        iters,
    )
    del x, weight, x_q, b_q, x_scale, b_scale
    torch.cuda.empty_cache()
    return avg_us


def bench_mxfp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> float:
    from flashinfer import mm_mxfp8, mxfp8_quantize

    if x.shape[1] % 32 != 0:
        raise ValueError(f"MXFP8 requires K divisible by 32, got K={x.shape[1]}.")
    if weight.shape[0] < 128 or weight.shape[1] < 128:
        raise ValueError("MXFP8 mm_mxfp8 requires N and K >= 128.")

    orig_m = x.shape[0]
    x, padded_m = pad_mxfp8_m_to_128(x)
    x_q, x_scale = mxfp8_quantize(x, is_sf_swizzled_layout=True)
    weight_q, weight_scale = mxfp8_quantize(weight, is_sf_swizzled_layout=True)

    print('mxfp8: flashinfer.mm_mxfp8 backend="cutlass"')
    if padded_m != orig_m:
        print(f"  padded M: {orig_m} -> {padded_m}")
    print(
        "  A={}, B={}, scale_a={}, scale_b={}".format(
            tuple(x_q.shape),
            tuple(weight_q.t().shape),
            tuple(x_scale.shape),
            tuple(weight_scale.shape),
        )
    )
    return time_cuda(
        lambda: mm_mxfp8(
            x_q,
            weight_q.t(),
            x_scale,
            weight_scale,
            out=None,
            out_dtype=out_dtype,
            backend="cutlass",
        ),
        warmup,
        iters,
    )


def bench_mxfp8_us(
    m: int,
    n: int,
    k: int,
    out_dtype: torch.dtype,
    warmup: int,
    iters: int,
    seed: int,
) -> float:
    from flashinfer import mm_mxfp8, mxfp8_quantize

    if k % 32 != 0:
        raise ValueError(f"MXFP8 requires K divisible by 32, got K={k}.")
    if n < 128 or k < 128:
        raise ValueError("MXFP8 mm_mxfp8 requires N and K >= 128.")

    x = rand_bf16((m, k), seed)
    weight = rand_bf16((n, k), seed + 1)
    x, _ = pad_mxfp8_m_to_128(x)
    x_q, x_scale = mxfp8_quantize(x, is_sf_swizzled_layout=True)
    weight_q, weight_scale = mxfp8_quantize(weight, is_sf_swizzled_layout=True)

    avg_us = time_cuda_quiet(
        lambda: mm_mxfp8(
            x_q,
            weight_q.t(),
            x_scale,
            weight_scale,
            out=None,
            out_dtype=out_dtype,
            backend="cutlass",
        ),
        warmup,
        iters,
    )
    del x, weight, x_q, x_scale, weight_q, weight_scale
    torch.cuda.empty_cache()
    return avg_us


def qwen3_stage_m(args: argparse.Namespace, stage: str) -> int:
    return args.prefill_m if stage == "prefill" else args.decode_m


def run_qwen3_14b_profile(args: argparse.Namespace, out_dtype: torch.dtype) -> None:
    stages = ("prefill", "decode") if args.stage == "both" else (args.stage,)
    modes = selected_modes(args.mode)

    print(
        "Qwen3-14B GEMM profile: "
        f"hidden={QWEN3_14B_HIDDEN_SIZE}, "
        f"intermediate={QWEN3_14B_INTERMEDIATE_SIZE}, "
        f"q_heads={QWEN3_14B_NUM_ATTN_HEADS}, "
        f"kv_heads={QWEN3_14B_NUM_KV_HEADS}, "
        f"layers={args.num_layers}, out_dtype={out_dtype}"
    )
    print("Timing excludes quantization/scale preparation and measures GEMM calls only.")

    for stage in stages:
        m = qwen3_stage_m(args, stage)
        print(f"\n[{stage}] M={m}")
        header = f"{'gemm':<14} {'M':>6} {'N':>8} {'K':>8}"
        for mode in modes:
            col_name = {
                FP8_PER_TENSOR: "tensor_us",
                FP8_PER_CHANNEL: "channel_us",
                FP8_PER_BLOCK: "block_us",
                MXFP8: "mxfp8_us",
            }[mode]
            header += f" {col_name:>12}"
        print(header)
        print("-" * len(header))

        totals = {mode: 0.0 for mode in modes}
        rows = list(QWEN3_14B_GEMMS)
        if args.include_lm_head:
            rows.append(("lm_head", QWEN3_14B_HIDDEN_SIZE, QWEN3_14B_VOCAB_SIZE))

        for idx, (name, k, n) in enumerate(rows):
            seed = args.seed + idx * 17 + (0 if stage == "prefill" else 1000)
            values: dict[str, float] = {}
            for mode in modes:
                if mode == MXFP8:
                    values[mode] = bench_mxfp8_us(
                        m, n, k, out_dtype, args.warmup, args.iters, seed
                    )
                else:
                    values[mode] = bench_cutlass_fp8_scheme_us(
                        mode, m, n, k, out_dtype, args.warmup, args.iters, seed
                    )
                if name != "lm_head":
                    totals[mode] += values[mode]

            line = f"{name:<14} {m:>6} {n:>8} {k:>8}"
            for mode in modes:
                line += f" {values[mode]:>12.3f}"
            print(line)

        print("-" * len(header))
        total_line = f"{'one_layer':<14} {'':>6} {'':>8} {'':>8}"
        for mode in modes:
            total_line += f" {totals[mode]:>12.3f}"
        print(total_line)

        layers_line = f"{str(args.num_layers) + '_layers':<14} {'':>6} {'':>8} {'':>8}"
        for mode in modes:
            layers_line += f" {totals[mode] * args.num_layers:>12.3f}"
        print(layers_line)


def main() -> None:
    args = parse_args()
    maybe_add_vllm_path(args.vllm_path)
    require_cuda()

    out_dtype = dtype_from_name(args.out_dtype)
    torch.manual_seed(args.seed)

    if args.profile == "qwen3-14b":
        run_qwen3_14b_profile(args, out_dtype)
        return

    x = rand_bf16((args.m, args.k), args.seed)
    weight = rand_bf16((args.n, args.k), args.seed + 1)

    print(
        f"shape: M={args.m}, N={args.n}, K={args.k}, "
        f"out_dtype={out_dtype}, warmup={args.warmup}, iters={args.iters}"
    )

    for mode in selected_modes(args.mode):
        if mode == MXFP8:
            bench_mxfp8(x, weight, out_dtype, args.warmup, args.iters)
        else:
            bench_cutlass_fp8_scheme(mode, x, weight, out_dtype, args.warmup, args.iters)


if __name__ == "__main__":
    main()
