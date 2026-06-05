#!/usr/bin/env python3
"""Benchmark CUTLASS and DeepGEMM FP8 block GEMMs on Qwen3-14B shapes.

Both backends consume the same pre-quantized tensors:

* A uses FP8 scales for [1, 128] blocks.
* B uses FP8 scales for [128, 128] blocks.
* GEMM output is BF16.

Quantization and backend-specific scale layout preparation are excluded from
the measured region. DeepGEMM is invoked with ``disable_ue8m0_cast=True`` so
that it measures the same FP32-scale FP8 block scheme as CUTLASS.

Example:
  python scripts/bench_fp8_block_cutlass_deepgemm_qwen3_14b.py \
      --stage both --warmup 20 --iters 100 --verify
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch


FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max
BLOCK_K = 128
BLOCK_N = 128

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
        (QWEN3_14B_NUM_ATTN_HEADS + 2 * QWEN3_14B_NUM_KV_HEADS)
        * QWEN3_14B_HEAD_DIM,
    ),
    ("o_proj", QWEN3_14B_HIDDEN_SIZE, QWEN3_14B_HIDDEN_SIZE),
    ("gate_up_proj", QWEN3_14B_HIDDEN_SIZE, 2 * QWEN3_14B_INTERMEDIATE_SIZE),
    ("down_proj", QWEN3_14B_INTERMEDIATE_SIZE, QWEN3_14B_HIDDEN_SIZE),
)


@dataclass
class QuantizedInputs:
    a: torch.Tensor
    b: torch.Tensor
    scale_a: torch.Tensor
    scale_b: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark CUTLASS and DeepGEMM FP8 block GEMMs for Qwen3-14B."
    )
    parser.add_argument(
        "--backend",
        choices=["cutlass", "deepgemm", "both"],
        default="both",
        help="Kernel backend(s) to benchmark.",
    )
    parser.add_argument(
        "--stage",
        choices=["prefill", "decode", "both"],
        default="both",
        help="Run prefill, decode, or both token-count profiles.",
    )
    parser.add_argument("--prefill-m", type=int, default=2048)
    parser.add_argument("--decode-m", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=QWEN3_14B_NUM_LAYERS)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--include-lm-head",
        action="store_true",
        help="Also benchmark the final hidden_size x vocab_size projection.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare CUTLASS and DeepGEMM outputs once per shape before timing.",
    )
    parser.add_argument("--atol", type=float, default=1.0)
    parser.add_argument("--rtol", type=float, default=2.0e-2)
    return parser.parse_args()


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")


def selected_backends(name: str) -> tuple[str, ...]:
    return ("cutlass", "deepgemm") if name == "both" else (name,)


def rand_bf16(shape: tuple[int, int], seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return torch.randn(
        shape, device="cuda", dtype=torch.bfloat16, generator=generator
    )


def quantize_activation(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape[1] % BLOCK_K:
        raise ValueError(f"A K dimension must be divisible by {BLOCK_K}: {x.shape}")
    blocks = x.float().view(x.shape[0], x.shape[1] // BLOCK_K, BLOCK_K)
    scale = blocks.abs().amax(dim=2).clamp(min=1.0e-12) / FP8_MAX
    x_q = (
        (blocks / scale.unsqueeze(2))
        .clamp(-FP8_MAX, FP8_MAX)
        .to(FP8_DTYPE)
        .reshape_as(x)
    )
    return x_q, scale.float()


def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n, k = weight.shape
    if n % BLOCK_N or k % BLOCK_K:
        raise ValueError(
            f"B dimensions must be divisible by ({BLOCK_N}, {BLOCK_K}): "
            f"{tuple(weight.shape)}"
        )
    blocks = weight.float().view(n // BLOCK_N, BLOCK_N, k // BLOCK_K, BLOCK_K)
    scale = blocks.abs().amax(dim=(1, 3)).clamp(min=1.0e-12) / FP8_MAX
    weight_q = (
        (blocks / scale[:, None, :, None])
        .clamp(-FP8_MAX, FP8_MAX)
        .to(FP8_DTYPE)
        .reshape_as(weight)
    )
    return weight_q, scale.float()


def make_inputs(m: int, n: int, k: int, seed: int) -> QuantizedInputs:
    x = rand_bf16((m, k), seed)
    weight = rand_bf16((n, k), seed + 1)
    x_q, x_scale = quantize_activation(x)
    weight_q, weight_scale = quantize_weight(weight)
    return QuantizedInputs(x_q, weight_q, x_scale, weight_scale)


def make_cutlass_runner(inputs: QuantizedInputs) -> Callable[[], torch.Tensor]:
    from sgl_kernel import fp8_blockwise_scaled_mm

    # CUTLASS consumes B as [K, N], and both scales in K-major memory layout.
    b_k_n = inputs.b.t()
    scale_a_mn_major = inputs.scale_a.t().contiguous().t()
    scale_b_k_major = inputs.scale_b.t()

    def run() -> torch.Tensor:
        return fp8_blockwise_scaled_mm(
            inputs.a,
            b_k_n,
            scale_a_mn_major,
            scale_b_k_major,
            torch.bfloat16,
        )

    return run


def make_deepgemm_runner(inputs: QuantizedInputs) -> Callable[[], torch.Tensor]:
    import deep_gemm
    from deep_gemm.utils.layout import get_mn_major_tma_aligned_tensor

    scale_a = get_mn_major_tma_aligned_tensor(inputs.scale_a)
    out = torch.empty(
        (inputs.a.shape[0], inputs.b.shape[0]),
        device=inputs.a.device,
        dtype=torch.bfloat16,
    )

    def run() -> torch.Tensor:
        deep_gemm.fp8_gemm_nt(
            (inputs.a, scale_a),
            (inputs.b, inputs.scale_b),
            out,
            disable_ue8m0_cast=True,
        )
        return out

    return run


def make_runners(
    inputs: QuantizedInputs, backends: tuple[str, ...]
) -> dict[str, Callable[[], torch.Tensor]]:
    runners: dict[str, Callable[[], torch.Tensor]] = {}
    for backend in backends:
        if backend == "cutlass":
            runners[backend] = make_cutlass_runner(inputs)
        else:
            runners[backend] = make_deepgemm_runner(inputs)
    return runners


def time_cuda(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            out = fn()
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            out = fn()
        torch.cuda.synchronize()

    _ = out.shape
    return (time.perf_counter() - start) * 1.0e6 / iters


def verify_outputs(
    runners: dict[str, Callable[[], torch.Tensor]], atol: float, rtol: float
) -> None:
    if not {"cutlass", "deepgemm"}.issubset(runners):
        raise ValueError("--verify requires --backend both.")
    with torch.inference_mode():
        cutlass_out = runners["cutlass"]()
        deepgemm_out = runners["deepgemm"]()
        torch.cuda.synchronize()
    torch.testing.assert_close(cutlass_out, deepgemm_out, atol=atol, rtol=rtol)


def profile_rows(include_lm_head: bool) -> list[tuple[str, int, int]]:
    rows = list(QWEN3_14B_GEMMS)
    if include_lm_head:
        rows.append(("lm_head", QWEN3_14B_HIDDEN_SIZE, QWEN3_14B_VOCAB_SIZE))
    return rows


def run_stage(args: argparse.Namespace, stage: str, backends: tuple[str, ...]) -> None:
    m = args.prefill_m if stage == "prefill" else args.decode_m
    total_us = {backend: 0.0 for backend in backends}

    print(f"\n[{stage}] M={m}")
    header = f"{'gemm':<14} {'M':>6} {'N':>8} {'K':>8}"
    for backend in backends:
        header += f" {backend + '_us':>14} {backend + '_TF':>13}"
    if len(backends) == 2:
        header += f" {'cutlass/dg':>12}"
    print(header)
    print("-" * len(header))

    for index, (name, k, n) in enumerate(profile_rows(args.include_lm_head)):
        seed = args.seed + index * 17 + (0 if stage == "prefill" else 1000)
        inputs = make_inputs(m, n, k, seed)
        runners = make_runners(inputs, backends)

        if args.verify:
            verify_outputs(runners, args.atol, args.rtol)

        latency_us: dict[str, float] = {}
        for backend in backends:
            latency_us[backend] = time_cuda(runners[backend], args.warmup, args.iters)
            if name != "lm_head":
                total_us[backend] += latency_us[backend]

        line = f"{name:<14} {m:>6} {n:>8} {k:>8}"
        for backend in backends:
            tflops = 2.0 * m * n * k / latency_us[backend] / 1.0e6
            line += f" {latency_us[backend]:>14.3f} {tflops:>13.2f}"
        if len(backends) == 2:
            line += f" {latency_us['cutlass'] / latency_us['deepgemm']:>12.3f}"
        print(line)

        del runners, inputs
        torch.cuda.empty_cache()

    print("-" * len(header))
    total_line = f"{'one_layer':<14} {'':>6} {'':>8} {'':>8}"
    layers_line = f"{str(args.num_layers) + '_layers':<14} {'':>6} {'':>8} {'':>8}"
    for backend in backends:
        total_line += f" {total_us[backend]:>14.3f} {'':>13}"
        layers_line += f" {total_us[backend] * args.num_layers:>14.3f} {'':>13}"
    if len(backends) == 2:
        ratio = total_us["cutlass"] / total_us["deepgemm"]
        total_line += f" {ratio:>12.3f}"
        layers_line += f" {ratio:>12.3f}"
    print(total_line)
    print(layers_line)


def main() -> None:
    args = parse_args()
    require_cuda()
    if args.verify and args.backend != "both":
        raise ValueError("--verify requires --backend both.")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be >= 0 and --iters must be > 0.")

    backends = selected_backends(args.backend)
    stages = ("prefill", "decode") if args.stage == "both" else (args.stage,)
    print(
        "Qwen3-14B FP8 block GEMM: "
        f"hidden={QWEN3_14B_HIDDEN_SIZE}, "
        f"intermediate={QWEN3_14B_INTERMEDIATE_SIZE}, "
        f"layers={args.num_layers}, backends={','.join(backends)}"
    )
    print(
        "Block scales: A=[1,128], B=[128,128], output=bf16; "
        "quantization/layout conversion excluded from timing."
    )
    print("DeepGEMM uses FP32 scales with disable_ue8m0_cast=True.")

    for stage in stages:
        run_stage(args, stage, backends)


if __name__ == "__main__":
    main()
