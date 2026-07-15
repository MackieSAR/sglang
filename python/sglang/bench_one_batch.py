"""
Benchmark the latency of running a single static batch without a server.

This script does not launch a server and uses the low-level APIs.
It accepts server arguments (the same as launch_server.py) and benchmark arguments (e.g., batch size, input lengths).

# Usage (latency test)
## with dummy weights:
python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --load-format dummy
## sweep through multiple data points and store (append) the results in a jsonl file:
python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 1 12 14 --input-len 256 512 --output-len 32 256 --run-name test_run
## run with profiling:
python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 1 12 14 --input-len 256 512 --profile
## run with profiling to custom directory:
export SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log
python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 1 --input-len 256 --profile
## run with CUDA profiler (nsys):
nsys profile --force-overwrite=true -o bench_one_batch python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 1 --input-len 256 --profile --profile-activities CUDA_PROFILER
# Usage (correctness test):
python -m sglang.bench_one_batch --model-path TinyLlama/TinyLlama-1.1B-Chat-v0.4 --correct

## Reference output (of the correctness test above, can be gpu dependent):
input_ids=[[1, 450, 7483, 310, 3444, 338], [1, 450, 7483, 310, 278, 3303, 13187, 290, 338], [1, 20628, 338, 263, 6575, 1460, 2462, 322, 306, 763]]

prefill logits (first half): tensor([[-10.0312,  -9.5000,   0.8931,  ...,  -4.9414,  -3.2422,  -3.3633],
        [-10.0312,  -9.5000,   0.8931,  ...,  -4.9414,  -3.2422,  -3.3633],
        [ -9.1875, -10.2500,   2.7129,  ...,  -4.3359,  -4.0664,  -4.1328]],
       device='cuda:0')

prefill logits (final): tensor([[-8.3125, -7.1172,  3.3457,  ..., -4.9570, -4.1328, -3.4141],
        [-8.9141, -9.0156,  4.1445,  ..., -4.9922, -4.4961, -4.0781],
        [-9.6328, -9.0547,  4.0195,  ..., -5.3047, -4.7148, -4.4570]],
       device='cuda:0')

========== Prompt 0 ==========
<s> The capital of France is Paris.
The capital of the United States is Washington, D.C.


========== Prompt 1 ==========
<s> The capital of the United Kindom is London.
The capital of the United Kingdom is London.
The capital of the

========== Prompt 2 ==========
<s> Today is a sunny day and I like to go for a walk in the park.
I'm going to the park
"""

import argparse
import copy
import dataclasses
import itertools
import json
import logging
import multiprocessing
import os
import time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import destroy_distributed_environment
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.eplb.expert_distribution import (
    get_global_expert_distribution_recorder,
    set_global_expert_distribution_recorder,
)
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler_dp_attn_mixin import prepare_mlp_sync_batch_raw
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    configure_logger,
    get_bool_env_var,
    kill_process_tree,
    maybe_reindex_device_id,
    require_mlp_sync,
    require_mlp_tp_gather,
    set_gpu_proc_affinity,
    suppress_other_loggers,
)
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.srt.utils.tensor_bridge import use_mlx


class _BenchMoEExpertDistributionRecorder:
    def __init__(self, device, max_num_experts, max_num_layers, max_profile_steps):
        self.device = device
        self.max_num_experts = max_num_experts
        self.max_num_layers = max_num_layers
        self.max_profile_steps = max_profile_steps
        self._recording = False
        self._enabled = torch.zeros((), dtype=torch.int64, device=device)
        self._current_layer_idx = None
        self._expert_seen = torch.zeros(max_num_experts, dtype=torch.int64, device=device)
        self._active_experts_per_layer = torch.zeros(
            max_num_layers, dtype=torch.int64, device=device
        )
        self._active_experts_records = []

    @contextmanager
    def with_current_layer(self, layer_idx):
        old_layer_idx = self._current_layer_idx
        self._current_layer_idx = layer_idx
        try:
            yield
        finally:
            self._current_layer_idx = old_layer_idx

    @contextmanager
    def with_debug_name(self, debug_name):
        yield

    @contextmanager
    def disable_this_region(self):
        yield

    @contextmanager
    def with_forward_pass(self, forward_pass_id: int, forward_batch: ForwardBatch):
        yield {}

    def start_record(self):
        self._active_experts_per_layer.zero_()
        self._active_experts_records.clear()
        self._enabled.fill_(1)
        self._recording = True

    def stop_record(self):
        self._recording = False
        self._enabled.fill_(0)

    def dump_record(self, output_mode="object"):
        if self._active_experts_records:
            active_experts = torch.stack(self._active_experts_records)
        else:
            active_experts = torch.zeros(
                (0, self.max_num_layers), dtype=torch.int64, device="cpu"
            )
        return {
            "active_experts_per_step_layer": active_experts,
        }

    def begin_profile_step(self):
        self._active_experts_per_layer.zero_()

    def finish_profile_step(self):
        active_experts = self._active_experts_per_layer.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(active_experts, op=dist.ReduceOp.MAX)
        self._active_experts_records.append(active_experts.detach().cpu())

    @property
    def recording(self):
        return self._recording

    def on_select_experts(self, topk_ids: torch.Tensor):
        self.on_select_experts_for_layer(topk_ids, self._current_layer_idx)

    def on_select_experts_for_layer(self, topk_ids: torch.Tensor, layer_idx: int):
        if layer_idx is None or layer_idx >= self.max_num_layers:
            return

        topk_ids = topk_ids.reshape(-1)
        topk_ids = topk_ids.to(device=self.device, dtype=torch.int64)
        valid = (topk_ids >= 0) & (topk_ids < self.max_num_experts)
        safe_topk_ids = topk_ids.masked_fill(~valid, 0)
        self._expert_seen.zero_()
        self._expert_seen.scatter_add_(
            dim=0,
            index=safe_topk_ids,
            src=valid.to(torch.int64),
        )
        active_experts = (self._expert_seen > 0).sum() * self._enabled
        self._active_experts_per_layer[layer_idx].copy_(
            torch.maximum(
                self._active_experts_per_layer[layer_idx],
                active_experts,
            )
        )

    def on_bypassed_topk_logits(
        self, router_logits: torch.Tensor, top_k: int, layer_idx: int
    ):
        _, topk_ids = torch.topk(router_logits, k=top_k, dim=-1, sorted=False)
        self.on_select_experts_for_layer(topk_ids, layer_idx)


def _format_moe_expert_distribution(expert_distribution_output, stage):
    if not expert_distribution_output:
        return f"MoE expert distribution ({stage}): no data recorded"

    active_experts = expert_distribution_output.get("active_experts_per_step_layer")
    if active_experts is None:
        return f"MoE expert distribution ({stage}): no active_experts_per_step_layer recorded"

    active_experts = active_experts.to(torch.int64)
    active_rows = active_experts.sum(dim=1) > 0
    if active_rows.any():
        active_experts = active_experts[
            : int(active_rows.nonzero()[-1].item()) + 1
        ]
    active_cols = active_experts.sum(dim=0) > 0
    if active_cols.any():
        active_experts = active_experts[:, active_cols]

    return (
        f"MoE active experts per {stage} step/layer. "
        f"shape={list(active_experts.shape)}, "
        f"per_step_layer={active_experts.tolist()}, "
        f"per_step_avg={active_experts.float().mean(dim=1).tolist() if active_experts.numel() > 0 else []}"
    )


def _get_config_value(config, *names, default=None):
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return default


def _estimate_weight_bytes_per_value(model_config: ModelConfig):
    quant_config = getattr(model_config.hf_config, "quantization_config", None)
    if isinstance(quant_config, dict):
        quant_method = str(quant_config.get("quant_method", "")).lower()
        if "fp8" in quant_method:
            return 1
    try:
        return torch.empty((), dtype=model_config.dtype).element_size()
    except TypeError:
        return 2


def _estimate_moe_expert_weight_bytes(model_config: ModelConfig):
    hf_config = model_config.hf_text_config
    hidden_size = _get_config_value(hf_config, "hidden_size")
    intermediate_size = _get_config_value(
        hf_config,
        "moe_intermediate_size",
        "ffn_dim",
        "intermediate_size",
    )
    if hidden_size is None or intermediate_size is None:
        return None
    return int(hidden_size) * int(intermediate_size) * 3 * _estimate_weight_bytes_per_value(
        model_config
    )


def _format_bytes(num_bytes):
    if num_bytes is None:
        return "n/a"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    unit = units[0]
    for unit in units:
        if abs(value) < 1024 or unit == units[-1]:
            break
        value /= 1024
    return f"{value:.3f} {unit}"


def _summarize_moe_distribution_for_profile(
    expert_distribution_output, model_config: ModelConfig
):
    if not expert_distribution_output:
        return None

    active_experts = expert_distribution_output.get("active_experts_per_step_layer")
    if active_experts is None or active_experts.numel() == 0:
        return None

    active_experts = active_experts.to(torch.float32)
    active_rows = active_experts.sum(dim=1) > 0
    if active_rows.any():
        active_experts = active_experts[
            : int(active_rows.nonzero()[-1].item()) + 1
        ]
    active_cols = active_experts.sum(dim=0) > 0
    if active_cols.any():
        active_experts = active_experts[:, active_cols]
    if active_experts.numel() == 0:
        return None

    expert_weight_bytes = _estimate_moe_expert_weight_bytes(model_config)
    active_expert_instances = int(active_experts.sum().item())
    active_weight_bytes = (
        active_expert_instances * expert_weight_bytes
        if expert_weight_bytes is not None
        else None
    )
    return {
        "shape": list(active_experts.shape),
        "avg_active_experts_per_step_layer": float(active_experts.mean().item()),
        "min_active_experts_per_step_layer": int(active_experts.min().item()),
        "max_active_experts_per_step_layer": int(active_experts.max().item()),
        "active_expert_instances": active_expert_instances,
        "expert_weight_bytes": expert_weight_bytes,
        "active_weight_bytes": active_weight_bytes,
    }


def _categorize_profiler_event(name):
    key = name.lower()
    if any(x in key for x in ("fused_moe", "inplace_fused_experts")):
        return "moe_gemm"
    if any(
        x in key
        for x in (
            "topkgating",
            "topk_softmax",
            "moe_align",
            "count_and_sort_expert",
        )
    ):
        return "moe_route"
    if any(
        x in key
        for x in (
            "moe_sum",
            "silu_and_mul",
            "act_and_mul",
            "memcpy32_post",
            "triton_per_fused_copy__mul_sum",
        )
    ):
        return "moe_postprocess"
    if any(
        x in key
        for x in (
            "create_flashinfer_kv_indices",
            "devicescan",
            "cub::devicescan",
            "clamp_position",
        )
    ):
        return "graph_metadata"
    if any(
        x in key
        for x in (
            "gated_delta",
            "gdn",
            "causal_conv1d",
            "qkvzba",
            "chunkgateddeltarule",
        )
    ):
        return "linear_attn_gdn"
    if any(
        x in key
        for x in (
            "batchdecodewithpagedkvcache",
            "batchprefillwithpagedkvcache",
            "flash_fwd",
            "paged_attention",
            "attention",
        )
    ):
        return "attention_kernel"
    if any(
        x in key
        for x in (
            "fp8_blockwise_scaled_mm",
            "scaled_mm",
            "gemm",
            "cublas",
            "cutlass::kernel",
            "aten::mm",
            "aten::matmul",
            "aten::linear",
        )
    ):
        return "gemm_other"
    if "quant" in key:
        return "quant"
    if any(x in key for x in ("rmsnorm", "layer_norm", "layernorm", "norm")):
        return "norm"
    if any(
        x in key
        for x in (
            "elementwise_kernel",
            "unrolled_elementwise_kernel",
            "vectorized_elementwise_kernel",
            "_scatter_gather_elementwise_kernel",
            "reduce_kernel",
        )
    ):
        return "torch_elementwise"
    if "memcpy" in key or "copy" in key:
        return "memcpy"
    return "other"


def _get_profiler_self_device_time_us(event):
    for attr in (
        "self_device_time_total",
        "self_cuda_time_total",
        "self_xpu_time_total",
    ):
        value = getattr(event, attr, None)
        if value is not None:
            return float(value or 0.0)
    return 0.0


def _is_cuda_kernel_like_event(name):
    key = name.strip()
    lower = key.lower()
    if lower.startswith(
        (
            "aten::",
            "sglang::",
            "sgl_kernel::",
            "cuda",
            "cu",
            "record_",
            "torch-compiled",
            "## call",
            "runtime ",
            "lazy ",
            "command buffer",
        )
    ):
        return lower.startswith(("memcpy", "memset"))
    return lower.startswith(
        (
            "_zn",
            "_",
            "void ",
            "fused_",
            "kernel_",
            "cutlass::",
            "flashinfer::",
            "topkgating",
            "std::",
            "memcpy",
        )
    )


def _format_profiler_stack_source(event):
    stack = getattr(event, "stack", None) or []
    for frame in stack:
        frame = str(frame)
        if "/sglang/" in frame and "bench_one_batch.py" not in frame:
            return frame.strip()
    if stack:
        return str(stack[0]).strip()
    return "n/a"


def _format_profile_key_summary(
    profiler,
    model_config: ModelConfig,
    expert_distribution_output,
    stage,
):
    lines = [f"Key profile summary ({stage}):"]
    hf_config = model_config.hf_text_config
    layer_types = getattr(hf_config, "layer_types", None)
    num_linear_layers = (
        sum(1 for x in layer_types if x == "linear_attention")
        if layer_types is not None
        else 0
    )
    num_full_layers = (
        sum(1 for x in layer_types if x == "full_attention")
        if layer_types is not None
        else int(getattr(model_config, "num_attention_layers", 0) or 0)
    )
    num_layers = int(getattr(model_config, "num_hidden_layers", 0) or 0)
    num_experts = _get_config_value(
        hf_config, "num_experts", "n_routed_experts", "moe_num_experts"
    )
    top_k = _get_config_value(hf_config, "num_experts_per_tok", "num_experts_per_token")
    moe_intermediate_size = _get_config_value(hf_config, "moe_intermediate_size")

    lines.append(
        "  model: "
        f"layers={num_layers}, hidden={getattr(model_config, 'hidden_size', None)}, "
        f"heads={getattr(model_config, 'num_attention_heads', None)}, "
        f"kv_heads={getattr(model_config, 'num_key_value_heads', None)}, "
        f"head_dim={getattr(model_config, 'head_dim', None)}, "
        f"linear_layers={num_linear_layers}, full_attn_layers={num_full_layers}, "
        f"experts={num_experts}, top_k={top_k}, moe_intermediate={moe_intermediate_size}"
    )

    moe_summary = _summarize_moe_distribution_for_profile(
        expert_distribution_output, model_config
    )
    if moe_summary is not None:
        lines.append(
            "  moe_active: "
            f"shape={moe_summary['shape']}, "
            f"avg={moe_summary['avg_active_experts_per_step_layer']:.2f}, "
            f"min={moe_summary['min_active_experts_per_step_layer']}, "
            f"max={moe_summary['max_active_experts_per_step_layer']}, "
            f"active_expert_instances={moe_summary['active_expert_instances']}, "
            f"expert_weight={_format_bytes(moe_summary['expert_weight_bytes'])}, "
            f"active_weight={_format_bytes(moe_summary['active_weight_bytes'])}"
        )

    if profiler is None:
        return "\n".join(lines)

    categories = {}
    top_events = []
    top_elementwise_events = []
    for event in profiler.key_averages(group_by_input_shape=True):
        if not _is_cuda_kernel_like_event(event.key):
            continue
        cuda_us = _get_profiler_self_device_time_us(event)
        cpu_us = float(getattr(event, "self_cpu_time_total", 0.0) or 0.0)
        count = int(getattr(event, "count", 0) or 0)
        if cuda_us <= 0 and cpu_us <= 0:
            continue
        category = _categorize_profiler_event(event.key)
        item = categories.setdefault(category, {"cuda_us": 0.0, "cpu_us": 0.0, "count": 0})
        item["cuda_us"] += cuda_us
        item["cpu_us"] += cpu_us
        item["count"] += count
        top_events.append((cuda_us, count, category, event.key))
        if category == "torch_elementwise":
            top_elementwise_events.append(
                (cuda_us, count, event.key, _format_profiler_stack_source(event))
            )

    total_cuda_us = sum(item["cuda_us"] for item in categories.values())
    lines.append("  cuda_by_category:")
    for category, item in sorted(
        categories.items(), key=lambda x: x[1]["cuda_us"], reverse=True
    ):
        pct = item["cuda_us"] / total_cuda_us * 100 if total_cuda_us > 0 else 0.0
        lines.append(
            f"    {category}: {item['cuda_us'] / 1000:.3f} ms "
            f"({pct:.1f}%), calls={item['count']}"
        )

    lines.append("  top_cuda_events:")
    for cuda_us, count, category, key in sorted(top_events, reverse=True)[:12]:
        avg_us = cuda_us / count if count else 0.0
        lines.append(
            f"    {cuda_us / 1000:.3f} ms, calls={count}, "
            f"avg={avg_us:.3f} us, category={category}, name={key}"
        )
    if top_elementwise_events:
        lines.append("  top_torch_elementwise_sources:")
        for cuda_us, count, key, source in sorted(
            top_elementwise_events, reverse=True
        )[:8]:
            avg_us = cuda_us / count if count else 0.0
            lines.append(
                f"    {cuda_us / 1000:.3f} ms, calls={count}, "
                f"avg={avg_us:.3f} us, name={key}, source={source}"
            )
    return "\n".join(lines)


def _get_max_num_moe_experts_for_bench(model_config: ModelConfig):
    hf_config = model_config.hf_text_config
    for attr in (
        "n_routed_experts",
        "num_experts",
        "moe_num_experts",
        "num_local_experts",
    ):
        value = getattr(hf_config, attr, None)
        if value:
            return int(value) + 64
    return 65536


def _get_max_num_layers_for_bench(model_config: ModelConfig):
    return int(model_config.num_hidden_layers) + 8


def _begin_moe_profile_step():
    recorder = get_global_expert_distribution_recorder()
    if hasattr(recorder, "begin_profile_step"):
        recorder.begin_profile_step()


def _finish_moe_profile_step():
    recorder = get_global_expert_distribution_recorder()
    if hasattr(recorder, "finish_profile_step"):
        recorder.finish_profile_step()


def start_profile(
    profile_activities,
    profile_record_shapes=False,
    rank_print=print,
    enable_torch_profile=True,
    record_moe_expert_distribution=False,
):
    """
    Abstracted function to start profiling based on profile_activities.
    Returns profiler object (or None).
    """
    if record_moe_expert_distribution:
        get_global_expert_distribution_recorder().start_record()

    if not enable_torch_profile:
        return None

    if "CUDA_PROFILER" in profile_activities:
        try:
            torch.cuda.cudart().cudaProfilerStart()
            rank_print("CUDA Profiler started (nsys will begin capturing)")
        except Exception as e:
            rank_print(f"Failed to start CUDA profiler: {e}")
        return None
    else:
        activities = []
        if "CPU" in profile_activities:
            activities.append(torch.profiler.ProfilerActivity.CPU)
        if "GPU" in profile_activities:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        if "XPU" in profile_activities:
            activities.append(torch.profiler.ProfilerActivity.XPU)
        if activities:
            profiler = torch.profiler.profile(
                activities=activities,
                with_stack=True,
                record_shapes=profile_record_shapes,
            )
            profiler.start()
            return profiler
        return None


def stop_profile(
    profiler,
    profile_activities,
    rank_print=print,
    save_trace=False,
    trace_filename=None,
    stage=None,
    enable_torch_profile=True,
    record_moe_expert_distribution=False,
    print_moe_expert_distribution=True,
    print_profile_summary=False,
    model_config=None,
):
    """
    Abstracted function to stop profiling based on profile_activities.
    Optionally saves trace results and prints completion messages.
    """
    if enable_torch_profile and "CUDA_PROFILER" in profile_activities:
        try:
            torch.cuda.cudart().cudaProfilerStop()
            rank_print("CUDA Profiler stopped (nsys should dump traces)")
        except Exception as e:
            rank_print(f"Failed to stop CUDA profiler: {e}")
    elif enable_torch_profile and profiler is not None:
        profiler.stop()

    expert_distribution_output = None
    if record_moe_expert_distribution:
        recorder = get_global_expert_distribution_recorder()
        recorder.stop_record()
        expert_distribution_output = recorder.dump_record(output_mode="object")
        if print_moe_expert_distribution:
            rank_print(
                _format_moe_expert_distribution(expert_distribution_output, stage)
            )

    if print_profile_summary and model_config is not None:
        rank_print(
            _format_profile_key_summary(
                profiler, model_config, expert_distribution_output, stage
            )
        )

    if enable_torch_profile and save_trace:
        if profiler is not None:
            if trace_filename:
                _save_profile_trace_results(profiler, trace_filename)
                stage_desc = f"for {stage}" if stage else ""
                rank_print(
                    f"torch profiler chrome trace {stage_desc} saved to {trace_filename}"
                )
        if "CUDA_PROFILER" in profile_activities:
            rank_print(f"CUDA profiler trace for {stage} completed")

    return expert_distribution_output


@dataclasses.dataclass
class BenchArgs:
    run_name: str = "default"
    batch_size: Tuple[int] = (1,)
    input_len: Tuple[int] = (1024,)
    output_len: Tuple[int] = (16,)
    prompt_filename: str = ""
    result_filename: str = "result.jsonl"
    correctness_test: bool = False
    # This is only used for correctness test
    cut_len: int = 4
    log_decode_step: int = 0
    profile: bool = True
    profile_record_shapes: bool = False
    profile_activities: Tuple[str] = ("CPU", "GPU")
    profile_stage: str = "all"
    profile_filename_prefix: str = "profile"
    profile_start_step: Optional[int] = None
    profile_steps: Optional[int] = None
    print_moe_expert_distribution: bool = False
    print_profile_summary: bool = True

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--run-name", type=str, default=BenchArgs.run_name)
        parser.add_argument(
            "--batch-size", type=int, nargs="+", default=BenchArgs.batch_size
        )
        parser.add_argument(
            "--input-len", type=int, nargs="+", default=BenchArgs.input_len
        )
        parser.add_argument(
            "--output-len", type=int, nargs="+", default=BenchArgs.output_len
        )
        parser.add_argument(
            "--prompt-filename", type=str, default=BenchArgs.prompt_filename
        )
        parser.add_argument(
            "--result-filename", type=str, default=BenchArgs.result_filename
        )
        parser.add_argument("--correctness-test", action="store_true")
        parser.add_argument("--cut-len", type=int, default=BenchArgs.cut_len)
        parser.add_argument(
            "--log-decode-step",
            type=int,
            default=BenchArgs.log_decode_step,
            help="Log decode latency by step, default is set to zero to disable.",
        )
        parser.add_argument(
            "--profile",
            action=argparse.BooleanOptionalAction,
            default=BenchArgs.profile,
            help="Enable profiling.",
        )
        parser.add_argument(
            "--profile-record-shapes",
            action="store_true",
            help="Record tensor shapes in profiling results.",
        )
        parser.add_argument(
            "--profile-activities",
            type=str,
            nargs="+",
            default=["CPU", "GPU"],
            choices=["CPU", "GPU", "CUDA_PROFILER", "XPU"],
            help="Profiler activities: CPU, GPU, XPU, CUDA_PROFILER. If CPU/GPU/XPU, use torch profiler. If CUDA_PROFILER, use CUDA profiler.",
        )
        parser.add_argument(
            "--profile-stage",
            type=str,
            default=BenchArgs.profile_stage,
            choices=["all", "prefill", "decode"],
            help="Which stage to profile: all, prefill, or decode only.",
        )
        parser.add_argument(
            "--profile-filename-prefix",
            type=str,
            default=BenchArgs.profile_filename_prefix,
            help="Prefix of the profiling file names. The full profiling result file(s) be "
            '"[profile_filename_prefix]_batch[batch_size]_input[input_len]_output[output_len].trace.json.gz"',
        )
        parser.add_argument(
            "--profile-start-step",
            type=int,
            default=None,
            help="Decode step at which to start profiling (0-indexed). If not specified, defaults to output_len // 2.",
        )
        parser.add_argument(
            "--profile-steps",
            type=int,
            default=None,
            help="Number of decode steps to profile starting from profile-start-step. If not specified, profiles only one step.",
        )
        parser.add_argument(
            "--print-moe-expert-distribution",
            action="store_true",
            help="Print MoE expert selection totals recorded between start_profile and stop_profile.",
        )
        parser.add_argument(
            "--print-profile-summary",
            action=argparse.BooleanOptionalAction,
            default=BenchArgs.print_profile_summary,
            help="Print a compact summary of model config, key profiler categories, and MoE active weight estimates.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # use the default value's type to cast the args into correct types.
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        result = {}
        for attr, attr_type in attrs:
            value = getattr(args, attr)
            # Handle None values - don't try to cast them
            if value is None or attr_type == type(None):
                result[attr] = value
            else:
                result[attr] = attr_type(value)
        return cls(**result)


def load_model(server_args, port_args, gpu_id, tp_rank):
    suppress_other_loggers()
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None
    moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)

    model_config = ModelConfig.from_server_args(server_args)
    runner_kwargs = dict(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        moe_ep_rank=moe_ep_rank,
        moe_ep_size=server_args.ep_size,
        pp_rank=0,
        pp_size=1,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )

    _use_mlx = use_mlx()
    if _use_mlx:
        from sglang.srt.hardware_backend.mlx.model_runner_stub import (
            MlxModelRunnerStub,
        )

        model_runner = MlxModelRunnerStub(**runner_kwargs)
    else:
        model_runner = ModelRunner(**runner_kwargs)
    rank_print(f"max_total_num_tokens={model_runner.max_total_num_tokens}")
    tokenizer = get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )
    if server_args.tp_size > 1:
        dist.barrier()

    if _use_mlx:
        model_runner = _MlxBenchRunner(model_runner, server_args)
    else:
        model_runner = _TorchBenchRunner(model_runner)

    return model_runner, tokenizer


def prepare_inputs_for_correctness_test(bench_args, tokenizer, custom_prompts):
    if custom_prompts:
        custom_input_len = len(custom_prompts)
        bs = bench_args.batch_size[0]
        if custom_input_len > bs:
            logging.warning(
                f"Custom input size ({custom_input_len}) is larger than batch_size ({bs}). "
                f"Using the first {bs} prompts."
            )
            custom_prompts = custom_prompts[:bs]

    prompts = (
        custom_prompts
        if custom_prompts
        else [
            "The capital of France is",
            "The capital of the United Kindom is",
            "Today is a sunny day and I like",
        ]
    )
    input_ids = [tokenizer.encode(p) for p in prompts]
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=BenchArgs.output_len,
    )

    reqs = []
    for i in range(len(prompts)):
        assert len(input_ids[i]) > bench_args.cut_len

        tmp_input_ids = input_ids[i][: bench_args.cut_len]
        req = Req(
            rid=i,
            origin_input_text=prompts[i],
            origin_input_ids=tmp_input_ids,
            sampling_params=sampling_params,
        )
        req.fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))
        reqs.append(req)

    return input_ids, reqs


def prepare_extend_inputs_for_correctness_test(
    bench_args, input_ids, reqs, model_runner
):
    for i in range(len(reqs)):
        req: Req = reqs[i]
        req.fill_ids += input_ids[i][bench_args.cut_len :]
        if model_runner is not None:
            req.prefix_indices = model_runner.req_to_token_pool.req_to_token[
                i, : bench_args.cut_len
            ].to(req.prefix_indices.dtype)
            req.logprob_start_len = -1
            req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))
    return reqs


def prepare_synthetic_inputs_for_latency_test(
    batch_size, input_len, custom_inputs=None
):
    input_ids = (
        custom_inputs
        if custom_inputs
        else np.random.randint(0, 10000, (batch_size, input_len), dtype=np.int32)
    )
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=BenchArgs.output_len,
    )

    reqs = []
    for i in range(len(input_ids)):
        req = Req(
            rid=i,
            origin_input_text="",
            origin_input_ids=list(input_ids[i]),
            sampling_params=sampling_params,
        )
        req.fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))
        reqs.append(req)

    return reqs


class TreeCacheNamespace(SimpleNamespace):
    def supports_swa(self) -> bool:
        return False

    def supports_mamba(self) -> bool:
        return False

    def is_chunk_cache(self) -> bool:
        return False

    def is_tree_cache(self) -> bool:
        return not self.is_chunk_cache()

    def evict(self, params: EvictParams):
        pass


@torch.no_grad
def extend(reqs, model_runner):
    # Create dummy tree_cache for benchmarks (no prefix caching, just allocation)
    dummy_tree_cache = TreeCacheNamespace(
        page_size=model_runner.server_args.page_size,
        device=model_runner.device,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
    )

    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=dummy_tree_cache,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output = model_runner.forward(forward_batch).logits_output
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits, batch


@torch.no_grad
def decode(input_token_ids, batch, model_runner):
    batch.output_ids = input_token_ids
    batch.prepare_for_decode()
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output = model_runner.forward(forward_batch).logits_output
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits


def _maybe_prepare_mlp_sync_batch(batch: ScheduleBatch, model_runner):
    if require_mlp_sync(model_runner.server_args):
        prepare_mlp_sync_batch_raw(
            batch,
            dp_size=model_runner.server_args.dp_size,
            attn_tp_size=get_attention_tp_size(),
            attn_cp_size=model_runner.attn_cp_size,
            tp_group=model_runner.tp_group,
            get_idle_batch=None,
            disable_cuda_graph=model_runner.server_args.disable_cuda_graph,
            require_mlp_tp_gather=require_mlp_tp_gather(model_runner.server_args),
            disable_overlap_schedule=model_runner.server_args.disable_overlap_schedule,
            offload_tags=set(),
        )


class _TorchBenchRunner:
    """Wraps ModelRunner for the standard PyTorch benchmark path."""

    def __init__(self, model_runner):
        self.torch_runner = model_runner

    def clear(self):
        self.torch_runner.req_to_token_pool.clear()
        self.torch_runner.token_to_kv_pool_allocator.clear()

    def extend(self, reqs):
        return extend(reqs, self.torch_runner)

    def decode(self, next_token_ids, batch):
        return decode(next_token_ids, batch, self.torch_runner)

    def cleanup(self, batch):
        pass

    def synchronize(self):
        synchronize(self.torch_runner.device)

    def max_batch_size(self, input_len, output_len):
        return self.torch_runner.max_total_num_tokens // (input_len + output_len)


class _MlxBenchRunner:
    """Wraps MlxModelRunner for the MLX benchmark path."""

    def __init__(self, model_runner, server_args):
        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

        self.mlx_runner = MlxModelRunner(
            model_path=server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
        )
        self.fake_torch_runner = model_runner

    def clear(self):
        self.mlx_runner.clear()

    def extend(self, reqs):
        req_ids = [str(req.rid) for req in reqs]
        token_ids_list = [[int(t) for t in req.fill_ids] for req in reqs]
        next_token_ids = self.mlx_runner.prefill_batch(req_ids, token_ids_list)
        return torch.tensor(next_token_ids), None, req_ids

    def decode(self, next_token_ids, req_ids):
        next_token_ids = self.mlx_runner.decode_batch(req_ids)
        return torch.tensor(next_token_ids), None

    def cleanup(self, batch):
        if isinstance(batch, list):
            for req_id in batch:
                self.mlx_runner.remove_request(req_id)

    def synchronize(self):
        pass

    def max_batch_size(self, input_len, output_len):
        return self.fake_torch_runner.max_total_num_tokens // (input_len + output_len)


def _read_prompts_from_file(prompt_file, rank_print):
    """Read custom prompts from the file specified by `--prompt-filename`."""
    if not prompt_file:
        return []
    if not os.path.exists(prompt_file):
        rank_print(
            f"Custom prompt file {prompt_file} not found. Using default inputs..."
        )
        return []
    with open(prompt_file, "r") as pf:
        return pf.readlines()


def _get_torch_profiler_output_dir():
    return os.environ.get("SGLANG_TORCH_PROFILER_DIR", "/tmp")


def _create_torch_profiler_filename(
    profile_filename_prefix, batch_size, input_len, output_len, stage
):
    output_dir = _get_torch_profiler_output_dir()
    filename = f"{profile_filename_prefix}_batch{batch_size}_input{input_len}_output{output_len}_{stage}.trace.json.gz"
    return os.path.join(output_dir, filename)


def _save_profile_trace_results(profiler, filename):
    parent_dir = os.path.dirname(os.path.abspath(filename))
    os.makedirs(parent_dir, exist_ok=True)
    profiler.export_chrome_trace(filename)
    print(
        profiler.key_averages(group_by_input_shape=True).table(
            sort_by="self_cpu_time_total"
        )
    )


def _get_bench_model_config(model_runner):
    if hasattr(model_runner, "model_config"):
        return model_runner.model_config
    torch_runner = getattr(model_runner, "torch_runner", None)
    if torch_runner is not None and hasattr(torch_runner, "model_config"):
        return torch_runner.model_config
    return None


def correctness_test(
    server_args,
    port_args,
    bench_args,
    gpu_id,
    tp_rank,
):
    # Configure the logger
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Load the model
    model_runner, tokenizer = load_model(server_args, port_args, gpu_id, tp_rank)

    # Prepare inputs
    custom_prompts = _read_prompts_from_file(bench_args.prompt_filename, rank_print)
    input_ids, reqs = prepare_inputs_for_correctness_test(
        bench_args, tokenizer, custom_prompts
    )
    rank_print(f"\n{input_ids=}\n")

    if bench_args.cut_len > 0:
        # Prefill
        next_token_ids, next_token_logits, batch = model_runner.extend(reqs)
        rank_print(f"prefill logits (first half): {next_token_logits} \n")

        # Prepare extend inputs
        torch_runner = getattr(model_runner, "torch_runner", None)
        reqs = prepare_extend_inputs_for_correctness_test(
            bench_args, input_ids, reqs, torch_runner
        )

    # Extend (prefill w/ KV cache)
    next_token_ids, next_token_logits, batch = model_runner.extend(reqs)
    rank_print(f"prefill logits (final): {next_token_logits} \n")

    # Decode
    output_ids = [input_ids[i] + [next_token_ids[i]] for i in range(len(input_ids))]
    for _ in range(bench_args.output_len[0] - 1):
        next_token_ids, _ = model_runner.decode(next_token_ids, batch)
        next_token_ids_list = next_token_ids.tolist()
        for i in range(len(reqs)):
            output_ids[i].append(next_token_ids_list[i])

    # Clean up
    model_runner.cleanup(batch)

    # Print output texts
    for i in range(len(reqs)):
        rank_print(f"========== Prompt {i} ==========")
        rank_print(tokenizer.decode(output_ids[i]), "\n")


def synchronize(device):
    torch.get_device_module(device).synchronize()


def latency_test_run_once(
    run_name,
    model_runner,
    rank_print,
    reqs,
    batch_size,
    input_len,
    output_len,
    log_decode_step,
    profile,
    profile_record_shapes,
    profile_activities,
    profile_filename_prefix,
    profile_stage,
    tp_rank,
    profile_start_step=None,
    profile_steps=None,
    print_moe_expert_distribution=False,
    print_profile_summary=False,
):
    max_batch_size = model_runner.max_batch_size(input_len, output_len)
    if batch_size > max_batch_size:
        rank_print(
            f"skipping ({batch_size}, {input_len}, {output_len}) due to max batch size limit"
        )
        return

    model_runner.clear()

    measurement_results = {
        "run_name": run_name,
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
    }
    model_config = _get_bench_model_config(model_runner)

    tot_latency = 0

    profiler = None
    enable_torch_profile_prefill = profile and profile_stage in ["all", "prefill"]
    enable_moe_profile_prefill = (
        (print_moe_expert_distribution or print_profile_summary)
        and profile_stage in ["all", "prefill"]
    )
    if enable_torch_profile_prefill or enable_moe_profile_prefill:
        if enable_moe_profile_prefill:
            _begin_moe_profile_step()
        profiler = start_profile(
            profile_activities,
            profile_record_shapes=profile_record_shapes,
            rank_print=rank_print,
            enable_torch_profile=enable_torch_profile_prefill,
            record_moe_expert_distribution=enable_moe_profile_prefill,
        )

    model_runner.synchronize()
    tic = time.perf_counter()
    next_token_ids, _, batch = model_runner.extend(reqs)
    model_runner.synchronize()
    prefill_latency = time.perf_counter() - tic
    if enable_moe_profile_prefill:
        _finish_moe_profile_step()

    if enable_torch_profile_prefill or enable_moe_profile_prefill:
        trace_filename = _create_torch_profiler_filename(
            profile_filename_prefix, batch_size, input_len, output_len, "prefill"
        )
        stop_profile(
            profiler,
            profile_activities,
            rank_print=rank_print,
            save_trace=True,
            trace_filename=trace_filename,
            stage="prefill",
            enable_torch_profile=enable_torch_profile_prefill,
            record_moe_expert_distribution=enable_moe_profile_prefill,
            print_moe_expert_distribution=print_moe_expert_distribution,
            print_profile_summary=print_profile_summary,
            model_config=model_config,
        )

    tot_latency += prefill_latency
    throughput = input_len * batch_size / prefill_latency
    rank_print(
        f"Prefill. latency: {prefill_latency:6.5f} s, throughput: {throughput:9.2f} token/s"
    )
    measurement_results["prefill_latency"] = prefill_latency
    measurement_results["prefill_throughput"] = throughput

    decode_latencies = []
    # Determine profiling start step and end step
    profile_start = (
        profile_start_step if profile_start_step is not None else (output_len // 2)
    )
    profile_end = profile_start + (profile_steps if profile_steps is not None else 1)
    enable_torch_profile_decode = profile and profile_stage in ["all", "decode"]
    enable_moe_profile_decode = (
        (print_moe_expert_distribution or print_profile_summary)
        and profile_stage in ["all", "decode"]
    )
    profiler = None
    decode_profile_started = False
    for i in range(output_len - 1):
        model_runner.synchronize()
        # Start profiler at the specified step
        if (
            (enable_torch_profile_decode or enable_moe_profile_decode)
            and i == profile_start
        ):
            if enable_moe_profile_decode:
                _begin_moe_profile_step()
            profiler = start_profile(
                profile_activities,
                profile_record_shapes=profile_record_shapes,
                rank_print=rank_print,
                enable_torch_profile=enable_torch_profile_decode,
                record_moe_expert_distribution=enable_moe_profile_decode,
            )
            decode_profile_started = True
        elif enable_moe_profile_decode and decode_profile_started:
            _begin_moe_profile_step()

        tic = time.perf_counter()
        next_token_ids, _ = model_runner.decode(next_token_ids, batch)
        model_runner.synchronize()
        latency = time.perf_counter() - tic
        if enable_moe_profile_decode and decode_profile_started:
            _finish_moe_profile_step()

        # Stop profiler after the specified number of steps
        if (
            (enable_torch_profile_decode or enable_moe_profile_decode)
            and decode_profile_started
            and i >= profile_end - 1
        ):
            trace_filename = _create_torch_profiler_filename(
                profile_filename_prefix, batch_size, input_len, output_len, "decode"
            )
            stop_profile(
                profiler,
                profile_activities,
                rank_print=rank_print,
                save_trace=True,
                trace_filename=trace_filename,
                stage="decode",
                enable_torch_profile=enable_torch_profile_decode,
                record_moe_expert_distribution=enable_moe_profile_decode,
                print_moe_expert_distribution=print_moe_expert_distribution,
                print_profile_summary=print_profile_summary,
                model_config=model_config,
            )
            profiler = None
            decode_profile_started = False

        tot_latency += latency
        throughput = batch_size / latency
        decode_latencies.append(latency)
        if i < 5 or (log_decode_step > 0 and i % log_decode_step == 0):
            rank_print(
                f"Decode {i}. Batch size: {batch_size}, latency: {latency:6.5f} s, throughput: {throughput:9.2f} token/s"
            )

    if decode_profile_started:
        trace_filename = _create_torch_profiler_filename(
            profile_filename_prefix, batch_size, input_len, output_len, "decode"
        )
        stop_profile(
            profiler,
            profile_activities,
            rank_print=rank_print,
            save_trace=True,
            trace_filename=trace_filename,
            stage="decode",
            enable_torch_profile=enable_torch_profile_decode,
            record_moe_expert_distribution=enable_moe_profile_decode,
            print_moe_expert_distribution=print_moe_expert_distribution,
            print_profile_summary=print_profile_summary,
            model_config=model_config,
        )

    # Record decode timing from 2nd output
    if output_len > 1:
        med_decode_latency = np.median(decode_latencies)
        med_decode_throughput = batch_size / med_decode_latency
        rank_print(
            f"Decode.  median latency: {med_decode_latency:6.5f} s, median throughput: {med_decode_throughput:9.2f} token/s"
        )
        measurement_results["median_decode_latency"] = med_decode_latency
        measurement_results["median_decode_throughput"] = med_decode_throughput

    throughput = (input_len + output_len) * batch_size / tot_latency
    rank_print(
        f"Total. latency: {tot_latency:6.3f} s, throughput: {throughput:9.2f} token/s"
    )
    measurement_results["total_latency"] = tot_latency
    measurement_results["overall_throughput"] = throughput

    model_runner.cleanup(batch)
    return measurement_results


def latency_test(
    server_args,
    port_args,
    bench_args,
    gpu_id,
    tp_rank,
):
    initialize_moe_config(server_args)
    initialize_fp8_gemm_config(server_args)
    initialize_fp4_gemm_config(server_args)

    # Set CPU affinity
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(
            server_args.pp_size, server_args.tp_size, server_args.nnodes, tp_rank
        )

    # Configure the logger
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    if bench_args.print_moe_expert_distribution or bench_args.print_profile_summary:
        model_config = ModelConfig.from_server_args(server_args)
        max_profile_steps = max(
            1,
            bench_args.profile_steps or 1,
            max(bench_args.output_len),
        )
        set_global_expert_distribution_recorder(
            _BenchMoEExpertDistributionRecorder(
                server_args.device,
                _get_max_num_moe_experts_for_bench(model_config),
                _get_max_num_layers_for_bench(model_config),
                max_profile_steps,
            )
        )

    # Load the model
    model_runner, tokenizer = load_model(server_args, port_args, gpu_id, tp_rank)

    # Prepare inputs for warm up
    reqs = prepare_synthetic_inputs_for_latency_test(
        bench_args.batch_size[0], bench_args.input_len[0]
    )

    # Warm up
    rank_print("Warmup ...")
    latency_test_run_once(
        bench_args.run_name,
        model_runner,
        rank_print,
        reqs,
        bench_args.batch_size[0],
        bench_args.input_len[0],
        min(32, bench_args.output_len[0]),  # shorter decoding to speed up the warmup
        log_decode_step=0,
        profile=False,
        profile_record_shapes=False,
        profile_activities=("CPU", "GPU"),
        profile_filename_prefix="",
        profile_stage="all",
        tp_rank=tp_rank,
        profile_start_step=None,
        profile_steps=None,
        print_moe_expert_distribution=False,
        print_profile_summary=False,
    )

    rank_print("Benchmark ...")

    custom_inputs = _read_prompts_from_file(bench_args.prompt_filename, rank_print)
    custom_inputs = [tokenizer.encode(p.strip()) for p in custom_inputs]
    custom_input_len = len(custom_inputs)

    # Run the sweep
    result_list = []
    for bs, il, ol in itertools.product(
        bench_args.batch_size, bench_args.input_len, bench_args.output_len
    ):
        bs_aligned_inputs = []
        if custom_inputs:
            if custom_input_len == bs:
                bs_aligned_inputs = custom_inputs
            elif custom_input_len > bs:
                rank_print(
                    f"Custom input size ({custom_input_len}) is larger than batch_size ({bs}). "
                    f"Using the first {bs} prompts."
                )
                bs_aligned_inputs = copy.deepcopy(custom_inputs[:bs])
            else:
                rank_print(
                    f"Custom input size ({custom_input_len}) is smaller than batch_size ({bs}). "
                    f"Pad to the desired batch_size with the last prompt."
                )
                bs_aligned_inputs = copy.deepcopy(custom_inputs)
                bs_aligned_inputs.extend(
                    [bs_aligned_inputs[-1]] * (bs - custom_input_len)
                )

        reqs = prepare_synthetic_inputs_for_latency_test(bs, il, bs_aligned_inputs)
        ret = latency_test_run_once(
            bench_args.run_name,
            model_runner,
            rank_print,
            reqs,
            bs,
            il,
            ol,
            bench_args.log_decode_step,
            bench_args.profile if tp_rank == 0 else False,
            bench_args.profile_record_shapes if tp_rank == 0 else False,
            bench_args.profile_activities,
            bench_args.profile_filename_prefix,
            bench_args.profile_stage,
            tp_rank,
            bench_args.profile_start_step,
            bench_args.profile_steps,
            bench_args.print_moe_expert_distribution,
            bench_args.print_profile_summary,
        )
        if ret is not None:
            result_list.append(ret)

    # Write results in jsonlines format on rank 0.
    if tp_rank == 0 and bench_args.result_filename:
        with open(bench_args.result_filename, "a") as fout:
            for result in result_list:
                fout.write(json.dumps(result) + "\n")

    if server_args.tp_size > 1:
        destroy_distributed_environment()


def main(server_args, bench_args):
    server_args.cuda_graph_max_bs = max(bench_args.batch_size)
    if bench_args.print_moe_expert_distribution or bench_args.print_profile_summary:
        if server_args.expert_distribution_recorder_mode is not None:
            logging.warning(
                "bench_one_batch MoE profiling uses a bench-local recorder; ignoring "
                f"expert_distribution_recorder_mode={server_args.expert_distribution_recorder_mode!r}."
            )
        server_args.expert_distribution_recorder_mode = None

    _set_envs_and_config(server_args)

    if server_args.model_path:
        if bench_args.correctness_test:
            work_func = correctness_test
        else:
            work_func = latency_test
    else:
        raise ValueError(
            "Provide --model-path for running the tests or "
            "provide --result-filename for plotting the results"
        )

    port_args = PortArgs.init_new(server_args)

    if server_args.tp_size == 1:
        work_func(server_args, port_args, bench_args, 0, 0)
    else:
        workers = []
        for tp_rank in range(server_args.tp_size):
            with maybe_reindex_device_id(tp_rank) as gpu_id:
                proc = multiprocessing.Process(
                    target=work_func,
                    args=(
                        server_args,
                        port_args,
                        bench_args,
                        gpu_id,
                        tp_rank,
                    ),
                )
                proc.start()
                workers.append(proc)

        for proc in workers:
            proc.join()

        proc.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        main(server_args, bench_args)
    finally:
        if server_args.tp_size != 1:
            kill_process_tree(os.getpid(), include_parent=False)
