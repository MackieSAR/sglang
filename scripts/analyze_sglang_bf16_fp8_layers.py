#!/usr/bin/env python3
"""Compare BF16 and FP8/mixed checkpoints through SGLang's model path.

This script intentionally uses SGLang's low-level bench_one_batch loader and
prefill path instead of plain Hugging Face forward hooks, so the quantized
checkpoint is loaded by the same SGLang quantization code used in serving.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import torch

from sglang.bench_one_batch import (
    _maybe_prepare_mlp_sync_batch,
    load_model,
)
import sglang.srt.distributed.parallel_state as parallel_state
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

try:
    from sglang.srt.distributed.parallel_state import cleanup_dist_env_and_memory
except ImportError:
    def cleanup_dist_env_and_memory(shutdown_ray: bool = False):
        destroy_model_parallel()
        destroy_distributed_environment()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            with contextlib.suppress(Exception):
                torch.distributed.destroy_process_group()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

try:
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
except ImportError:
    def initialize_fp8_gemm_config(_server_args):
        pass

try:
    from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
except ImportError:
    def initialize_fp4_gemm_config(_server_args):
        pass


class TreeCacheNamespace(SimpleNamespace):
    """Minimal prefix-cache stub for one-shot prefill benchmarking.

    Some SGLang branches define this helper in bench_one_batch.py and some do
    not, so keeping it local makes this script work across both layouts.
    """

    def supports_swa(self) -> bool:
        return False

    def supports_mamba(self) -> bool:
        return False

    def is_chunk_cache(self) -> bool:
        return False

    def is_tree_cache(self) -> bool:
        return True

    def evict(self, params):
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Layer-wise BF16 vs FP8 analysis using SGLang internals."
    )
    parser.add_argument("--bf16-model-path", required=True)
    parser.add_argument("--fp8-model-path", required=True)
    parser.add_argument(
        "--fp8-quantization",
        default="compressed-tensors",
        help="SGLang quantization name for the FP8 checkpoint.",
    )
    parser.add_argument(
        "--bf16-quantization",
        default=None,
        help="Usually leave unset. Useful only for non-standard reference checkpoints.",
    )
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.82)
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument(
        "--kv-cache-dtype",
        default=None,
        help="Deprecated compatibility option. Sets both BF16 and FP8 KV cache dtype.",
    )
    parser.add_argument(
        "--bf16-kv-cache-dtype",
        default=None,
        help="KV cache dtype for the BF16 reference model. Defaults to auto, which follows --dtype.",
    )
    parser.add_argument(
        "--fp8-kv-cache-dtype",
        default=None,
        help="KV cache dtype for the FP8 model. Defaults to fp8_e4m3.",
    )
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-total-tokens", type=int, default=None)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument(
        "--prompts-file",
        default=None,
        help="Text file, JSON, or JSONL. JSONL rows with messages use the tokenizer chat template.",
    )
    parser.add_argument("--max-prompts", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--bf16-override-module-regex",
        action="append",
        default=[],
        help=(
            "Regex for FP8 model modules whose weights should be replaced from "
            "the BF16 checkpoint after loading, e.g. 'model\\.embed_tokens$' "
            "or 'model\\.layers\\.1\\.self_attn\\..*'."
        ),
    )
    parser.add_argument(
        "--force-bf16-moe-routing",
        action="store_true",
        help=(
            "Debug option for MoE models. Record BF16 TopK expert ids and weights "
            "and force the FP8 run to reuse both."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def read_prompts(args: argparse.Namespace, tokenizer) -> List[str]:
    prompts: List[str] = list(args.prompt)
    if args.prompts_file:
        path = Path(args.prompts_file)
        for line in path.read_text().splitlines():
            if line.strip():
                prompts.append(extract_prompt(json.loads(line), tokenizer))
        # else:
        #     prompts.extend([line.strip() for line in path.read_text().splitlines() if line.strip()])

    if not prompts:
        prompts = [
            "Explain in three sentences why quantized models can lose accuracy.",
            "Solve: If x + 2y = 7 and 3x - y = 4, what is x?",
            "Write a short Python function that checks whether a number is prime.",
        ]
    return prompts[: args.max_prompts]


def extract_prompt(item, tokenizer) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        if "messages" in item:
            return apply_chat_template_text(tokenizer, item["messages"])
        for key in ("text", "prompt", "input", "query"):
            if key in item:
                return item[key]
    raise ValueError(f"Cannot extract prompt from {item!r}")


def apply_chat_template_text(tokenizer, messages) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def build_server_args(
    args: argparse.Namespace,
    model_path: str,
    quantization: str | None,
    kv_cache_dtype: str,
) -> ServerArgs:
    if args.tp_size != 1:
        raise ValueError(
            "This analysis script currently supports --tp-size 1 only. "
            "bench_one_batch needs multiprocessing setup for TP > 1."
        )
    server_args = ServerArgs(
        model_path=model_path,
        tokenizer_path=args.tokenizer_path or args.bf16_model_path,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        device=args.device,
        tp_size=args.tp_size,
        quantization=quantization,
        kv_cache_dtype=kv_cache_dtype,
        mem_fraction_static=args.mem_fraction_static,
        disable_cuda_graph=True,
        disable_piecewise_cuda_graph=True,
        disable_radix_cache=True,
        log_level="warning",
    )
    if args.attention_backend:
        server_args.attention_backend = args.attention_backend
    if args.max_model_len is not None:
        server_args.context_length = args.max_model_len
    if args.max_total_tokens is not None:
        server_args.max_total_tokens = args.max_total_tokens
    return server_args


def setup_sglang_runtime(server_args: ServerArgs) -> PortArgs:
    _set_envs_and_config(server_args)
    initialize_moe_config(server_args)
    initialize_fp8_gemm_config(server_args)
    initialize_fp4_gemm_config(server_args)
    return PortArgs.init_new(server_args)


def unwrap_model_runner(bench_runner):
    """Return the underlying SGLang ModelRunner from bench_one_batch wrappers."""
    if hasattr(bench_runner, "torch_runner"):
        return bench_runner.torch_runner
    if hasattr(bench_runner, "mlx_runner"):
        raise ValueError("This script needs the PyTorch/CUDA SGLang runner, not MLX.")
    return bench_runner


def cleanup_sglang_distributed() -> None:
    with contextlib.suppress(Exception):
        cleanup_dist_env_and_memory(shutdown_ray=False)
    with contextlib.suppress(Exception):
        destroy_model_parallel()
    force_clear_model_parallel_globals()
    force_clear_moe_globals()
    with contextlib.suppress(Exception):
        destroy_distributed_environment()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        with contextlib.suppress(Exception):
            torch.distributed.destroy_process_group()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def force_clear_model_parallel_globals() -> None:
    """Clear v0.5.6 parallel_state globals that destroy_model_parallel misses."""
    for name in (
        "_TP",
        "_PP",
        "_MOE_EP",
        "_MOE_TP",
        "_MOE_DP",
        "_ATTN_CP",
        "_ATTN_TP",
        "_PDMUX_PREFILL_TP_GROUP",
    ):
        group = getattr(parallel_state, name, None)
        if group is not None:
            with contextlib.suppress(Exception):
                group.destroy()
            with contextlib.suppress(Exception):
                setattr(parallel_state, name, None)


def force_clear_moe_globals() -> None:
    """Clear MoE globals that can survive between two in-process model loads."""
    with contextlib.suppress(Exception):
        from sglang.srt.eplb import expert_location

        expert_location._global_expert_location_metadata = None


def tokenize_prompts(tokenizer, prompts: Iterable[str], args: argparse.Namespace) -> List[List[int]]:
    input_ids: List[List[int]] = []
    for prompt in prompts:
        if args.apply_chat_template and isinstance(prompt, list):
            ids = tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
                tokenize=True,
            )
        elif args.apply_chat_template and isinstance(prompt, str):
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
            )
        else:
            ids = tokenizer.encode(prompt)
        ids = ids[: args.max_input_tokens]
        if len(ids) < 2:
            raise ValueError("Each prompt must contain at least two tokens.")
        input_ids.append(ids)
    return input_ids


def make_reqs(input_ids: List[List[int]]) -> List[Req]:
    sampling_params = SamplingParams(temperature=0.0, max_new_tokens=1)
    reqs: List[Req] = []
    for i, ids in enumerate(input_ids):
        req = Req(
            rid=i,
            origin_input_text="",
            origin_input_ids=list(ids),
            sampling_params=sampling_params,
        )
        req.fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        set_extend_input_len(req, len(req.fill_ids) - len(req.prefix_indices))
        reqs.append(req)
    return reqs


def set_extend_input_len(req: Req, extend_input_len: int) -> None:
    if hasattr(req, "set_extend_input_len"):
        req.set_extend_input_len(extend_input_len)
    else:
        req.extend_input_len = extend_input_len


def flatten_float(t: torch.Tensor) -> torch.Tensor:
    return t.detach().float().reshape(-1).cpu()


def layer_state_from_output(output) -> torch.Tensor:
    if isinstance(output, tuple):
        hidden = output[0]
        residual = output[1] if len(output) > 1 else None
        if torch.is_tensor(hidden) and torch.is_tensor(residual):
            return hidden + residual
        if torch.is_tensor(hidden):
            return hidden
    if torch.is_tensor(output):
        return output
    raise TypeError(f"Unsupported layer output type: {type(output)}")


def get_decoder_layers(model_runner):
    candidates = [
        getattr(model_runner.model, "model", None),
        getattr(model_runner.model, "language_model", None),
        model_runner.model,
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate.layers
        nested = getattr(candidate, "model", None) if candidate is not None else None
        if nested is not None and hasattr(nested, "layers"):
            return nested.layers
    raise AttributeError("Cannot find decoder layers on model_runner.model")


def register_layer_hooks(model_runner) -> Tuple[Dict[int, torch.Tensor], List[torch.utils.hooks.RemovableHandle]]:
    captures: Dict[int, torch.Tensor] = {}
    layers = get_decoder_layers(model_runner)
    handles = []

    for idx, layer in enumerate(layers):
        def hook(_module, _inputs, output, layer_idx=idx):
            captures[layer_idx] = flatten_float(layer_state_from_output(output))

        handles.append(layer.register_forward_hook(hook))

    return captures, handles


def compute_topk_weights_for_ids(
    router_logits: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_config,
) -> torch.Tensor:
    """Compute current router weights for externally supplied expert ids."""
    if topk_config.num_fused_shared_experts:
        raise NotImplementedError(
            "--force-bf16-moe-routing does not support fused shared experts yet."
        )
    if topk_config.correction_bias is not None:
        raise NotImplementedError(
            "--force-bf16-moe-routing does not support correction_bias yet."
        )
    if topk_config.use_grouped_topk:
        raise NotImplementedError(
            "--force-bf16-moe-routing does not support grouped top-k yet."
        )

    if topk_config.scoring_func == "softmax":
        scores = torch.softmax(router_logits.float(), dim=-1)
    elif topk_config.scoring_func == "sigmoid":
        scores = torch.sigmoid(router_logits.float())
    else:
        scores = router_logits.float()

    valid = topk_ids >= 0
    gather_ids = topk_ids.clamp(min=0).long()
    topk_weights = scores.gather(1, gather_ids).masked_fill(~valid, 0.0)

    if topk_config.renormalize:
        denom = topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        topk_weights = topk_weights / denom

    if topk_config.routed_scaling_factor is not None:
        topk_weights = topk_weights * topk_config.routed_scaling_factor

    return topk_weights.to(torch.float32)


def patch_moe_topk_routing(
    model_runner,
    mode: str,
    routing_cache: Dict[str, dict],
    replay_weights: bool = False,
) -> int:
    """Patch MoE TopK modules to capture or replay expert ids for analysis."""
    from sglang.srt.layers.moe.topk import (
        StandardTopKOutput,
        TopK,
        TopKOutputFormat,
    )

    patched = 0
    for name, module in model_runner.model.named_modules():
        if not isinstance(module, TopK):
            continue

        module.topk_config.output_format = TopKOutputFormat.STANDARD
        original_forward = module._forward_method

        if mode == "capture":

            def capture_forward(
                hidden_states,
                router_logits,
                *args,
                __name=name,
                __original_forward=original_forward,
                **kwargs,
            ):
                topk_output = __original_forward(
                    hidden_states,
                    router_logits,
                    *args,
                    **kwargs,
                )
                if not isinstance(topk_output, StandardTopKOutput):
                    raise TypeError(
                        "Expected StandardTopKOutput while capturing BF16 routing "
                        f"for {__name}, got {type(topk_output)}"
                    )
                routing_cache[__name] = {
                    "topk_ids": topk_output.topk_ids.detach().cpu(),
                    "topk_weights": topk_output.topk_weights.detach().cpu(),
                    "num_tokens": topk_output.topk_ids.shape[0],
                }
                return topk_output

            module._forward_method = capture_forward

        elif mode == "replay":

            def replay_forward(
                hidden_states,
                router_logits,
                *args,
                __name=name,
                __module=module,
                **kwargs,
            ):
                if __name not in routing_cache:
                    raise KeyError(f"No captured BF16 MoE routing found for {__name}")
                topk_ids = routing_cache[__name]["topk_ids"].to(
                    device=router_logits.device,
                    dtype=torch.int32,
                )
                if topk_ids.shape[0] != router_logits.shape[0]:
                    raise ValueError(
                        f"Captured BF16 routing token count mismatch for {__name}: "
                        f"captured={topk_ids.shape[0]} current={router_logits.shape[0]}"
                    )
                if replay_weights:
                    topk_weights = routing_cache[__name]["topk_weights"].to(
                        device=router_logits.device,
                        dtype=torch.float32,
                    )
                else:
                    topk_weights = compute_topk_weights_for_ids(
                        router_logits,
                        topk_ids,
                        __module.topk_config,
                    )
                return StandardTopKOutput(topk_weights, topk_ids, router_logits)

            module._forward_method = replay_forward

        else:
            raise ValueError(f"Unsupported MoE routing patch mode: {mode}")

        patched += 1

    return patched


def matches_any(name: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, name) for pattern in patterns)


def torch_dtype_from_name(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in ("float16", "fp16", "half"):
        return torch.float16
    if normalized in ("bfloat16", "bf16"):
        return torch.bfloat16
    if normalized in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def validate_unquantized_linear_weight(layer, name: str) -> None:
    input_size = getattr(layer, "input_size", None)
    output_size = getattr(
        layer,
        "output_size_per_partition",
        getattr(layer, "output_size", None),
    )
    if input_size is None or output_size is None:
        return
    expected = (output_size, input_size)
    actual = tuple(layer.weight.shape)
    if actual != expected:
        raise ValueError(
            f"Dequantized linear weight has wrong shape for {name}: "
            f"expected={expected} actual={actual}. "
            f"Original module input_size={input_size} output_size={output_size}."
        )


class SafetensorsWeightReader:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.key_to_file: Dict[str, Path] = {}
        self._handles = {}
        self._build_index()

    def _build_index(self) -> None:
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise ImportError(
                "safetensors is required for --bf16-override-module-regex"
            ) from exc

        files = sorted(self.model_path.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No safetensors files found under {self.model_path}")

        for file in files:
            with safe_open(file, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    self.key_to_file[key] = file

    def get(self, key: str) -> torch.Tensor:
        if key not in self.key_to_file:
            raise KeyError(f"Cannot find BF16 tensor {key!r} under {self.model_path}")
        file = self.key_to_file[key]
        if file not in self._handles:
            from safetensors import safe_open

            self._handles[file] = safe_open(file, framework="pt", device="cpu")
        return self._handles[file].get_tensor(key)

    def has(self, key: str) -> bool:
        return key in self.key_to_file

    def close(self) -> None:
        self._handles.clear()


def clear_memory_after_parameter_drop() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def replace_parameter_from_cpu(
    module,
    parameter_name: str,
    value: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Replace one parameter while keeping GPU peak memory low."""
    old_parameter = getattr(module, parameter_name, None)
    if (
        isinstance(old_parameter, torch.nn.Parameter)
        and tuple(old_parameter.shape) == tuple(value.shape)
        and old_parameter.dtype == dtype
    ):
        old_parameter.data.copy_(value.to(device=device, dtype=dtype))
        return

    if parameter_name in getattr(module, "_parameters", {}):
        del module._parameters[parameter_name]
    elif hasattr(module, parameter_name):
        delattr(module, parameter_name)
    clear_memory_after_parameter_drop()

    parameter = torch.nn.Parameter(
        value.to(device=device, dtype=dtype).contiguous(),
        requires_grad=False,
    )
    module.register_parameter(parameter_name, parameter)


def load_direct_weight_from_bf16(
    reader: SafetensorsWeightReader,
    module_name: str,
    module,
    target_dtype: torch.dtype,
) -> bool:
    """Load direct .weight/.bias tensors, including embeddings and norms.

    VocabParallelEmbedding.weight_loader handles padding and TP sharding in
    place, so it avoids allocating a second full embedding tensor on GPU.
    """
    weight_key = f"{module_name}.weight"
    if not reader.has(weight_key) or not hasattr(module, "weight"):
        return False

    weight = reader.get(weight_key)
    parameter = module.weight
    if (
        hasattr(module, "weight_loader")
        and isinstance(parameter, torch.nn.Parameter)
        and parameter.dtype == target_dtype
    ):
        module.weight_loader(parameter, weight.to(dtype=target_dtype))
    else:
        if tuple(weight.shape) != tuple(parameter.shape):
            if weight.dim() != parameter.dim() or any(
                weight.shape[i] > parameter.shape[i] for i in range(weight.dim())
            ):
                raise ValueError(
                    f"BF16 weight shape does not fit {module_name}: "
                    f"loaded={tuple(weight.shape)} current={tuple(parameter.shape)}"
                )
            padded = torch.zeros(tuple(parameter.shape), dtype=target_dtype)
            slices = tuple(slice(0, size) for size in weight.shape)
            padded[slices] = weight.to(dtype=target_dtype)
            weight = padded
        replace_parameter_from_cpu(
            module,
            "weight",
            weight,
            parameter.device,
            target_dtype,
        )

    if hasattr(module, "quant_method") and hasattr(module.quant_method, "embedding"):
        from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod

        module.quant_method = UnquantizedEmbeddingMethod()

    bias_key = f"{module_name}.bias"
    bias = getattr(module, "bias", None)
    if bias is not None and reader.has(bias_key):
        replace_parameter_from_cpu(
            module,
            "bias",
            reader.get(bias_key),
            bias.device,
            target_dtype,
        )
    return True


def load_bf16_linear_weight(reader: SafetensorsWeightReader, module_name: str) -> torch.Tensor:
    if module_name.endswith(".qkv_proj"):
        prefix = module_name[: -len(".qkv_proj")]
        return torch.cat(
            [
                reader.get(f"{prefix}.q_proj.weight"),
                reader.get(f"{prefix}.k_proj.weight"),
                reader.get(f"{prefix}.v_proj.weight"),
            ],
            dim=0,
        )
    if module_name.endswith(".gate_up_proj"):
        prefix = module_name[: -len(".gate_up_proj")]
        return torch.cat(
            [
                reader.get(f"{prefix}.gate_proj.weight"),
                reader.get(f"{prefix}.up_proj.weight"),
            ],
            dim=0,
        )
    return reader.get(f"{module_name}.weight")


def override_fp8_modules_from_bf16(
    model_runner,
    patterns: Iterable[str],
    bf16_model_path: str,
    dtype_name: str,
) -> None:
    patterns = list(patterns)
    if not patterns:
        return

    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.quantization.unquant import (
        UnquantizedFusedMoEMethod,
        UnquantizedLinearMethod,
    )

    target_dtype = torch_dtype_from_name(dtype_name)
    reader = SafetensorsWeightReader(bf16_model_path)
    converted = []
    try:
        for name, module in model_runner.model.named_modules():
            if not matches_any(name, patterns):
                continue

            if isinstance(module, LinearBase):
                weight = load_bf16_linear_weight(reader, name)
                replace_parameter_from_cpu(
                    module,
                    "weight",
                    weight,
                    module.weight.device,
                    target_dtype,
                )
                module.weight_scale = None
                if hasattr(module, "input_scale"):
                    module.input_scale = None
                module.quant_method = UnquantizedLinearMethod()
                validate_unquantized_linear_weight(module, name)
                converted.append(name)

            elif isinstance(module, FusedMoE):
                w13, w2 = load_bf16_fused_moe_weights(reader, name, module, target_dtype)
                replace_parameter_from_cpu(
                    module,
                    "w13_weight",
                    w13,
                    module.w13_weight.device,
                    target_dtype,
                )
                replace_parameter_from_cpu(
                    module,
                    "w2_weight",
                    w2,
                    module.w2_weight.device,
                    target_dtype,
                )
                module.w13_weight_scale = None
                module.w2_weight_scale = None
                if hasattr(module, "w13_input_scale"):
                    module.w13_input_scale = None
                if hasattr(module, "w2_input_scale"):
                    module.w2_input_scale = None
                module.quant_method = UnquantizedFusedMoEMethod(
                    module.use_triton_kernels,
                    module.use_flashinfer_trtllm_moe,
                )
                module.quant_method.create_moe_runner(module, module.moe_runner_config)
                converted.append(name)

            elif load_direct_weight_from_bf16(reader, name, module, target_dtype):
                converted.append(name)
    finally:
        reader.close()

    print(
        f"[bf16-override] replaced {len(converted)} modules from {bf16_model_path}: "
        + ", ".join(converted[:20])
        + (" ..." if len(converted) > 20 else "")
    )


def load_bf16_fused_moe_weights(
    reader: SafetensorsWeightReader,
    module_name: str,
    module,
    target_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    w13_chunks = []
    w2_chunks = []
    for expert_id in range(module.num_local_experts):
        gate = reader.get(f"{module_name}.{expert_id}.gate_proj.weight")
        up = reader.get(f"{module_name}.{expert_id}.up_proj.weight")
        down = reader.get(f"{module_name}.{expert_id}.down_proj.weight")

        w13 = torch.cat([gate, up], dim=0)
        w2 = down
        if module.use_triton_kernels:
            w13 = w13.t().contiguous()
            w2 = w2.t().contiguous()
        w13_chunks.append(w13.to(dtype=target_dtype))
        w2_chunks.append(w2.to(dtype=target_dtype))

    return torch.stack(w13_chunks, dim=0), torch.stack(w2_chunks, dim=0)


def disable_fused_kv_buffer_for_scaled_attention(model_runner) -> None:
    """Avoid fused KV cache writes for attention layers with KV scales.

    Some MoE models, such as Qwen3 MoE, enable a fused set_kv_buffer path that
    currently asserts when the attention layer carries k_scale/v_scale. The
    unfused path supports those scales and is fine for this analysis script.
    """
    for layer in get_decoder_layers(model_runner):
        attn_wrapper = getattr(layer, "self_attn", None)
        attn = getattr(attn_wrapper, "attn", None)
        if attn_wrapper is not None and attn is not None:
            if getattr(attn, "k_scale", None) is not None or getattr(attn, "v_scale", None) is not None:
                if hasattr(attn_wrapper, "compatible_with_fused_kv_buffer"):
                    attn_wrapper.compatible_with_fused_kv_buffer = False


@torch.no_grad()
def run_prefill(model_runner, input_ids: List[List[int]], capture_layers: bool):
    captures: Dict[int, torch.Tensor] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []
    disable_fused_kv_buffer_for_scaled_attention(model_runner)
    if capture_layers:
        captures, handles = register_layer_hooks(model_runner)

    reqs = make_reqs(input_ids)
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
    forward_output = model_runner.forward(forward_batch)
    if hasattr(forward_output, "logits_output"):
        logits_output = forward_output.logits_output
    elif isinstance(forward_output, tuple):
        logits_output = forward_output[0]
    else:
        logits_output = forward_output
    logits = logits_output.next_token_logits.detach().float().cpu()

    for handle in handles:
        handle.remove()
    return {"layers": captures, "logits": logits}


def run_one_model(
    label: str,
    args: argparse.Namespace,
    model_path: str,
    quantization: str | None,
    kv_cache_dtype: str,
    input_ids: List[List[int]] | None = None,
    moe_routing_mode: str | None = None,
    moe_routing_cache: Dict[str, dict] | None = None,
    moe_routing_replay_weights: bool = False,
):
    print(
        f"[{label}] loading model: {model_path} "
        f"quantization={quantization} kv_cache_dtype={kv_cache_dtype}"
    )
    bench_runner = None
    model_runner = None
    tokenizer = None
    try:
        cleanup_sglang_distributed()
        server_args = build_server_args(args, model_path, quantization, kv_cache_dtype)
        port_args = setup_sglang_runtime(server_args)
        bench_runner, tokenizer = load_model(server_args, port_args, gpu_id=0, tp_rank=0)
        model_runner = unwrap_model_runner(bench_runner)
        if moe_routing_mode is not None:
            if moe_routing_cache is None:
                raise ValueError("moe_routing_cache is required when patching MoE routing.")
            patched = patch_moe_topk_routing(
                model_runner,
                moe_routing_mode,
                moe_routing_cache,
                replay_weights=moe_routing_replay_weights,
            )
            print(f"[{label}] {moe_routing_mode} BF16 MoE routing on {patched} TopK modules")
        if label == "fp8":
            override_fp8_modules_from_bf16(
                model_runner,
                args.bf16_override_module_regex,
                args.bf16_model_path,
                args.dtype,
            )
        if input_ids is None:
            prompts = read_prompts(args, tokenizer)
            input_ids = tokenize_prompts(tokenizer, prompts, args)

        print(f"[{label}] running prefill on {len(input_ids)} prompts")
        result = run_prefill(model_runner, input_ids, capture_layers=True)
        return tokenizer, input_ids, result
    finally:
        del model_runner
        del bench_runner
        cleanup_sglang_distributed()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().double().reshape(-1).cpu()
    b = b.detach().double().reshape(-1).cpu()
    denom = a.norm() * b.norm()
    if denom.item() == 0:
        return float("nan")
    value = (a.dot(b) / denom).item()
    if value > 1.0001 or value < -1.0001:
        print(
            "[warn] impossible cosine "
            f"value={value:.8f} a_norm={a.norm().item():.8e} "
            f"b_norm={b.norm().item():.8e}"
        )
    return value


def compare_results(ref, quant, top_k: int):
    rows = []
    ref_layers = ref["layers"]
    quant_layers = quant["layers"]
    common_layers = sorted(set(ref_layers) & set(quant_layers))
    for idx in common_layers:
        a = ref_layers[idx]
        b = quant_layers[idx]
        if a.shape != b.shape:
            rows.append(
                {
                    "layer": idx,
                    "shape": [list(a.shape), list(b.shape)],
                    "cos": None,
                    "rel_l2": None,
                    "mae": None,
                    "max_abs": None,
                }
            )
            continue
        diff = a - b
        rows.append(
            {
                "layer": idx,
                "shape": list(a.shape),
                "cos": cosine(a, b),
                "rel_l2": (diff.norm() / (a.norm() + 1e-12)).item(),
                "mae": diff.abs().mean().item(),
                "max_abs": diff.abs().max().item(),
            }
        )

    ref_logits = ref["logits"]
    quant_logits = quant["logits"]
    logit_diff = ref_logits - quant_logits
    ref_top = torch.topk(ref_logits, k=min(top_k, ref_logits.shape[-1]), dim=-1).indices
    quant_top = torch.topk(quant_logits, k=min(top_k, quant_logits.shape[-1]), dim=-1).indices
    top1_match = (ref_top[:, 0] == quant_top[:, 0]).float().mean().item()
    topk_overlap = []
    for i in range(ref_logits.shape[0]):
        overlap = len(set(ref_top[i].tolist()) & set(quant_top[i].tolist())) / ref_top.shape[1]
        topk_overlap.append(overlap)

    logits_summary = {
        "cos": cosine(ref_logits.reshape(-1), quant_logits.reshape(-1)),
        "rel_l2": (logit_diff.norm() / (ref_logits.norm() + 1e-12)).item(),
        "mae": logit_diff.abs().mean().item(),
        "max_abs": logit_diff.abs().max().item(),
        "top1_match": top1_match,
        f"top{ref_top.shape[1]}_overlap_mean": sum(topk_overlap) / len(topk_overlap),
    }
    return rows, logits_summary


def print_table(rows, logits_summary):
    print("\nLayer comparison:")
    print(f"{'layer':>5} {'cos':>11} {'rel_l2':>11} {'mae':>11} {'max_abs':>11}")
    worst = sorted(
        [r for r in rows if r["cos"] is not None],
        key=lambda r: (r["cos"], -r["rel_l2"]),
    )
    for row in rows:
        cos_s = "NA" if row["cos"] is None else f"{row['cos']:.6f}"
        rel_s = "NA" if row["rel_l2"] is None else f"{row['rel_l2']:.6f}"
        mae_s = "NA" if row["mae"] is None else f"{row['mae']:.6e}"
        max_s = "NA" if row["max_abs"] is None else f"{row['max_abs']:.6e}"
        print(f"{row['layer']:5d} {cos_s:>11} {rel_s:>11} {mae_s:>11} {max_s:>11}")

    print("\nWorst layers by cosine:")
    for row in worst[:10]:
        print(
            f"layer={row['layer']} cos={row['cos']:.6f} "
            f"rel_l2={row['rel_l2']:.6f} mae={row['mae']:.6e}"
        )

    print("\nLogits comparison:")
    for key, value in logits_summary.items():
        print(f"{key}: {value:.6f}")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    bf16_kv_cache_dtype = args.bf16_kv_cache_dtype or args.kv_cache_dtype or "bfloat16"
    fp8_kv_cache_dtype = args.fp8_kv_cache_dtype or args.kv_cache_dtype or "fp8_e4m3"
    moe_routing_cache: Dict[str, dict] = {}

    _, input_ids, ref = run_one_model(
        "bf16",
        args,
        args.bf16_model_path,
        args.bf16_quantization,
        bf16_kv_cache_dtype,
        moe_routing_mode="capture" if args.force_bf16_moe_routing else None,
        moe_routing_cache=moe_routing_cache,
        moe_routing_replay_weights=args.force_bf16_moe_routing,
    )
    _, _, quant = run_one_model(
        "fp8",
        args,
        args.fp8_model_path,
        args.fp8_quantization,
        fp8_kv_cache_dtype,
        input_ids=input_ids,
        moe_routing_mode="replay" if args.force_bf16_moe_routing else None,
        moe_routing_cache=moe_routing_cache,
        moe_routing_replay_weights=args.force_bf16_moe_routing,
    )

    rows, logits_summary = compare_results(ref, quant, args.top_k)
    print_table(rows, logits_summary)

    if args.output_json:
        payload = {
            "layers": rows,
            "logits": logits_summary,
            "num_prompts": len(input_ids),
            "input_lengths": [len(x) for x in input_ids],
            "bf16_kv_cache_dtype": bf16_kv_cache_dtype,
            "fp8_kv_cache_dtype": fp8_kv_cache_dtype,
            "force_bf16_moe_routing": args.force_bf16_moe_routing,
            "captured_moe_routing_layers": sorted(moe_routing_cache),
        }
        Path(args.output_json).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
