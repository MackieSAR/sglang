#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark one offline vLLM batch with prefill/decode step breakdown.

This is intentionally similar in spirit to SGLang's bench_one_batch: it runs
one fixed batch repeatedly and reports model-execution time split by scheduler
step type. The split is based on vLLM's SchedulerOutput:

* prefill: all scheduled requests are still in context phase.
* decode: all scheduled requests already have output tokens.
* mixed: a scheduler step contains both.

The timing wraps EngineCore's model_executor execution, not tokenization or
request-output postprocessing.

Usage:

  # Single data point, SGLang-like output.
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  .venv/bin/python benchmarks/bench_one_batch_vllm.py \
      --model-path /path/to/model \
      --batch 8 --input-len 1024 --output-len 128

  # Sweep multiple data points and append results to result.jsonl.
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  .venv/bin/python benchmarks/bench_one_batch_vllm.py \
      --model-path /path/to/model \
      --batch 1 8 16 --input-len 1024 4096 --output-len 128 \
      --run-name flashinfer_fp8 \
      --attention-backend FLASHINFER --kv-cache-dtype fp8

  # Print per-operator torch profiler table for one decode step.
  SGLANG_TORCH_PROFILER_DIR=/tmp \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  .venv/bin/python benchmarks/bench_one_batch_vllm.py \
      --model-path /path/to/model \
      --batch 8 --input-len 1024 --output-len 128 \
      --profile --profile-stage decode --profile-start-step 64 \
      --profile-record-shapes
"""

import argparse
import itertools
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.outputs import RequestOutput
from vllm.usage.usage_lib import UsageContext
from vllm.v1.core.sched.output import SchedulerOutput


@dataclass
class StepRecord:
    iteration: int
    phase: str
    prefill_tokens: int
    decode_tokens: int
    total_tokens: int
    elapsed_ms: float


@dataclass
class TorchProfileConfig:
    enabled: bool = False
    activities: tuple[str, ...] = ("CPU", "GPU")
    record_shapes: bool = False
    stage: str = "all"
    filename_prefix: str = "profile"
    start_step: int | None = None
    steps: int = 1
    table_row_limit: int = 200
    trace_dir: str = "/tmp"


class OneBatchProfiler:
    def __init__(
        self,
        llm: LLM,
        sync_cuda: bool = True,
        torch_profile_config: TorchProfileConfig | None = None,
    ) -> None:
        self.llm = llm
        self.sync_cuda = sync_cuda
        self.torch_profile_config = torch_profile_config or TorchProfileConfig()
        self.records: list[StepRecord] = []
        self.enabled = False
        self.iteration = -1
        self.batch_size = 0
        self.input_len = 0
        self.output_len = 0
        self._prefill_step_index = 0
        self._decode_step_index = 0

        engine_core_client = llm.llm_engine.engine_core
        core = getattr(engine_core_client, "engine_core", None)
        if core is None:
            raise RuntimeError(
                "bench_one_batch_vllm requires an in-process EngineCore. "
                "Disable VLLM_ENABLE_V1_MULTIPROCESSING for this benchmark."
            )
        batch_queue = getattr(core, "batch_queue", None)
        if batch_queue is not None and len(batch_queue) > 0:
            raise RuntimeError(
                "bench_one_batch_vllm cannot install the profiler after "
                "EngineCore has queued batches. Install it before running requests."
            )
        if batch_queue is not None:
            print(
                "bench_one_batch_vllm: batch queue is enabled; using a "
                "synchronous EngineCore step for SGLang-like per-step timing."
            )

        self.core = core
        self._orig_step_fn = core.step_fn
        core.step_fn = lambda: self._profiled_step(core)

    def uninstall(self) -> None:
        self.core.step_fn = self._orig_step_fn

    def start_iteration(
        self,
        iteration: int,
        batch_size: int,
        input_len: int,
        output_len: int,
    ) -> None:
        self.iteration = iteration
        self.batch_size = batch_size
        self.input_len = input_len
        self.output_len = output_len
        self._prefill_step_index = 0
        self._decode_step_index = 0

    @staticmethod
    def _classify(scheduler_output: SchedulerOutput) -> tuple[str, int, int]:
        prefill_tokens = 0
        decode_tokens = 0

        for req in scheduler_output.scheduled_new_reqs:
            num_tokens = scheduler_output.num_scheduled_tokens.get(req.req_id, 0)
            # New requests are in context phase in the normal generate path.
            prefill_tokens += num_tokens

        cached = scheduler_output.scheduled_cached_reqs
        for req_id, num_output_tokens in zip(cached.req_ids, cached.num_output_tokens):
            num_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if num_output_tokens == 0:
                prefill_tokens += num_tokens
            else:
                decode_tokens += num_tokens

        if prefill_tokens and decode_tokens:
            phase = "mixed"
        elif prefill_tokens:
            phase = "prefill"
        elif decode_tokens:
            phase = "decode"
        else:
            phase = "empty"
        return phase, prefill_tokens, decode_tokens

    def _sync(self) -> None:
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
    def _get_model_runner_output(model_output: Any) -> Any:
        if hasattr(model_output, "get_output"):
            return model_output.get_output()
        return model_output

    def _torch_profiler_activities(self) -> list[torch.profiler.ProfilerActivity]:
        activities = []
        requested = {a.upper() for a in self.torch_profile_config.activities}
        if "CPU" in requested:
            activities.append(torch.profiler.ProfilerActivity.CPU)
        if ("GPU" in requested or "CUDA" in requested) and torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        return activities

    def _profile_name(
        self,
        prefill_tokens: int,
        decode_tokens: int,
        prefill_step_index: int,
        decode_step_index: int,
    ) -> str | None:
        config = self.torch_profile_config
        if not self.enabled or not config.enabled:
            return None

        if prefill_tokens > 0 and config.stage in ("all", "prefill"):
            if prefill_step_index == 0:
                return "prefill"

        if decode_tokens > 0 and config.stage in ("all", "decode"):
            start_step = (
                config.start_step
                if config.start_step is not None
                else self.output_len // 2
            )
            end_step = start_step + config.steps
            if start_step <= decode_step_index < end_step:
                return f"decode_step{decode_step_index}"

        return None

    def _profile_trace_filename(self, profile_name: str) -> str:
        config = self.torch_profile_config
        filename = (
            f"{config.filename_prefix}_batch{self.batch_size}"
            f"_input{self.input_len}_output{self.output_len}"
            f"_iter{self.iteration}_{profile_name}.trace.json"
        )
        return os.path.join(config.trace_dir, filename)

    def _start_torch_profile(
        self, profile_name: str | None
    ) -> torch.profiler.profile | None:
        if profile_name is None:
            return None

        activities = self._torch_profiler_activities()
        if not activities:
            print("Torch profiler skipped: no requested activities are available.")
            return None

        profiler = torch.profiler.profile(
            activities=activities,
            with_stack=True,
            record_shapes=self.torch_profile_config.record_shapes,
        )
        profiler.start()
        return profiler

    def _stop_torch_profile(
        self,
        profiler: torch.profiler.profile | None,
        profile_name: str | None,
    ) -> None:
        if profiler is None or profile_name is None:
            return

        profiler.stop()
        sort_by = (
            "cuda_time_total"
            if torch.profiler.ProfilerActivity.CUDA
            in self._torch_profiler_activities()
            else "self_cpu_time_total"
        )
        print(f"\nTorch profiler ({profile_name})")
        print(
            profiler.key_averages(
                group_by_input_shape=self.torch_profile_config.record_shapes
            ).table(
                sort_by=sort_by,
                row_limit=self.torch_profile_config.table_row_limit,
            )
        )

        trace_filename = self._profile_trace_filename(profile_name)
        os.makedirs(os.path.dirname(os.path.abspath(trace_filename)), exist_ok=True)
        profiler.export_chrome_trace(trace_filename)
        print(f"Torch profiler chrome trace saved to {trace_filename}\n")

    def _profiled_step(self, core) -> tuple[dict[int, Any], bool]:
        if not core.scheduler.has_requests():
            return {}, False

        scheduler_output = core.scheduler.schedule()
        phase, prefill_tokens, decode_tokens = self._classify(scheduler_output)
        prefill_step_index = self._prefill_step_index
        decode_step_index = self._decode_step_index
        profile_name = self._profile_name(
            prefill_tokens,
            decode_tokens,
            prefill_step_index,
            decode_step_index,
        )

        self._sync()
        torch_profiler = self._start_torch_profile(profile_name)
        start = time.perf_counter()
        try:
            future = core.model_executor.execute_model(scheduler_output, non_block=True)
            grammar_output = core.scheduler.get_grammar_bitmask(scheduler_output)
            with (
                core.log_error_detail(scheduler_output),
                core.log_iteration_details(scheduler_output),
            ):
                model_output = future.result()
                if model_output is None:
                    model_output = core.model_executor.sample_tokens(grammar_output)
                model_output = self._get_model_runner_output(model_output)
            self._sync()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
        finally:
            self._stop_torch_profile(torch_profiler, profile_name)

        if prefill_tokens > 0:
            self._prefill_step_index += 1
        if decode_tokens > 0:
            self._decode_step_index += 1

        core._process_aborts_queue()
        engine_core_outputs = core.scheduler.update_from_output(
            scheduler_output, model_output
        )

        total_tokens = scheduler_output.total_num_scheduled_tokens
        if self.enabled and total_tokens > 0:
            self.records.append(
                StepRecord(
                    iteration=self.iteration,
                    phase=phase,
                    prefill_tokens=prefill_tokens,
                    decode_tokens=decode_tokens,
                    total_tokens=total_tokens,
                    elapsed_ms=elapsed_ms,
                )
            )

        return engine_core_outputs, total_tokens > 0


def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--model-path",
        dest="model",
        type=str,
        default=argparse.SUPPRESS,
        help="SGLang-style alias for vLLM's --model.",
    )
    parser.add_argument("--run-name", type=str, default="default")
    parser.add_argument("--input-len", type=int, nargs="+", default=[1024])
    parser.add_argument("--output-len", type=int, nargs="+", default=[16])
    parser.add_argument("--batch-size", "--batch", type=int, nargs="+", default=[1])
    parser.add_argument("--num-iters-warmup", type=int, default=1)
    parser.add_argument(
        "--num-iters",
        type=int,
        default=1,
        help="Repeat each benchmark data point this many times.",
    )
    parser.add_argument("--result-filename", type=str, default="result.jsonl")
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument(
        "--log-decode-step",
        type=int,
        default=0,
        help="Log decode latency by step. The first 5 decode steps are always logged.",
    )
    parser.add_argument(
        "--steady-decode-start-step",
        type=int,
        default=None,
        help=(
            "First decode step index included in steady-state metrics. Defaults "
            "to output_len // 2."
        ),
    )
    parser.add_argument(
        "--disable-cuda-sync",
        action="store_true",
        help="Do not synchronize CUDA around each model execution step.",
    )
    parser.add_argument("--profile", action="store_true", help="Enable torch profiler.")
    parser.add_argument(
        "--profile-record-shapes",
        action="store_true",
        help="Record tensor shapes in torch profiler results.",
    )
    parser.add_argument(
        "--profile-activities",
        type=str,
        nargs="+",
        default=["CPU", "GPU"],
        choices=["CPU", "GPU", "CUDA"],
        help="Torch profiler activities. GPU and CUDA both map to CUDA activity.",
    )
    parser.add_argument(
        "--profile-stage",
        type=str,
        default="all",
        choices=["all", "prefill", "decode"],
        help="Which stage to profile: all, prefill, or decode.",
    )
    parser.add_argument(
        "--profile-filename-prefix",
        type=str,
        default="profile",
        help="Prefix for exported torch profiler chrome traces.",
    )
    parser.add_argument(
        "--profile-start-step",
        type=int,
        default=None,
        help="Decode step to start profiling. Defaults to output_len // 2.",
    )
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=1,
        help="Number of decode steps to profile.",
    )
    parser.add_argument(
        "--profile-table-row-limit",
        type=int,
        default=200,
        help="Maximum rows printed in the torch profiler operator table.",
    )
    parser = EngineArgs.add_cli_args(parser)
    parser.set_defaults(enable_prefix_caching=False)
    return parser


def make_dummy_prompts(
    batch_size: int, input_len: int, vocab_size: int, seed: int | None
) -> list[PromptType]:
    rng = np.random.default_rng(seed)
    # Avoid tiny token ids that are often special tokens.
    low = 100 if vocab_size > 1000 else 0
    high = max(vocab_size, low + 1)
    token_ids = rng.integers(low, high, size=(batch_size, input_len))
    return [{"prompt_token_ids": row.tolist()} for row in token_ids]


def run_one_batch(
    llm: LLM,
    prompts: list[PromptType],
    sampling_params: SamplingParams,
) -> list[RequestOutput]:
    llm.enqueue(prompts, sampling_params=sampling_params, use_tqdm=False)
    outputs = llm.wait_for_completion(RequestOutput, use_tqdm=False)
    return outputs


def summarize_phase_records(records: list[StepRecord]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for phase in ("prefill", "decode", "mixed"):
        phase_records = [r for r in records if r.phase == phase]
        elapsed_ms = sum(r.elapsed_ms for r in phase_records)
        prefill_tokens = sum(r.prefill_tokens for r in phase_records)
        decode_tokens = sum(r.decode_tokens for r in phase_records)
        total_tokens = sum(r.total_tokens for r in phase_records)
        per_step = [r.elapsed_ms for r in phase_records]
        summary[phase] = {
            "steps": len(phase_records),
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            "total_tokens": total_tokens,
            "total_ms": elapsed_ms,
            "mean_step_ms": statistics.mean(per_step) if per_step else 0.0,
            "p50_step_ms": statistics.median(per_step) if per_step else 0.0,
            "tokens_per_second": total_tokens / (elapsed_ms / 1000.0)
            if elapsed_ms > 0
            else 0.0,
        }
    summary["total"] = {
        "steps": len(records),
        "total_tokens": sum(r.total_tokens for r in records),
        "total_ms": sum(r.elapsed_ms for r in records),
    }
    return summary


def print_phase_summary(summary: dict[str, Any]) -> None:
    print("\nPhase breakdown, model execution only")
    print(
        f"{'phase':<10} {'steps':>8} {'prefill_tok':>12} {'decode_tok':>11} "
        f"{'total_ms':>12} {'mean_ms':>10} {'tok/s':>12}"
    )
    for phase in ("prefill", "decode", "mixed"):
        row = summary[phase]
        print(
            f"{phase:<10} {row['steps']:>8} {row['prefill_tokens']:>12} "
            f"{row['decode_tokens']:>11} {row['total_ms']:>12.3f} "
            f"{row['mean_step_ms']:>10.3f} {row['tokens_per_second']:>12.2f}"
        )
    total = summary["total"]
    print(
        f"{'total':<10} {total['steps']:>8} {'-':>12} {'-':>11} "
        f"{total['total_ms']:>12.3f} {'-':>10} {'-':>12}"
    )


def print_sglang_style_result(
    run_name: str,
    records: list[StepRecord],
    batch_size: int,
    input_len: int,
    output_len: int,
    log_decode_step: int,
    steady_decode_start_step: int | None,
) -> dict[str, Any]:
    prefill_records = [r for r in records if r.phase == "prefill"]
    mixed_records = [r for r in records if r.phase == "mixed"]
    decode_records = [r for r in records if r.phase == "decode"]

    prefill_latency = sum(r.elapsed_ms for r in prefill_records) / 1000.0
    prefill_tokens = sum(r.prefill_tokens for r in prefill_records)
    prefill_throughput = (
        prefill_tokens / prefill_latency if prefill_latency > 0 else 0.0
    )
    print(
        f"Prefill. latency: {prefill_latency:6.5f} s, "
        f"throughput: {prefill_throughput:9.2f} token/s"
    )

    decode_latencies = []
    for i, record in enumerate(decode_records):
        latency = record.elapsed_ms / 1000.0
        throughput = record.decode_tokens / latency if latency > 0 else 0.0
        decode_latencies.append(latency)
        if i < 5 or (log_decode_step > 0 and i % log_decode_step == 0):
            print(
                f"Decode {i}. Batch size: {batch_size}, "
                f"latency: {latency:6.5f} s, "
                f"throughput: {throughput:9.2f} token/s"
            )

    measurement_results: dict[str, Any] = {
        "run_name": run_name,
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
        "prefill_latency": prefill_latency,
        "prefill_throughput": prefill_throughput,
    }

    if mixed_records:
        mixed_latency = sum(r.elapsed_ms for r in mixed_records) / 1000.0
        mixed_prefill_tokens = sum(r.prefill_tokens for r in mixed_records)
        mixed_decode_tokens = sum(r.decode_tokens for r in mixed_records)
        mixed_total_tokens = sum(r.total_tokens for r in mixed_records)
        mixed_throughput = (
            mixed_total_tokens / mixed_latency if mixed_latency > 0 else 0.0
        )
        print(
            f"Mixed.   steps: {len(mixed_records)}, "
            f"latency: {mixed_latency:6.5f} s, "
            f"prefill tokens: {mixed_prefill_tokens}, "
            f"decode tokens: {mixed_decode_tokens}, "
            f"throughput: {mixed_throughput:9.2f} token/s"
        )
        measurement_results["mixed_steps"] = len(mixed_records)
        measurement_results["mixed_latency"] = mixed_latency
        measurement_results["mixed_prefill_tokens"] = mixed_prefill_tokens
        measurement_results["mixed_decode_tokens"] = mixed_decode_tokens
        measurement_results["mixed_total_tokens"] = mixed_total_tokens
        measurement_results["mixed_throughput"] = mixed_throughput
    else:
        measurement_results["mixed_steps"] = 0

    if decode_latencies:
        med_decode_latency = float(statistics.median(decode_latencies))
        med_decode_throughput = (
            batch_size / med_decode_latency if med_decode_latency > 0 else 0.0
        )
        print(
            f"Decode.  median latency: {med_decode_latency:6.5f} s, "
            f"median throughput: {med_decode_throughput:9.2f} token/s"
        )
        measurement_results["median_decode_latency"] = med_decode_latency
        measurement_results["median_decode_throughput"] = med_decode_throughput

    steady_start = (
        output_len // 2
        if steady_decode_start_step is None
        else steady_decode_start_step
    )
    steady_decode_records = decode_records[max(0, steady_start) :]
    if steady_decode_records:
        steady_decode_latencies = [r.elapsed_ms / 1000.0 for r in steady_decode_records]
        steady_decode_total_latency = sum(steady_decode_latencies)
        steady_decode_tokens = sum(r.decode_tokens for r in steady_decode_records)
        steady_decode_throughput = (
            steady_decode_tokens / steady_decode_total_latency
            if steady_decode_total_latency > 0
            else 0.0
        )
        steady_decode_p50_latency = float(statistics.median(steady_decode_latencies))
        steady_decode_mean_latency = float(statistics.mean(steady_decode_latencies))
        steady_decode_min_tokens_seen = min(r.decode_tokens for r in steady_decode_records)
        steady_decode_max_tokens_seen = max(r.decode_tokens for r in steady_decode_records)
        print(
            f"Decode steady. steps: {len(steady_decode_records)}, "
            f"start_step: {max(0, steady_start)}, "
            f"tokens/step: {steady_decode_min_tokens_seen}-{steady_decode_max_tokens_seen}, "
            f"mean latency: {steady_decode_mean_latency:6.5f} s, "
            f"p50 latency: {steady_decode_p50_latency:6.5f} s, "
            f"throughput: {steady_decode_throughput:9.2f} token/s"
        )
        measurement_results["steady_decode_steps"] = len(steady_decode_records)
        measurement_results["steady_decode_start_step"] = max(0, steady_start)
        measurement_results["steady_decode_total_tokens"] = steady_decode_tokens
        measurement_results["steady_decode_total_latency"] = steady_decode_total_latency
        measurement_results["steady_decode_mean_latency"] = steady_decode_mean_latency
        measurement_results["steady_decode_p50_latency"] = steady_decode_p50_latency
        measurement_results["steady_decode_throughput"] = steady_decode_throughput
        measurement_results["steady_decode_min_tokens_per_step"] = (
            steady_decode_min_tokens_seen
        )
        measurement_results["steady_decode_max_tokens_per_step"] = (
            steady_decode_max_tokens_seen
        )
    else:
        print(
            "Decode steady. no steps matched "
            f"(start_step={max(0, steady_start)}, decode_steps={len(decode_records)})"
        )
        measurement_results["steady_decode_steps"] = 0

    total_latency = sum(r.elapsed_ms for r in records) / 1000.0
    total_tokens = (input_len + output_len) * batch_size
    overall_throughput = total_tokens / total_latency if total_latency > 0 else 0.0
    print(
        f"Total. latency: {total_latency:6.3f} s, "
        f"throughput: {overall_throughput:9.2f} token/s"
    )
    measurement_results["total_latency"] = total_latency
    measurement_results["overall_throughput"] = overall_throughput
    measurement_results["model_execution_scope"] = (
        "EngineCore model_executor.execute_model, excluding tokenization and "
        "request-output postprocessing"
    )
    return measurement_results


def run_profiled_one_batch(
    llm: LLM,
    profiler: OneBatchProfiler,
    prompts: list[PromptType],
    sampling_params: SamplingParams,
    iteration: int,
    batch_size: int,
    input_len: int,
    output_len: int,
) -> list[StepRecord]:
    start_idx = len(profiler.records)
    profiler.start_iteration(iteration, batch_size, input_len, output_len)
    profiler.enabled = True
    try:
        run_one_batch(llm, prompts, sampling_params)
    finally:
        profiler.enabled = False
    return profiler.records[start_idx:]


def main(args: argparse.Namespace) -> None:
    engine_args = EngineArgs.from_cli_args(args)
    vllm_config = engine_args.create_engine_config(UsageContext.LLM_CLASS)
    vocab_size = vllm_config.model_config.get_vocab_size()
    trace_dir = os.environ.get(
        "VLLM_TORCH_PROFILER_DIR",
        os.environ.get("SGLANG_TORCH_PROFILER_DIR", "/tmp"),
    )
    torch_profile_config = TorchProfileConfig(
        enabled=args.profile,
        activities=tuple(args.profile_activities),
        record_shapes=args.profile_record_shapes,
        stage=args.profile_stage,
        filename_prefix=args.profile_filename_prefix,
        start_step=args.profile_start_step,
        steps=max(1, args.profile_steps),
        table_row_limit=args.profile_table_row_limit,
        trace_dir=trace_dir,
    )

    llm = LLM.from_engine_args(engine_args)
    profiler = OneBatchProfiler(
        llm,
        sync_cuda=not args.disable_cuda_sync,
        torch_profile_config=torch_profile_config,
    )

    try:
        warmup_batch_size = args.batch_size[0]
        warmup_input_len = args.input_len[0]
        warmup_output_len = min(32, args.output_len[0])
        warmup_prompts = make_dummy_prompts(
            batch_size=warmup_batch_size,
            input_len=warmup_input_len,
            vocab_size=vocab_size,
            seed=engine_args.seed,
        )
        warmup_sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=warmup_output_len,
            detokenize=False,
        )

        print("Warmup ...")
        for _ in range(args.num_iters_warmup):
            run_one_batch(llm, warmup_prompts, warmup_sampling_params)

        print("Benchmark ...")
        result_list = []
        iteration = 0
        for batch_size, input_len, output_len in itertools.product(
            args.batch_size, args.input_len, args.output_len
        ):
            prompts = make_dummy_prompts(
                batch_size=batch_size,
                input_len=input_len,
                vocab_size=vocab_size,
                seed=engine_args.seed,
            )
            sampling_params = SamplingParams(
                temperature=1.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=output_len,
                detokenize=False,
            )
            for repeat in range(args.num_iters):
                if args.num_iters > 1:
                    print(
                        f"\nRun {repeat}. batch_size={batch_size}, "
                        f"input_len={input_len}, output_len={output_len}"
                    )
                records = run_profiled_one_batch(
                    llm,
                    profiler,
                    prompts,
                    sampling_params,
                    iteration=iteration,
                    batch_size=batch_size,
                    input_len=input_len,
                    output_len=output_len,
                )
                result = print_sglang_style_result(
                    args.run_name,
                    records,
                    batch_size,
                    input_len,
                    output_len,
                    args.log_decode_step,
                    args.steady_decode_start_step,
                )
                result["iteration"] = repeat
                result_list.append(result)
                iteration += 1
    finally:
        profiler.uninstall()

    if args.result_filename:
        with open(args.result_filename, "a") as f:
            for result in result_list:
                f.write(json.dumps(result) + "\n")

    if args.json_output is not None:
        payload = {
            "config": {
                "model": args.model,
                "input_len": args.input_len,
                "output_len": args.output_len,
                "batch_size": args.batch_size,
                "num_iters": args.num_iters,
                "num_iters_warmup": args.num_iters_warmup,
                "sync_cuda": not args.disable_cuda_sync,
                "profile": args.profile,
                "profile_stage": args.profile_stage,
                "profile_start_step": args.profile_start_step,
                "profile_steps": args.profile_steps,
                "profile_trace_dir": trace_dir,
                "steady_decode_start_step": args.steady_decode_start_step,
            },
            "summary": summarize_phase_records(profiler.records),
            "results": result_list,
            "steps": [asdict(r) for r in profiler.records],
        }
        with open(args.json_output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote JSON results to {args.json_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="vLLM one-batch benchmark with prefill/decode breakdown.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single data point, SGLang-like output.
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \\
  .venv/bin/python benchmarks/bench_one_batch_vllm.py \\
      --model-path /path/to/model \\
      --batch 8 --input-len 1024 --output-len 128

  # Sweep multiple data points and append results to result.jsonl.
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \\
  .venv/bin/python benchmarks/bench_one_batch_vllm.py \\
      --model-path /path/to/model \\
      --batch 1 8 16 --input-len 1024 4096 --output-len 128 \\
      --run-name flashinfer_fp8 \\
      --attention-backend FLASHINFER --kv-cache-dtype fp8

  # Print per-operator torch profiler table for one decode step.
  SGLANG_TORCH_PROFILER_DIR=/tmp \\
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \\
  .venv/bin/python benchmarks/bench_one_batch_vllm.py \\
      --model-path /path/to/model \\
      --batch 8 --input-len 1024 --output-len 128 \\
      --profile --profile-stage decode --profile-start-step 64 \\
      --profile-record-shapes
""",
    )
    main(add_cli_args(parser).parse_args())
