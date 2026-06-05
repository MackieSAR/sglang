#!/usr/bin/env python3
"""Poll SGLang PD worker load endpoints and print a compact live table."""

import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path


# Edit this command directly when you want this script to run a benchmark.
# Keep it as a Python list so paths and arguments are unambiguous.
#
# Example:
# BENCH_SERVING_CMD = [
#     "python", "-m", "sglang.bench_serving",
#     "--backend", "sglang",
#     "--host", "127.0.0.1",
#     "--port", "8000",
#     "--model", "/path/to/model",
#     "--tokenizer", "/path/to/model",
#     "--dataset-name", "random",
#     "--random-input-len", "1710",
#     "--random-output-len", "128",
#     "--random-range-ratio", "0.0",
#     "--num-prompts", "512",
#     "--request-rate", "16",
#     "--pd-separated",
#     "--ready-check-timeout-sec", "0",
# ]
BENCH_SERVING_CMD = []


class GpuSampler:
    def __init__(self, gpu_ids: list[str], interval: float):
        self.gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id is not None]
        self.interval = interval
        self.samples = {gpu_id: [] for gpu_id in self.gpu_ids}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        if not self.gpu_ids:
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(self.interval, 0.1) + 1.0)

    def _run(self):
        while not self.stop_event.is_set():
            for gpu_id in self.gpu_ids:
                stats = safe_fetch_gpu_stats(gpu_id)
                with self.lock:
                    self.samples[gpu_id].append(stats)
            self.stop_event.wait(self.interval)

    def summarize_since_last(self, gpu_id: str) -> dict:
        if gpu_id is None:
            return {}
        with self.lock:
            samples = self.samples.get(gpu_id, [])
            self.samples[gpu_id] = []
        if not samples:
            return {}

        gpu_utils = [to_float(x.get("gpu_util")) for x in samples]
        mem_utils = [to_float(x.get("mem_util")) for x in samples]
        powers = [to_float(x.get("power")) for x in samples]
        gpu_utils = [x for x in gpu_utils if x is not None]
        mem_utils = [x for x in mem_utils if x is not None]
        powers = [x for x in powers if x is not None]
        if not gpu_utils and any(x.get("gpu_util") == "ERR" for x in samples):
            return {"gpu_summary": "ERR"}
        return {
            "gpu_avg": sum(gpu_utils) / len(gpu_utils) if gpu_utils else None,
            "gpu_max": max(gpu_utils) if gpu_utils else None,
            "mem_max": max(mem_utils) if mem_utils else None,
            "power_max": max(powers) if powers else None,
            "num_gpu_samples": len(samples),
        }


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_url(url: str) -> str:
    url = url.rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url


def fetch_load(base_url: str, timeout: float) -> dict:
    query = urllib.parse.urlencode({"include": "all"})
    url = f"{normalize_url(base_url)}/v1/loads?{query}"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_gpu_stats(gpu_id: str) -> dict:
    cmd = [
        "nvidia-smi",
        f"--id={gpu_id}",
        "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    values = [x.strip() for x in output.split(",")]
    if len(values) != 5:
        return {}
    return {
        "gpu_util": values[0],
        "mem_util": values[1],
        "mem_used": values[2],
        "mem_total": values[3],
        "power": values[4],
    }


def safe_fetch_gpu_stats(gpu_id: str) -> dict:
    try:
        return fetch_gpu_stats(gpu_id)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"gpu_util": "ERR", "mem_util": "ERR", "power": "ERR"}


def get_path(data: dict, path: str, default="-"):
    cur = data
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur or cur[key] is None:
            return default
        cur = cur[key]
    return cur


def unwrap_load_response(data: dict) -> dict:
    """Return one display row from /v1/loads JSON.

    Newer SGLang returns {"loads": [...], "aggregate": {...}}. Older/internal
    callers may return a single load dict. Keep both working.
    """
    loads = data.get("loads")
    if not isinstance(loads, list):
        return data
    if not loads:
        return {}
    if len(loads) == 1:
        return loads[0]

    aggregate = data.get("aggregate") or {}
    merged = {
        "num_running_reqs": aggregate.get("total_running_reqs", "-"),
        "num_waiting_reqs": aggregate.get("total_waiting_reqs", "-"),
        "token_usage": aggregate.get("avg_token_usage", "-"),
        "gen_throughput": aggregate.get("avg_throughput", "-"),
        "utilization": aggregate.get("avg_utilization", "-"),
    }

    disagg_rows = [row.get("disaggregation") or {} for row in loads]
    modes = {row.get("mode") for row in disagg_rows if row.get("mode")}
    mode = next(iter(modes), "-") if len(modes) <= 1 else "mixed"
    merged["disaggregation"] = {
        "mode": mode,
        "prefill_prealloc_queue_reqs": sum(
            row.get("prefill_prealloc_queue_reqs", 0) for row in disagg_rows
        ),
        "prefill_inflight_queue_reqs": sum(
            row.get("prefill_inflight_queue_reqs", 0) for row in disagg_rows
        ),
        "decode_prealloc_queue_reqs": sum(
            row.get("decode_prealloc_queue_reqs", 0) for row in disagg_rows
        ),
        "decode_transfer_queue_reqs": sum(
            row.get("decode_transfer_queue_reqs", 0) for row in disagg_rows
        ),
        "decode_retracted_queue_reqs": sum(
            row.get("decode_retracted_queue_reqs", 0) for row in disagg_rows
        ),
    }

    latencies = [
        row.get("kv_transfer_latency_ms")
        for row in disagg_rows
        if row.get("kv_transfer_latency_ms") is not None
    ]
    speeds = [
        row.get("kv_transfer_speed_gb_s")
        for row in disagg_rows
        if row.get("kv_transfer_speed_gb_s") is not None
    ]
    if latencies:
        merged["disaggregation"]["kv_transfer_latency_ms"] = sum(latencies) / len(
            latencies
        )
    if speeds:
        merged["disaggregation"]["kv_transfer_speed_gb_s"] = sum(speeds)
    return merged


def fmt_num(value, width: int, precision: int = 2) -> str:
    if value == "-":
        return f"{value:>{width}}"
    if isinstance(value, float):
        return f"{value:>{width}.{precision}f}"
    return f"{str(value):>{width}}"


def build_row(role: str, data: dict, gpu_stats: dict = None) -> str:
    data = unwrap_load_response(data)
    disagg = data.get("disaggregation") or {}
    gpu_stats = gpu_stats or {}
    if "gpu_summary" in gpu_stats:
        gpu = gpu_stats["gpu_summary"]
    elif "gpu_avg" in gpu_stats or "gpu_max" in gpu_stats:
        gpu = (
            f"avg{fmt_compact(gpu_stats.get('gpu_avg'))}%"
            f"/max{fmt_compact(gpu_stats.get('gpu_max'))}%"
            f"/m{fmt_compact(gpu_stats.get('mem_max'))}%"
            f"/p{fmt_compact(gpu_stats.get('power_max'))}W"
        )
    else:
        gpu = (
            f"{gpu_stats.get('gpu_util', '-')}%"
            f"/{gpu_stats.get('mem_util', '-')}%"
            f"/{gpu_stats.get('power', '-')}W"
        )
    mode = disagg.get("mode", "-")
    if mode == "prefill":
        q1 = disagg.get("prefill_prealloc_queue_reqs", "-")
        q2 = disagg.get("prefill_inflight_queue_reqs", "-")
        qlabel = f"prealloc={q1},inflight={q2}"
    elif mode == "decode":
        q1 = disagg.get("decode_prealloc_queue_reqs", "-")
        q2 = disagg.get("decode_transfer_queue_reqs", "-")
        q3 = disagg.get("decode_retracted_queue_reqs", "-")
        qlabel = f"prealloc={q1},transfer={q2},retract={q3}"
    else:
        qlabel = "-"

    return (
        f"{role:<8}"
        f"{fmt_num(get_path(data, 'num_running_reqs'), 8)}"
        f"{fmt_num(get_path(data, 'num_waiting_reqs'), 8)}"
        f"{fmt_num(get_path(data, 'token_usage'), 10, 4)}"
        f"{fmt_num(get_path(data, 'gen_throughput'), 12)}"
        f"{fmt_num(disagg.get('kv_transfer_latency_ms', '-'), 12)}"
        f"{fmt_num(disagg.get('kv_transfer_speed_gb_s', '-'), 12)}"
        f"{gpu:>30}"
        f"  {qlabel}"
    )


def fmt_compact(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.0f}"
    return str(value)


def build_header() -> str:
    return (
        f"{'role':<8}"
        f"{'running':>8}"
        f"{'waiting':>8}"
        f"{'tok_use':>10}"
        f"{'gen_tok/s':>12}"
        f"{'kv_lat_ms':>12}"
        f"{'kv_GB/s':>12}"
        f"{'gpu(avg/max)/mem/pwr':>30}"
        "  disagg_queues"
    )


def emit(line: str, log_file=None):
    print(line, flush=True)
    if log_file is not None:
        log_file.write(line + "\n")
        log_file.flush()


def observe_once(args, step: int, log_file=None, gpu_sampler: GpuSampler = None):
    emit(f"\n[{datetime.now().strftime('%H:%M:%S')}] sample={step}", log_file)
    emit(build_header(), log_file)
    for role, url, gpu_id in (
        ("prefill", args.prefill, args.prefill_gpu),
        ("decode", args.decode, args.decode_gpu),
    ):
        try:
            if gpu_sampler is not None:
                gpu_stats = gpu_sampler.summarize_since_last(gpu_id)
            else:
                gpu_stats = safe_fetch_gpu_stats(gpu_id) if gpu_id is not None else None
            emit(build_row(role, fetch_load(url, args.timeout), gpu_stats), log_file)
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
        ) as exc:
            emit(f"{role:<8}ERROR: {exc}", log_file)


def observe_loop(args, stop_event: threading.Event, log_file=None, gpu_sampler=None):
    step = 0
    while not stop_event.is_set() and (args.count <= 0 or step < args.count):
        step += 1
        observe_once(args, step, log_file, gpu_sampler)
        stop_event.wait(args.interval)


def run_bench_command(cmd: list[str], bench_log_path: Path) -> int:
    emit("\nStarting bench_serving command:")
    emit(" ".join(cmd))
    with bench_log_path.open("w", encoding="utf-8") as bench_log:
        bench_log.write(" ".join(cmd) + "\n\n")
        bench_log.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            bench_log.write(line)
            bench_log.flush()
        return proc.wait()


def main():
    parser = argparse.ArgumentParser(
        description="Observe SGLang PD prefill/decode load from /v1/loads."
    )
    parser.add_argument("--prefill", required=True, help="Prefill worker URL.")
    parser.add_argument("--decode", required=True, help="Decode worker URL.")
    parser.add_argument("--interval", type=float, default=1.0, help="Poll interval.")
    parser.add_argument("--count", type=int, default=0, help="0 means run forever.")
    parser.add_argument("--timeout", type=float, default=2.0, help="HTTP timeout.")
    parser.add_argument("--prefill-gpu", help="Optional GPU id for prefill worker.")
    parser.add_argument("--decode-gpu", help="Optional GPU id for decode worker.")
    parser.add_argument(
        "--gpu-sample-interval",
        type=float,
        default=0.05,
        help="High-frequency GPU polling interval used between load samples.",
    )
    parser.add_argument(
        "--log-dir",
        default="pd_observe_logs",
        help="Directory for observer and bench logs.",
    )
    parser.add_argument(
        "--observe-only",
        action="store_true",
        help="Do not run BENCH_SERVING_CMD; only observe until interrupted/count ends.",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    observe_log_path = log_dir / f"pd_observe_{run_id}.log"
    bench_log_path = log_dir / f"bench_serving_{run_id}.log"

    should_run_bench = bool(BENCH_SERVING_CMD) and not args.observe_only
    if not should_run_bench:
        if not BENCH_SERVING_CMD and not args.observe_only:
            emit("BENCH_SERVING_CMD is empty; running observe-only mode.")
        with observe_log_path.open("w", encoding="utf-8") as observe_log:
            emit(f"Observer log: {observe_log_path}", observe_log)
            stop_event = threading.Event()
            gpu_sampler = GpuSampler(
                [args.prefill_gpu, args.decode_gpu], args.gpu_sample_interval
            )
            gpu_sampler.start()
            try:
                observe_loop(args, stop_event, observe_log, gpu_sampler)
            except KeyboardInterrupt:
                emit("\nStopped by user.", observe_log)
            finally:
                gpu_sampler.stop()
        return

    stop_event = threading.Event()
    gpu_sampler = GpuSampler([args.prefill_gpu, args.decode_gpu], args.gpu_sample_interval)
    gpu_sampler.start()
    with observe_log_path.open("w", encoding="utf-8") as observe_log:
        emit(f"Observer log: {observe_log_path}", observe_log)
        emit(f"Bench log: {bench_log_path}", observe_log)
        observer = threading.Thread(
            target=observe_loop,
            args=(args, stop_event, observe_log, gpu_sampler),
            daemon=True,
        )
        observer.start()
        time.sleep(min(args.interval, 0.5))
        try:
            return_code = run_bench_command(BENCH_SERVING_CMD, bench_log_path)
        finally:
            stop_event.set()
            observer.join(timeout=max(args.interval, 1.0) + 1.0)
            observe_once(args, 0, observe_log, gpu_sampler)
            gpu_sampler.stop()

    if return_code != 0:
        emit(f"bench_serving exited with code {return_code}")
        sys.exit(return_code)
    emit("bench_serving finished successfully.")


if __name__ == "__main__":
    main()
