"""
run_lab.py -- LLM inference lab: simulate requests on one Qwen2.5-32B server.

The server is one Qwen2.5-32B replica with tensor parallelism 4 (4 simulated
A100 GPUs), scheduled like vLLM v1: every step decodes all running requests and
fills the rest of a token budget (--chunk-size) with prompt tokens, splitting
long prompts across steps (chunked prefill). No prefix caching.

Nothing runs on a GPU; step times come from Vidur's prediction tables
(cache/cache_lite_qwen32b_a100_tp2_tp4.tar.gz), via lab/qwen32b_support.py.

Usage (from Vidur-Agent/):
  python lab/run_lab.py INPUT.csv [INPUT.csv ...] [--chunk-size N]

INPUT.csv has one row per request (lines starting with # are comments):
  arrival_time   when the request arrives, in seconds (0 = start)
  prompt_len     prompt tokens
  output_len     output tokens to generate
  count          optional: repeat this row that many times (default 1)
Requests are numbered 0, 1, 2, ... in file order.

For each input file, results go to lab/results/<file name>/:
  requests.csv   one row per request: TTFT, TPOT, end-to-end time, token-gap stats
  tokens.csv     one row per output token: when it came out, gap since the previous one
  steps.csv      one row per model step: when it ran, how long, what it processed
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

LAB_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(LAB_DIR)
sys.path.insert(0, LAB_DIR)
CLUSTER = os.path.join(LAB_DIR, "cluster_qwen32b_tp4.json")
MAX_CONTEXT = 32768      # Qwen2.5-32B context window: prompt + output
MAX_PROMPT = 4096        # prompt tokens per request; the profiled prefill range
MAX_CHUNK = 4096         # largest prefill chunk the prediction tables cover
BATCH_SIZE_CAP = 256     # max requests per step (set in the cluster config)
PER_REQUEST_ROWS = 80    # print one line per request up to this many requests in total


# ── Input ──────────────────────────────────────────────────────────────────────

def load_requests(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    missing = {"arrival_time", "prompt_len", "output_len"} - set(df.columns)
    if missing:
        sys.exit(f"{path}: missing column(s) {sorted(missing)}; "
                 "need arrival_time,prompt_len,output_len[,count]")
    if "count" not in df.columns:
        df["count"] = 1
    df = df.loc[df.index.repeat(df["count"])].reset_index(drop=True)
    df["arrival_time"] = df["arrival_time"].astype(float)
    df["prompt_len"] = df["prompt_len"].astype(int)
    df["output_len"] = df["output_len"].astype(int)

    problems = []
    if len(df) == 0:
        problems.append("no requests")
    if (df["arrival_time"] < 0).any():
        problems.append("arrival_time must be >= 0")
    if (df["prompt_len"] < 1).any() or (df["output_len"] < 1).any():
        problems.append("prompt_len and output_len must be >= 1")
    long_prompt = df["prompt_len"] > MAX_PROMPT
    if long_prompt.any():
        problems.append(f"prompt_len must be <= {MAX_PROMPT} "
                        f"(rows {list(df.index[long_prompt])})")
    too_long = df["prompt_len"] + df["output_len"] > MAX_CONTEXT
    if too_long.any():
        problems.append(f"prompt_len + output_len must be <= {MAX_CONTEXT} "
                        f"(rows {list(df.index[too_long])})")
    if problems:
        sys.exit(f"{path}: " + "; ".join(problems))

    df.insert(0, "request", range(len(df)))
    return df[["request", "arrival_time", "prompt_len", "output_len"]]


# ── One simulation (runs in its own process) ──────────────────────────────────

def simulate(input_path: str, out_dir: str, chunk_size: int) -> None:
    os.chdir(REPO_DIR)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    requests = load_requests(input_path)
    vidur_dir = os.path.join(out_dir, "vidur")   # the simulator's own outputs
    os.makedirs(vidur_dir, exist_ok=True)

    # Vidur's trace format; arrival times are applied separately below.
    trace_path = os.path.join(vidur_dir, "trace.csv")
    requests.rename(columns={
        "request": "request_id",
        "prompt_len": "num_prefill_tokens",
        "output_len": "num_decode_tokens",
    })[["request_id", "num_prefill_tokens", "num_decode_tokens"]].to_csv(trace_path, index=False)

    sys.argv = [
        "vidur",
        "--synthetic_request_generator_config_num_requests", str(len(requests)),
        "--length_generator_config_type", "trace",
        "--trace_request_length_generator_config_trace_file", trace_path,
        "--interval_generator_config_type", "static",
        "--global_scheduler_config_type", "load_aware",
        "--replica_scheduler_config_type", "vllm_v1",
        "--vllm_v1_scheduler_config_chunk_size", str(chunk_size),
        "--vllm_v1_scheduler_config_batch_size_cap", str(BATCH_SIZE_CAP),
        "--no-cache_config_enable_prefix_caching",
        "--cluster_config_replica_groups_config", CLUSTER,
        "--random_forest_execution_time_predictor_config_cache_mode", "require_cache",
        "--metrics_config_no_timestamp",
        "--metrics_config_output_dir", vidur_dir,
    ]

    import qwen32b_support
    qwen32b_support.apply()   # serve predictions from the cache; no fitting

    from vidur.config import SimulationConfig
    from vidur.entities.batch import Batch
    from vidur.events.request_arrival_event import RequestArrivalEvent
    from vidur.simulator import Simulator
    from vidur.utils.random import set_seeds

    # Record every model step and every output token as batches finish.
    steps, tokens, seen, emitted = [], [], {}, {}
    original_on_batch_end = Batch.on_batch_end

    def on_batch_end(batch, time):
        original_on_batch_end(batch, time)
        step = len(steps)
        steps.append({
            "step": step,
            "start_s": batch.scheduled_at,
            "end_s": time,
            "duration_ms": (time - batch.scheduled_at) * 1e3,
            "batch_size": batch.size,
            "prefill_tokens": batch.num_prefill_tokens,
            "decode_tokens": batch.num_decode_tokens,
            "requests": " ".join(str(r.id) for r in batch.requests),
        })
        for request in batch.requests:
            seen[request.id] = request
            # A step that ends with the prompt fully processed yields one output
            # token: the first one after prefill, then one per decode step.
            if request.is_prefill_complete:
                emitted[request.id] = emitted.get(request.id, 0) + 1
                tokens.append({"request": request.id, "token": emitted[request.id],
                               "time_s": time, "step": step})

    Batch.on_batch_end = on_batch_end

    config = SimulationConfig.create_from_cli_args()
    set_seeds(config.seed)
    simulator = Simulator(config)

    # The request generator only knows inter-arrival gaps; give every request
    # its exact arrival time from the input file instead.
    arrival = dict(zip(requests["request"], requests["arrival_time"]))
    pending = [event._request for event in simulator._event_queue]
    simulator._event_queue = []
    for request in pending:
        t = float(arrival[request.id])
        request._arrived_at = request._queued_at = t
        simulator._add_event(RequestArrivalEvent(t, request))

    simulator.run()

    # ── Per-token and per-request results ──
    tok = pd.DataFrame(tokens).sort_values(["request", "token"]).reset_index(drop=True)
    arrived = tok["request"].map(arrival)
    previous = tok.groupby("request")["time_s"].shift()
    tok["gap_ms"] = ((tok["time_s"] - previous.fillna(arrived)) * 1e3).round(3)
    # Did any step since this request's previous token process prompt tokens?
    # Covers both a prefill sharing the token's own step and a prefill-only step
    # that paused this request. Undefined for token 1 (its own prefill).
    starts = np.array([s["start_s"] for s in steps])
    order = np.argsort(starts, kind="stable")
    starts = starts[order]
    prefill_before = np.concatenate(
        [[0], np.cumsum(np.array([s["prefill_tokens"] > 0 for s in steps])[order])])
    since = prefill_before[np.searchsorted(starts, tok["time_s"])] - \
        prefill_before[np.searchsorted(starts, previous.fillna(0))]
    #tok["prefill_in_gap"] = pd.array(np.where(previous.isna(), pd.NA, since > 0), dtype="Int64")
    tok.to_csv(os.path.join(out_dir, "tokens.csv"), index=False)

    rows = []
    for _, req in requests.iterrows():
        r = seen[req["request"]]
        t = tok[tok["request"] == req["request"]]
        decode_gaps = t[t["token"] > 1]
        worst = decode_gaps.loc[decode_gaps["gap_ms"].idxmax()] if len(decode_gaps) else None
        #delayed = int(decode_gaps["prefill_in_gap"].sum())
        rows.append({
            "request": req["request"],
            "arrival_s": req["arrival_time"],
            "prompt_len": req["prompt_len"],
            "output_len": req["output_len"],
            "queue_ms": (r.scheduled_at - r.arrived_at) * 1e3,
            "ttft_ms": (r.prefill_completed_at - r.arrived_at) * 1e3,
            "tpot_ms": ((r.completed_at - r.prefill_completed_at) / (req["output_len"] - 1) * 1e3
                        if req["output_len"] > 1 else float("nan")),
            "e2e_s": r.completed_at - r.arrived_at,
            "worst_gap_ms": worst["gap_ms"] if worst is not None else float("nan"),
            "worst_gap_token": int(worst["token"]) if worst is not None else pd.NA,
            #"tokens_delayed_by_prefill": delayed,
            "restarts": r.num_restarts,
        })
    pd.DataFrame(rows).round(3).to_csv(os.path.join(out_dir, "requests.csv"), index=False)
    pd.DataFrame(steps).round(6).to_csv(os.path.join(out_dir, "steps.csv"), index=False)


# ── Driver ─────────────────────────────────────────────────────────────────────

def print_summary(results: list) -> None:
    frames = []
    for input_path, out_dir in results:
        df = pd.read_csv(os.path.join(out_dir, "requests.csv"))
        df.insert(0, "file", os.path.basename(input_path))
        frames.append(df)
    all_rows = pd.concat(frames, ignore_index=True)

    cols = ["file", "request", "arrival_s", "prompt_len", "output_len",
            "queue_ms", "ttft_ms", "tpot_ms", "e2e_s", "worst_gap_ms"]
    for col in ("request", "prompt_len", "output_len"):
        all_rows[col] = all_rows[col].astype(int)
    if len(all_rows) <= PER_REQUEST_ROWS:
        print(all_rows[cols].round(2).to_string(index=False))
    else:
        per_file = all_rows.groupby("file", sort=False).agg(
            requests=("request", "count"),
            ttft_ms_mean=("ttft_ms", "mean"), ttft_ms_max=("ttft_ms", "max"),
            tpot_ms_mean=("tpot_ms", "mean"), tpot_ms_max=("tpot_ms", "max"),
            worst_gap_ms=("worst_gap_ms", "max"),
        )
        print(per_file.round(2).to_string())
        print("\n(per-request rows are in each file's requests.csv)")
    if (all_rows["restarts"] > 0).any():
        print("\nNote: some requests were preempted and restarted (see the 'restarts' column).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="input CSV file(s)")
    parser.add_argument("--chunk-size", type=int, default=MAX_CHUNK,
                        help=f"token budget per step (max {MAX_CHUNK}, default {MAX_CHUNK})")
    parser.add_argument("--out", default=os.path.join(LAB_DIR, "results"),
                        help="results directory (default lab/results)")
    parser.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    chunk_size = args.chunk_size
    if not 1 <= chunk_size <= MAX_CHUNK:
        sys.exit(f"--chunk-size must be between 1 and {MAX_CHUNK}")

    results = []
    for input_path in args.inputs:
        input_path = os.path.abspath(input_path)
        name = os.path.splitext(os.path.basename(input_path))[0]
        out_dir = os.path.abspath(os.path.join(args.out, name))
        if args.one:
            simulate(input_path, out_dir, chunk_size)
            return

        load_requests(input_path)  # report input mistakes before simulating

        # Each simulation runs in a fresh process: the simulator keeps global state.
        print(f"[run_lab] {os.path.basename(input_path)} ...", flush=True)
        log_path = out_dir + ".log"
        os.makedirs(out_dir, exist_ok=True)
        with open(log_path, "w") as log:
            cmd = [sys.executable, os.path.abspath(__file__), "--one", input_path,
                   "--chunk-size", str(chunk_size), "--out", args.out]
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        if code != 0:
            with open(log_path) as log:
                print(log.read()[-3000:])
            sys.exit(f"[run_lab] {input_path} failed (log: {log_path})")
        results.append((input_path, out_dir))

    print()
    print_summary(results)
    out = os.path.abspath(args.out)
    if out.startswith(os.getcwd() + os.sep):
        out = os.path.relpath(out)
    print(f"\nResults: {out}/<file name>/ (requests.csv, tokens.csv, steps.csv)")


if __name__ == "__main__":
    main()
