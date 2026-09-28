#!/usr/bin/env python3
"""Plot Lab 1 results as required by the PDF.

Each experiment was run as many one-CSV simulations. This script merges those
result folders in memory (one requests.csv per run) and writes figures plus
summary tables under lab1/plots/.

    python lab1/scripts/plot_results.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LAB_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = LAB_DIR / "results"
PLOTS_DIR = LAB_DIR / "plots"

# Shared palette for every figure (not matplotlib C0/C1 defaults).
TEAL = "#0F766E"
AMBER = "#D97706"
INDIGO = "#4338CA"
ROSE = "#E11D48"
CYAN = "#0891B2"
TERRACOTTA = "#C2410C"
NAVY = "#1D4ED8"
CRIMSON = "#BE123C"
VIOLET = "#7C3AED"
CHUNK_COLORS = {
    256: "#0EA5E9",
    512: "#22C55E",
    1024: "#EAB308",
    2048: "#F97316",
    4096: "#A21CAF",
}

# e2e_s = ttft + tpot * (output_len - 1). TTFT/TPOT are ms; e2e is seconds.
def computed_e2e_s(row: pd.Series) -> float:
    return row["ttft_ms"] / 1000.0 + row["tpot_ms"] / 1000.0 * (row["output_len"] - 1)


def style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (7.2, 4.4),
            "figure.dpi": 140,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.color": "#94A3B8",
            "axes.facecolor": "#F8FAFC",
            "figure.facecolor": "white",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "lines.markersize": 6,
            "axes.prop_cycle": plt.cycler(color=list(CHUNK_COLORS.values())),
        }
    )


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path.relative_to(LAB_DIR)}")


def load_requests(folder: Path) -> pd.DataFrame:
    df = pd.read_csv(folder / "requests.csv")
    df["run"] = folder.name
    return df


def load_tokens(folder: Path, request: int = 0) -> pd.DataFrame:
    df = pd.read_csv(folder / "tokens.csv")
    return df[df["request"] == request].sort_values("token")


def collect(pattern: str, fields: dict[str, int]) -> pd.DataFrame:
    """Load every matching results/<name>/requests.csv and parse name fields."""
    regex = re.compile(pattern)
    rows = []
    for folder in sorted(RESULTS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        match = regex.fullmatch(folder.name)
        if not match or not (folder / "requests.csv").exists():
            continue
        df = load_requests(folder)
        for col, group in fields.items():
            df[col] = int(match.group(group))
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"no results matching {pattern} under {RESULTS_DIR}")
    return pd.concat(rows, ignore_index=True)


def add_latency_parts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["computed_e2e_s"] = out.apply(computed_e2e_s, axis=1)
    out["prefill_s"] = out["ttft_ms"] / 1000.0
    out["decode_s"] = out["e2e_s"] - out["prefill_s"]
    out["prefill_pct"] = 100.0 * out["prefill_s"] / out["e2e_s"]
    out["decode_pct"] = 100.0 * out["decode_s"] / out["e2e_s"]
    out["e2e_abs_err_s"] = (out["computed_e2e_s"] - out["e2e_s"]).abs()
    return out


def plot_xy(x, y, xlabel, ylabel, title, path, y2=None, y2label=None,
            color=TEAL, color2=ROSE) -> None:
    fig, ax = plt.subplots()
    ax.plot(x, y, "o-", color=color, label=ylabel)
    if y2 is not None:
        ax.plot(x, y2, "s--", color=color2, label=y2label)
        ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if y2 is None else "seconds")
    ax.set_title(title)
    save(fig, path)


def plot_breakdown(x, prefill_pct, decode_pct, xlabel, title, path, xticklabels=None) -> None:
    fig, ax = plt.subplots()
    labels = xticklabels if xticklabels is not None else [str(v) for v in x]
    idx = range(len(x))
    ax.bar(idx, prefill_pct, color=CYAN, label="Prefill (TTFT / e2e)")
    ax.bar(idx, decode_pct, bottom=prefill_pct, color=TERRACOTTA,
           label="Decode ((e2e − TTFT) / e2e)")
    ax.set_xticks(list(idx), labels, rotation=45, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Share of e2e latency (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.legend()
    save(fig, path)


def plot_warmup1() -> None:
    df = collect(r"warmup1_prompt_(\d+)", {"prompt_len_run": 1})
    df = df[df["request"] == 0].sort_values("prompt_len")
    df = add_latency_parts(df)
    out = PLOTS_DIR / "warmup1"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "summary.csv", index=False)

    plot_xy(
        df["prompt_len"], df["ttft_ms"],
        "prompt_len (tokens)", "TTFT (ms)",
        "Warm-up 1: TTFT vs prompt length",
        out / "01_ttft_vs_prompt.png",
        color=TEAL,
    )
    plot_xy(
        df["prompt_len"], df["tpot_ms"],
        "prompt_len (tokens)", "TPOT (ms)",
        "Warm-up 1: TPOT vs prompt length",
        out / "02_tpot_vs_prompt.png",
        color=AMBER,
    )
    plot_xy(
        df["prompt_len"], df["e2e_s"],
        "prompt_len (tokens)", "Reported e2e (s)",
        "Warm-up 1: e2e from TTFT + TPOT vs reported e2e_s",
        out / "03_e2e_computed_vs_reported.png",
        y2=df["computed_e2e_s"],
        y2label="TTFT + TPOT × (output_len − 1)",
        color=INDIGO,
        color2=ROSE,
    )
    plot_breakdown(
        df["prompt_len"], df["prefill_pct"], df["decode_pct"],
        "prompt_len (tokens)",
        "Warm-up 1: prefill vs decode share of e2e",
        out / "04_prefill_decode_breakdown.png",
    )


def plot_warmup2() -> None:
    df = collect(r"warmup2_output_(\d+)", {"output_len_run": 1})
    df = df[df["request"] == 0].sort_values("output_len")
    df = add_latency_parts(df)
    out = PLOTS_DIR / "warmup2"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "summary.csv", index=False)

    plot_xy(
        df["output_len"], df["ttft_ms"],
        "output_len (tokens)", "TTFT (ms)",
        "Warm-up 2: TTFT vs output length",
        out / "01_ttft_vs_output.png",
        color=TEAL,
    )
    plot_xy(
        df["output_len"], df["tpot_ms"],
        "output_len (tokens)", "TPOT (ms)",
        "Warm-up 2: TPOT vs output length",
        out / "02_tpot_vs_output.png",
        color=AMBER,
    )
    plot_xy(
        df["output_len"], df["e2e_s"],
        "output_len (tokens)", "Reported e2e (s)",
        "Warm-up 2: e2e from TTFT + TPOT vs reported e2e_s",
        out / "03_e2e_computed_vs_reported.png",
        y2=df["computed_e2e_s"],
        y2label="TTFT + TPOT × (output_len − 1)",
        color=INDIGO,
        color2=ROSE,
    )
    plot_breakdown(
        df["output_len"], df["prefill_pct"], df["decode_pct"],
        "output_len (tokens)",
        "Warm-up 2: prefill vs decode share of e2e",
        out / "04_prefill_decode_breakdown.png",
    )


def plot_exp1() -> None:
    raw = collect(r"exp1_batch_(\d+)", {"N": 1})
    df = (
        raw.groupby("N", as_index=False)
        .agg(ttft_ms=("ttft_ms", "mean"), tpot_ms=("tpot_ms", "mean"), n_requests=("request", "count"))
        .sort_values("N")
    )
    out = PLOTS_DIR / "exp1"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "summary.csv", index=False)

    plot_xy(
        df["N"], df["ttft_ms"],
        "N (batch size)", "TTFT (ms)",
        "Experiment 1: TTFT vs batch size",
        out / "01_ttft_vs_batch.png",
        color=TEAL,
    )
    plot_xy(
        df["N"], df["tpot_ms"],
        "N (batch size)", "TPOT (ms)",
        "Experiment 1: TPOT vs batch size",
        out / "02_tpot_vs_batch.png",
        color=AMBER,
    )


def plot_exp2() -> None:
    out = PLOTS_DIR / "exp2"
    out.mkdir(parents=True, exist_ok=True)

    baseline = RESULTS_DIR / "exp2_4a_baseline"
    interrupt = RESULTS_DIR / "exp2_4b_second_prompt_1024"
    tok_a = load_tokens(baseline, 0)
    tok_b = load_tokens(interrupt, 0)
    req_a = load_requests(baseline).iloc[0]
    req_b = load_requests(interrupt)
    req_b0 = req_b[req_b["request"] == 0].iloc[0]

    fig, ax = plt.subplots()
    ax.plot(tok_a["token"], tok_a["gap_ms"], "-", color=NAVY, linewidth=2.2,
            label="4a baseline (request 0 alone)")
    ax.plot(tok_b["token"], tok_b["gap_ms"], "-", color=CRIMSON, linewidth=1.4, alpha=0.9,
            label="4b + second request at t=1.0s")
    ax.set_xlabel("Output token index (request 0)")
    ax.set_ylabel("Token gap (ms)")
    ax.set_title("Experiment 2 / 4a–4b: request 0 token gaps")
    ax.legend()
    save(fig, out / "4a_4b_token_gaps.png")

    fig, ax = plt.subplots()
    labels = ["TPOT (ms)", "Worst gap (ms)"]
    ax.bar([0, 1], [req_a["tpot_ms"], req_a["worst_gap_ms"]], width=0.35,
           color=NAVY, label="4a baseline")
    ax.bar([0.35, 1.35], [req_b0["tpot_ms"], req_b0["worst_gap_ms"]], width=0.35,
           color=CRIMSON, label="4b interrupted")
    ax.set_xticks([0.175, 1.175], labels)
    ax.set_ylabel("milliseconds")
    ax.set_title("Experiment 2 / 4a–4b: request 0 TPOT and worst token gap")
    ax.legend()
    save(fig, out / "4a_4b_tpot_worstgap.png")

    cmp = pd.DataFrame(
        [
            {"stage": "4a", **req_a.to_dict()},
            {"stage": "4b", **req_b0.to_dict()},
        ]
    )
    cmp.to_csv(out / "summary_4a_4b_request0.csv", index=False)

    df4c = collect(r"exp2_4c_second_prompt_(\d+)", {"second_prompt": 1})
    df4c = df4c[df4c["request"] == 0].sort_values("second_prompt")
    df4c.to_csv(out / "summary_4c_request0.csv", index=False)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(df4c["second_prompt"], df4c["worst_gap_ms"], "o-", color=VIOLET, label="Worst gap")
    ax2.plot(df4c["second_prompt"], df4c["tpot_ms"], "s--", color=TEAL, label="TPOT")
    ax1.set_xlabel("Second request prompt_len (tokens)")
    ax1.set_ylabel("Request 0 worst gap (ms)", color=VIOLET)
    ax2.set_ylabel("Request 0 TPOT (ms)", color=TEAL)
    ax1.set_title("Experiment 2 / 4c: request 0 vs interrupting prompt length")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    ax2.grid(False)
    save(fig, out / "4c_worstgap_tpot_vs_second_prompt.png")

    df4d = collect(
        r"exp2_4d_chunk_(\d+)_second_prompt_(\d+)",
        {"chunk_size": 1, "second_prompt": 2},
    )
    df4d = df4d[df4d["request"] == 0].sort_values(["chunk_size", "second_prompt"])
    df4d.to_csv(out / "summary_4d_request0.csv", index=False)

    fig, ax = plt.subplots()
    for chunk, part in df4d.groupby("chunk_size"):
        part = part.sort_values("second_prompt")
        ax.plot(part["second_prompt"], part["worst_gap_ms"], "o-",
                color=CHUNK_COLORS[int(chunk)], label=f"chunk {chunk}")
    ax.set_xlabel("Second request prompt_len (tokens)")
    ax.set_ylabel("Request 0 worst gap (ms)")
    ax.set_title("Experiment 2 / 4d: request 0 worst gap vs chunk size")
    ax.legend(title="--chunk-size")
    save(fig, out / "4d_worstgap_vs_second_prompt.png")

    fig, ax = plt.subplots()
    for chunk, part in df4d.groupby("chunk_size"):
        part = part.sort_values("second_prompt")
        ax.plot(part["second_prompt"], part["tpot_ms"], "o-",
                color=CHUNK_COLORS[int(chunk)], label=f"chunk {chunk}")
    ax.set_xlabel("Second request prompt_len (tokens)")
    ax.set_ylabel("Request 0 TPOT (ms)")
    ax.set_title("Experiment 2 / 4d: request 0 TPOT vs chunk size")
    ax.legend(title="--chunk-size")
    save(fig, out / "4d_tpot_vs_second_prompt.png")


PLOTTERS = {
    "warmup1": plot_warmup1,
    "warmup2": plot_warmup2,
    "exp1": plot_exp1,
    "exp2": plot_exp2,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=list(PLOTTERS), help="plot only these groups")
    args = parser.parse_args()

    style()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    groups = args.only or list(PLOTTERS)
    print(f"Writing plots under {PLOTS_DIR}")
    for name in groups:
        print(f"{name}:")
        PLOTTERS[name]()


if __name__ == "__main__":
    main()
