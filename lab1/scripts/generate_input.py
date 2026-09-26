#!/usr/bin/env python3
"""Generate labeled CSVs for every Lab 1 experiment.

Run from anywhere:

    python lab1/scripts/generate_input.py

Writes one CSV per simulation into lab1/inputs/. Chunk size is a run_lab.py
flag, not a CSV column, so Experiment 2 / 4d gets copies of the 4c files with
the chunk size in the filename. Use that same value with --chunk-size.

On W135, from Vidur-Agent/:

    uv run python lab1/scripts/generate_input.py
    uv run python lab1/run_lab.py lab1/inputs/warmup1_*.csv
    uv run python lab1/run_lab.py lab1/inputs/warmup2_*.csv
    uv run python lab1/run_lab.py lab1/inputs/exp1_*.csv
    uv run python lab1/run_lab.py lab1/inputs/exp2_4a_*.csv
    uv run python lab1/run_lab.py lab1/inputs/exp2_4b_*.csv
    uv run python lab1/run_lab.py lab1/inputs/exp2_4c_*.csv
    uv run python lab1/run_lab.py lab1/inputs/exp2_4d_chunk_4096_*.csv --chunk-size 4096
    uv run python lab1/run_lab.py lab1/inputs/exp2_4d_chunk_2048_*.csv --chunk-size 2048
    uv run python lab1/run_lab.py lab1/inputs/exp2_4d_chunk_1024_*.csv --chunk-size 1024
    uv run python lab1/run_lab.py lab1/inputs/exp2_4d_chunk_0512_*.csv --chunk-size 512
    uv run python lab1/run_lab.py lab1/inputs/exp2_4d_chunk_0256_*.csv --chunk-size 256
"""

from __future__ import annotations

import argparse
from pathlib import Path

INPUTS_DIR = Path(__file__).resolve().parent.parent / "inputs"

# Warm-up 1: one request, output fixed, sweep prompt up to 4096.
WARMUP1_OUTPUT_LEN = 128
WARMUP1_PROMPTS = (16, 64, 128, 256, 512, 1024, 2048, 3072, 4096)

# Warm-up 2: one request, prompt fixed, sweep output.
WARMUP2_PROMPT_LEN = 2048
WARMUP2_OUTPUTS = (16, 32, 64, 128, 256, 512, 1024, 2048)

# Experiment 1: N copies of the same short request, all at t=0.
EXP1_PROMPT_LEN = 16
EXP1_OUTPUT_LEN = 128
EXP1_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 192, 256)

# Experiment 2: same first request in every stage.
# 1024-token prompt is "moderate". 512-token output is long enough that a
# second request at t=1.0s lands in the middle of decode (example.csv's
# 128-token decode already lasted ~4s).
EXP2_FIRST_PROMPT = 1024
EXP2_FIRST_OUTPUT = 512
EXP2_SECOND_ARRIVAL_S = 1.0
EXP2_SECOND_OUTPUT = 64
EXP2_4B_SECOND_PROMPT = 1024
EXP2_4C_SECOND_PROMPTS = (256, 512, 1024, 2048, 3072, 4000)
EXP2_4D_CHUNK_SIZES = (4096, 2048, 1024, 512, 256)


def write_csv(name: str, comments: list[str], rows: list[tuple]) -> Path:
    path = INPUTS_DIR / name
    lines = [f"# {c}" for c in comments]
    lines.append("arrival_time,prompt_len,output_len,count")
    for arrival, prompt, output, count in rows:
        lines.append(f"{arrival},{prompt},{output},{count}")
    path.write_text("\n".join(lines) + "\n")
    return path


def generate_warmup1() -> list[Path]:
    written = []
    for prompt in WARMUP1_PROMPTS:
        written.append(
            write_csv(
                f"warmup1_prompt_{prompt:04d}.csv",
                [
                    "Warm-up 1: prompt length vs latency (single request).",
                    f"prompt_len={prompt}, output_len={WARMUP1_OUTPUT_LEN}.",
                    "Run with default --chunk-size 4096.",
                    "Questions: TTFT vs prompt, TPOT vs prompt, e2e from TTFT+TPOT, prefill vs decode share.",
                ],
                [(0.0, prompt, WARMUP1_OUTPUT_LEN, 1)],
            )
        )
    return written


def generate_warmup2() -> list[Path]:
    written = []
    for output in WARMUP2_OUTPUTS:
        written.append(
            write_csv(
                f"warmup2_output_{output:04d}.csv",
                [
                    "Warm-up 2: output length vs latency (single request).",
                    f"prompt_len={WARMUP2_PROMPT_LEN}, output_len={output}.",
                    "Run with default --chunk-size 4096.",
                    "Questions: TTFT vs output, TPOT vs output, e2e from TTFT+TPOT, prefill vs decode share.",
                ],
                [(0.0, WARMUP2_PROMPT_LEN, output, 1)],
            )
        )
    return written


def generate_exp1() -> list[Path]:
    written = []
    for n in EXP1_BATCH_SIZES:
        written.append(
            write_csv(
                f"exp1_batch_{n:03d}.csv",
                [
                    "Experiment 1: batch size vs latency.",
                    f"N={n} identical requests at t=0, prompt_len={EXP1_PROMPT_LEN}, "
                    f"output_len={EXP1_OUTPUT_LEN}.",
                    "Uses the count column. Run with default --chunk-size 4096.",
                    "Questions: TTFT vs N, TPOT vs N.",
                ],
                [(0.0, EXP1_PROMPT_LEN, EXP1_OUTPUT_LEN, n)],
            )
        )
    return written


def _exp2_pair(second_prompt: int) -> list[tuple]:
    return [
        (0.0, EXP2_FIRST_PROMPT, EXP2_FIRST_OUTPUT, 1),
        (EXP2_SECOND_ARRIVAL_S, second_prompt, EXP2_SECOND_OUTPUT, 1),
    ]


def generate_exp2() -> list[Path]:
    written = []
    written.append(
        write_csv(
            "exp2_4a_baseline.csv",
            [
                "Experiment 2 / 4a: baseline, first request alone.",
                f"prompt_len={EXP2_FIRST_PROMPT}, output_len={EXP2_FIRST_OUTPUT}, arrival=0.",
                "Keep this first request unchanged in 4b–4d.",
                "Run with default --chunk-size 4096.",
            ],
            [(0.0, EXP2_FIRST_PROMPT, EXP2_FIRST_OUTPUT, 1)],
        )
    )
    written.append(
        write_csv(
            f"exp2_4b_second_prompt_{EXP2_4B_SECOND_PROMPT:04d}.csv",
            [
                "Experiment 2 / 4b: second request arrives during the first request's decode.",
                f"Request 0: prompt={EXP2_FIRST_PROMPT}, output={EXP2_FIRST_OUTPUT}, t=0.",
                f"Request 1: prompt={EXP2_4B_SECOND_PROMPT}, output={EXP2_SECOND_OUTPUT}, "
                f"t={EXP2_SECOND_ARRIVAL_S}s.",
                "Compare request 0 TPOT and token gaps to 4a. Use tokens.csv and steps.csv.",
                "Run with default --chunk-size 4096.",
            ],
            _exp2_pair(EXP2_4B_SECOND_PROMPT),
        )
    )
    for prompt in EXP2_4C_SECOND_PROMPTS:
        written.append(
            write_csv(
                f"exp2_4c_second_prompt_{prompt:04d}.csv",
                [
                    "Experiment 2 / 4c: sweep the interrupting request's prompt length.",
                    f"Request 0 unchanged: prompt={EXP2_FIRST_PROMPT}, output={EXP2_FIRST_OUTPUT}, t=0.",
                    f"Request 1: prompt={prompt}, output={EXP2_SECOND_OUTPUT}, t={EXP2_SECOND_ARRIVAL_S}s.",
                    "Discuss request 0 worst token gap and TPOT vs second prompt length.",
                    "Run with default --chunk-size 4096.",
                ],
                _exp2_pair(prompt),
            )
        )
    for chunk in EXP2_4D_CHUNK_SIZES:
        for prompt in EXP2_4C_SECOND_PROMPTS:
            written.append(
                write_csv(
                    f"exp2_4d_chunk_{chunk:04d}_second_prompt_{prompt:04d}.csv",
                    [
                        "Experiment 2 / 4d: same requests as 4c, different prefill chunk size.",
                        f"Must run with: --chunk-size {chunk}",
                        f"Request 0: prompt={EXP2_FIRST_PROMPT}, output={EXP2_FIRST_OUTPUT}, t=0.",
                        f"Request 1: prompt={prompt}, output={EXP2_SECOND_OUTPUT}, t={EXP2_SECOND_ARRIVAL_S}s.",
                    ],
                    _exp2_pair(prompt),
                )
            )
    return written


GENERATORS = {
    "warmup1": generate_warmup1,
    "warmup2": generate_warmup2,
    "exp1": generate_exp1,
    "exp2": generate_exp2,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(GENERATORS),
        help="generate only these experiment groups (default: all)",
    )
    args = parser.parse_args()

    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    groups = args.only or list(GENERATORS)
    written: list[Path] = []
    for name in groups:
        written.extend(GENERATORS[name]())

    print(f"Wrote {len(written)} files under {INPUTS_DIR}:")
    for path in written:
        print(f"  {path.name}")

    print("\nHow to run on W135 (from Vidur-Agent/, after copying lab1/inputs/ over):")
    print("  uv run python lab1/run_lab.py lab1/inputs/warmup1_*.csv")
    print("  uv run python lab1/run_lab.py lab1/inputs/warmup2_*.csv")
    print("  uv run python lab1/run_lab.py lab1/inputs/exp1_*.csv")
    print("  uv run python lab1/run_lab.py lab1/inputs/exp2_4a_*.csv")
    print("  uv run python lab1/run_lab.py lab1/inputs/exp2_4b_*.csv")
    print("  uv run python lab1/run_lab.py lab1/inputs/exp2_4c_*.csv")
    for chunk in EXP2_4D_CHUNK_SIZES:
        print(
            f"  uv run python lab1/run_lab.py lab1/inputs/exp2_4d_chunk_{chunk:04d}_*.csv "
            f"--chunk-size {chunk}"
        )


if __name__ == "__main__":
    main()
