"""
link_cache.py -- make the shipped Qwen-32B prediction cache loadable here.

Run once per checkout, after extracting
cache/cache_lite_qwen32b_a100_tp2_tp4.tar.gz into cache/:

    python lab/link_cache.py            # link
    python lab/link_cache.py --dry-run  # show the mapping only

Vidur names a cached prediction table `<op>_<key>_predictions.pkl`, where the key
hashes the predictor's settings *including the absolute paths of the checkout that
built it*. The tarball was built elsewhere, so no name matches here. This script
hard-links every shipped table under the name this checkout computes, leaving the
originals in place. Nothing in vidur/ is modified; re-run it after moving the repo.

The tarball does not record which files are TP2 and which TP4. It holds two fitting
sessions; in the TP4 one every op whose work is split across GPUs is clearly faster.
The script checks that all of them agree before linking anything.
"""

import argparse
import os
import pickle
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import qwen32b_support  # noqa: E402  (registers the Qwen2.5-32B model config)
from vidur.config import (  # noqa: E402
    CacheConfig,
    RandomForestExecutionTimePredictorConfig,
    ReplicaConfig,
)
from vidur.execution_time_predictor.random_forest_execution_time_predictor import (  # noqa: E402
    RandomForestExecutionTimePredictor,
)

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = "cache"
TARBALL = f"{CACHE_DIR}/cache_lite_qwen32b_a100_tp2_tp4.tar.gz"
MODEL_NAME = qwen32b_support.Qwen25_32BModelConfig.get_name()
DEVICE, NETWORK_DEVICE = "a100", "a100_dgx"
SUFFIX = "_predictions.pkl"

# Ops split across tensor-parallel workers, with a probe input: the TP4 prediction
# must be clearly faster than the TP2 one.
TP_SENSITIVE_PROBES = {
    "attn_pre_proj": (4096,),
    "attn_post_proj": (4096,),
    "mlp_up_proj": (4096,),
    "mlp_down_proj": (4096,),
    "attn_prefill": (0, 4096),  # (kv_cache_size, prefill_chunk_size)
}
MIN_TP_SPEEDUP = 1.3


class _KeyOnlyPredictor(RandomForestExecutionTimePredictor):
    """Computes cache keys; loads nothing."""

    def _predict_from_models(self):
        return {}


def op_name(member) -> str:
    # "<op>_<key>_predictions.pkl" -> "<op>"
    return member.name[: -len(SUFFIX)].rsplit("_", 1)[0]


def load(member):
    with open(os.path.join(CACHE_DIR, member.name), "rb") as f:
        return pickle.load(f)


def split_sessions(members):
    """Split the shipped tables into the two fitting sessions, by mtime."""
    members = sorted(members, key=lambda m: m.mtime)
    gaps = [b.mtime - a.mtime for a, b in zip(members, members[1:])]
    cut = gaps.index(max(gaps)) + 1
    sessions = [members[:cut], members[cut:]]
    for session in sessions:
        ops = [op_name(m) for m in session]
        assert len(ops) == len(set(ops)), f"an op appears twice in one session: {ops}"
    assert {op_name(m) for m in sessions[0]} == {op_name(m) for m in sessions[1]}, (
        "the two sessions cover different ops"
    )
    return [{op_name(m): m for m in session} for session in sessions]


def assign_tp(sessions):
    """Return {tp: session}, checking every TP-sensitive op agrees."""
    print("TP check (prediction at probe input, ms):")
    faster = set()
    for op, probe in TP_SENSITIVE_PROBES.items():
        a, b = (load(s[op])[probe] for s in sessions)
        speedup = max(a, b) / min(a, b)
        print(f"  {op:16s} session0 {a:8.4f}  session1 {b:8.4f}  speedup {speedup:.2f}x")
        assert speedup >= MIN_TP_SPEEDUP, f"{op}: sessions too close to tell TP apart"
        faster.add(0 if a < b else 1)
    assert len(faster) == 1, "ops disagree on which session is TP4"
    tp4 = faster.pop()
    return {4: sessions[tp4], 2: sessions[1 - tp4]}


def new_key(op: str, tp: int) -> str:
    predictor = _KeyOnlyPredictor(
        predictor_config=RandomForestExecutionTimePredictorConfig(cache_dir=CACHE_DIR),
        replica_config=ReplicaConfig(
            model_name=MODEL_NAME,
            tensor_parallel_size=tp,
            device=DEVICE,
            network_device=NETWORK_DEVICE,
        ),
        cache_config=CacheConfig(),
    )
    return predictor._get_model_hash(op, df=None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.chdir(REPO_DIR)

    if not os.path.exists(TARBALL):
        sys.exit(f"{TARBALL} not found. Put the cache tarball there first.")

    with tarfile.open(TARBALL) as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(SUFFIX)]
        for m in members:
            path = os.path.join(CACHE_DIR, m.name)
            if not os.path.exists(path):
                print(f"extracting {m.name}")
                tar.extract(m, CACHE_DIR)
            assert os.path.getsize(path) == m.size, f"{path} differs from the tarball"

    by_tp = assign_tp(split_sessions(members))

    print(f"\nLinks for {MODEL_NAME} on {DEVICE}:")
    for tp, session in sorted(by_tp.items()):
        for op, member in sorted(session.items()):
            src = os.path.join(CACHE_DIR, member.name)
            dst = os.path.join(CACHE_DIR, f"{op}_{new_key(op, tp)}{SUFFIX}")
            if os.path.exists(dst) and os.path.samefile(src, dst):
                status = "already linked"
            elif os.path.exists(dst):
                # A link left over from an earlier copy of the shipped table.
                status = "would relink" if args.dry_run else "relinked"
                if not args.dry_run:
                    os.unlink(dst)
                    os.link(src, dst)
            elif args.dry_run:
                status = "would link"
            else:
                os.link(src, dst)
                status = "linked"
            print(f"  TP{tp} {member.name:45s} -> {os.path.basename(dst):45s} {status}")


if __name__ == "__main__":
    main()
