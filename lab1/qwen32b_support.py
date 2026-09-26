"""
qwen32b_support.py -- run the shipped Qwen-32B prediction cache without
modifying Vidur.

The tarball cache/cache_lite_qwen32b_a100_tp2_tp4.tar.gz holds execution-time
predictions for Qwen2.5-32B on A100 (TP2 and TP4). Two things are missing from
the checkout for Vidur to use them:

1. Vidur has no Qwen2.5-32B model config. `Qwen25_32BModelConfig` below adds one.
   Vidur finds model configs by scanning subclasses of BaseModelConfig, so simply
   importing this module registers it.
2. Vidur builds its prediction tables by reading profiling CSVs and fitting
   regressors. There are no profiling CSVs for this model -- only the finished
   tables. `apply()` replaces `SklearnExecutionTimePredictor._predict_from_models`
   with a version that loads every table from the cache and fails if one is
   missing, so nothing is ever read from disk profiling data or fitted.

Import this module and call `apply()` before building a SimulationConfig:

    import qwen32b_support
    qwen32b_support.apply()

Nothing in vidur/ is edited. Cache file names are Vidur's own (they hash the
absolute paths of this checkout), so after moving or re-cloning the repo, re-run
`python lab/link_cache.py`.
"""

import os
from dataclasses import dataclass
from typing import Optional

from vidur.config.model_config import QwenModelConfig
from vidur.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)

# Compute ops predicted per step, in Vidur's own order.
COMPUTE_MODELS = [
    "attn_pre_proj",
    "attn_post_proj",
    "mlp_up_proj",
    "mlp_down_proj",
    "mlp_act",
    "attn_rope",
    "attn_kv_cache_save",
    "input_layernorm",
    "post_attention_layernorm",
    "add",
]
ATTENTION_MODELS = ["attn_prefill", "attn_decode"]


@dataclass
class Qwen25_32BModelConfig(QwenModelConfig):
    """Qwen2.5-32B: 64 layers, 40 query / 8 KV heads, hidden 5120, MLP 27648.

    Inferred from the timings in the shipped cache (see instructor/CACHE_NOTES.md); the
    config the cache was fitted with is not in this repo.
    """

    num_layers: int = 64
    num_q_heads: int = 40
    num_kv_heads: int = 8
    embedding_dim: int = 5120
    mlp_hidden_dim: int = 27648
    rope_theta: Optional[float] = 1000000

    @staticmethod
    def get_name():
        return "Qwen/Qwen2.5-32B"


def _predictions_from_cache(self):
    """Every prediction table straight from the cache; no profiling, no fitting."""
    if self._replica_config.pd_disaggregation or self._replica_config.num_pipeline_stages > 1:
        raise NotImplementedError(
            "qwen32b_support covers single-stage replicas without prefill-decode "
            "disaggregation; KV-transfer time would need the fitted send_recv model."
        )
    if not self._config.skip_cpu_overhead_modeling:
        raise NotImplementedError(
            "qwen32b_support cannot model CPU overheads: the cache has no tables for them."
        )

    model_names = list(COMPUTE_MODELS)
    if self._replica_config.tensor_parallel_size > 1:
        model_names.append("all_reduce")
    model_names += ATTENTION_MODELS

    hashes = {name: self._get_model_hash(name, df=None) for name in model_names}
    missing = [
        f"{name}_{h}_predictions.pkl"
        for name, h in hashes.items()
        if not os.path.exists(f"{self._config.cache_dir}/{name}_{h}_predictions.pkl")
    ]
    if missing:
        raise FileNotFoundError(
            f"Prediction tables missing from {self._config.cache_dir}/: "
            + ", ".join(missing[:3])
            + (f" (+{len(missing) - 3} more)" if len(missing) > 3 else "")
            + ". Extract cache_lite_qwen32b_a100_tp2_tp4.tar.gz into cache/ and run "
            "`python lab/link_cache.py`."
        )

    return {
        name: self._load_model_predication_cache(name, h) for name, h in hashes.items()
    }


def apply() -> None:
    """Make Vidur serve predictions from the cache. Call before SimulationConfig."""
    SklearnExecutionTimePredictor._predict_from_models = _predictions_from_cache
