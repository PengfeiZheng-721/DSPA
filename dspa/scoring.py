"""Score-time log-probability utilities.

These helpers are deliberately framework-light. Model-specific code only needs
to provide logits and realized token IDs; DSPA handles score-time temperature,
top-p renormalization, and dropout seed bookkeeping consistently.
"""

from __future__ import annotations

import math
import random
from typing import Any, Sequence

import numpy as np


def set_replay_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        return


def set_dropout_enabled(model: Any, enabled: bool) -> None:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is expected in model runs.
        raise RuntimeError("torch is required to toggle dropout modules") from exc

    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train(enabled)


def log_softmax(values: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    arr = np.asarray(values, dtype=np.float64) / float(temperature)
    arr = arr - np.max(arr)
    log_den = math.log(float(np.exp(arr).sum()))
    return arr - log_den


def top_p_logprob(
    logits: Sequence[float],
    token_id: int,
    top_p: float,
    temperature: float = 1.0,
    include_realized: bool = True,
) -> float:
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    log_probs = log_softmax(logits, temperature=temperature)
    probs = np.exp(log_probs)
    order = np.argsort(-probs)
    keep: set[int] = set()
    cumulative = 0.0
    for index in order:
        keep.add(int(index))
        cumulative += float(probs[index])
        if cumulative >= top_p:
            break
    if include_realized:
        keep.add(int(token_id))
    if token_id not in keep:
        return float("-inf")
    kept_mass = float(sum(probs[index] for index in keep))
    return float(log_probs[token_id] - math.log(kept_mass))


def token_logprobs(
    logits: Sequence[Sequence[float]],
    token_ids: Sequence[int],
    temperature: float = 1.0,
    top_p: float | None = None,
    include_realized: bool = True,
) -> list[float]:
    if len(logits) != len(token_ids):
        raise ValueError(f"logits/token length mismatch ({len(logits)} vs {len(token_ids)})")
    values: list[float] = []
    for token_logits, token_id in zip(logits, token_ids):
        if top_p is None:
            values.append(float(log_softmax(token_logits, temperature=temperature)[int(token_id)]))
        else:
            values.append(
                top_p_logprob(
                    token_logits,
                    int(token_id),
                    top_p=top_p,
                    temperature=temperature,
                    include_realized=include_realized,
                )
            )
    return values


def replay_score_keys() -> dict[str, list[dict[str, Any]]]:
    return {
        "dropout": [
            {"score_key": "dropout/seed0", "temperature": 0.70, "top_p": 0.90, "dropout": True, "seed": 0},
            {"score_key": "dropout/seed1", "temperature": 0.70, "top_p": 0.90, "dropout": True, "seed": 1},
        ],
        "delta_temperature": [
            {"score_key": "delta_temperature/tau0.65", "temperature": 0.65, "top_p": 0.90, "dropout": False},
            {"score_key": "delta_temperature/tau0.75", "temperature": 0.75, "top_p": 0.90, "dropout": False},
        ],
        "delta_top_p": [
            {"score_key": "delta_top_p/p0.85", "temperature": 0.70, "top_p": 0.85, "dropout": False},
            {"score_key": "delta_top_p/p0.95", "temperature": 0.70, "top_p": 0.95, "dropout": False},
        ],
    }
