"""Selector implementations used by DSPA and SRP."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

from dspa.manifest import Candidate


ScoreFn = Callable[[Candidate, str], float]


@dataclass(frozen=True)
class SelectorConfig:
    name: str = "anchored"
    epsilon: float = 0.02
    native_score_key: str = "native"


def _as_float_list(value: Any, field_name: str) -> list[float]:
    if value is None:
        raise KeyError(field_name)
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(item) for item in value]


def _score_block(candidate: Candidate, score_key: str) -> Mapping[str, Any]:
    scores = candidate.scores
    if score_key in scores and isinstance(scores[score_key], Mapping):
        return scores[score_key]
    return scores


def _get_logps(candidate: Candidate, score_key: str, field: str) -> list[float]:
    block = _score_block(candidate, score_key)
    aliases = {
        "policy_logps": ("policy_logps", "policy", "logps", "per_token_logps"),
        "reference_logps": ("reference_logps", "ref_logps", "reference", "ref"),
        "self_logps": ("self_logps", "pmi_denominator_logps", "context_free_logps"),
        "moving_reference_logps": ("moving_reference_logps", "moving_ref_logps", "comoving_ref_logps"),
    }
    for name in aliases.get(field, (field,)):
        if name in block:
            return _as_float_list(block[name], name)
    raise KeyError(f"candidate {candidate.candidate_index} is missing {field} for score key {score_key}")


def _mean(values: Sequence[float]) -> float:
    if not values:
        return -math.inf
    return float(sum(values) / len(values))


def _sum(values: Sequence[float]) -> float:
    if not values:
        return -math.inf
    return float(sum(values))


def _diff(policy: Sequence[float], reference: Sequence[float], candidate: Candidate) -> list[float]:
    if len(policy) != len(reference):
        raise ValueError(
            f"candidate {candidate.candidate_index}: policy/reference logp length mismatch "
            f"({len(policy)} vs {len(reference)})"
        )
    return [p - r for p, r in zip(policy, reference)]


def candidate_length(candidate: Candidate, score_key: str = "native") -> int:
    if candidate.token_ids is not None:
        return len(candidate.token_ids)
    try:
        return len(_get_logps(candidate, score_key, "policy_logps"))
    except KeyError:
        return max(1, len(candidate.text.split()))


def length_norm_score(candidate: Candidate, score_key: str = "native") -> float:
    return _mean(_get_logps(candidate, score_key, "policy_logps"))


def anchored_score(candidate: Candidate, score_key: str = "native") -> float:
    policy = _get_logps(candidate, score_key, "policy_logps")
    reference = _get_logps(candidate, score_key, "reference_logps")
    return _mean(_diff(policy, reference, candidate))


def sequence_log_ratio_score(candidate: Candidate, score_key: str = "native") -> float:
    policy = _get_logps(candidate, score_key, "policy_logps")
    reference = _get_logps(candidate, score_key, "reference_logps")
    return _sum(_diff(policy, reference, candidate))


def pmi_score(candidate: Candidate, score_key: str = "native") -> float:
    policy = _get_logps(candidate, score_key, "policy_logps")
    self_logps = _get_logps(candidate, score_key, "self_logps")
    return _mean(_diff(policy, self_logps, candidate))


def co_moving_reference_score(candidate: Candidate, score_key: str = "native") -> float:
    policy = _get_logps(candidate, score_key, "policy_logps")
    moving_ref = _get_logps(candidate, score_key, "moving_reference_logps")
    return _mean(_diff(policy, moving_ref, candidate))


def _score_fn(name: str) -> ScoreFn:
    normalized = name.lower().replace("_", "-")
    mapping: dict[str, ScoreFn] = {
        "length-norm": length_norm_score,
        "bon-length-norm": length_norm_score,
        "anchored": anchored_score,
        "anchored-selector": anchored_score,
        "sequence-log-ratio": sequence_log_ratio_score,
        "seq-log-ratio": sequence_log_ratio_score,
        "pmi": pmi_score,
        "self-anchored": pmi_score,
        "co-moving-reference": co_moving_reference_score,
        "comoving-reference": co_moving_reference_score,
    }
    if normalized == "epsilon-margin":
        return length_norm_score
    if normalized not in mapping:
        raise ValueError(f"unknown selector: {name}")
    return mapping[normalized]


def rank_candidates(
    candidates: Iterable[Candidate],
    selector: SelectorConfig | str,
    score_key: str | None = None,
) -> list[tuple[Candidate, float]]:
    if isinstance(selector, str):
        selector = SelectorConfig(name=selector)
    effective_key = score_key or selector.native_score_key
    score_fn = _score_fn(selector.name)
    ranked = [(candidate, score_fn(candidate, effective_key)) for candidate in candidates]
    ranked.sort(key=lambda pair: (-pair[1], pair[0].candidate_index))
    return ranked


def select_candidate(
    candidates: Iterable[Candidate],
    selector: SelectorConfig | str = "anchored",
    score_key: str | None = None,
) -> tuple[Candidate, float]:
    if isinstance(selector, str):
        selector = SelectorConfig(name=selector)
    effective_key = score_key or selector.native_score_key
    normalized = selector.name.lower().replace("_", "-")
    ranked = rank_candidates(candidates, selector, effective_key)
    if not ranked:
        raise ValueError("candidate pool is empty")
    if normalized != "epsilon-margin" or len(ranked) < 2:
        return ranked[0]

    first, second = ranked[0], ranked[1]
    if abs(first[1] - second[1]) >= selector.epsilon:
        return first

    shorter = min(
        (first, second),
        key=lambda pair: (candidate_length(pair[0], effective_key), pair[0].candidate_index),
    )
    return shorter


def attach_score(
    candidate: Candidate,
    score_key: str,
    policy_logps: Sequence[float],
    reference_logps: Sequence[float] | None = None,
    **extra: Any,
) -> Candidate:
    scores = dict(candidate.scores)
    block = dict(scores.get(score_key, {}))
    block["policy_logps"] = list(map(float, policy_logps))
    if reference_logps is not None:
        block["reference_logps"] = list(map(float, reference_logps))
    block.update(extra)
    scores[score_key] = block
    return Candidate(
        candidate_index=candidate.candidate_index,
        text=candidate.text,
        token_ids=candidate.token_ids,
        scores=scores,
        candidate_hash=candidate.candidate_hash,
    )
