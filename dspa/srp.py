"""Same-pool/Same-budget Replay (SRP)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from dspa.manifest import PoolItem
from dspa.selectors import SelectorConfig, select_candidate
from dspa.stats import wilson_interval


@dataclass(frozen=True)
class ReplayFamily:
    name: str
    score_keys: list[str]


def _percent(value: float) -> float:
    return 100.0 * value


def _ci_percent(positives: int, total: int) -> tuple[float, float]:
    low, high = wilson_interval(positives, total)
    return (_percent(low), _percent(high))


def run_srp(
    items: Iterable[PoolItem],
    selector: SelectorConfig | str = "anchored",
    families: Iterable[ReplayFamily] | None = None,
    native_score_key: str = "native",
) -> dict[str, Any]:
    if isinstance(selector, str):
        selector = SelectorConfig(name=selector, native_score_key=native_score_key)
    else:
        selector = SelectorConfig(
            name=selector.name,
            epsilon=selector.epsilon,
            native_score_key=native_score_key,
        )

    items = list(items)
    if families is None:
        discovered: dict[str, list[str]] = {}
        for item in items:
            for candidate in item.candidates:
                for key in candidate.scores:
                    if key == native_score_key:
                        continue
                    family = key.split("/", 1)[0]
                    discovered.setdefault(family, [])
                    if key not in discovered[family]:
                        discovered[family].append(key)
        families = [ReplayFamily(name=name, score_keys=keys) for name, keys in sorted(discovered.items())]
    else:
        families = list(families)

    native_winners: dict[str, int] = {}
    native_scores: dict[str, float] = {}
    for item in items:
        winner, score = select_candidate(item.candidates, selector, score_key=native_score_key)
        native_winners[item.item_id] = winner.candidate_index
        native_scores[item.item_id] = score

    family_results = []
    for family in families:
        flipped_items = 0
        agreement_count = 0
        total_replays = 0
        item_rows = []
        for item in items:
            native_idx = native_winners[item.item_id]
            replay_winners: list[int] = []
            for score_key in family.score_keys:
                winner, score = select_candidate(item.candidates, selector, score_key=score_key)
                replay_winners.append(winner.candidate_index)
                agreement_count += int(winner.candidate_index == native_idx)
                total_replays += 1
                item_rows.append(
                    {
                        "item_id": item.item_id,
                        "family": family.name,
                        "score_key": score_key,
                        "native_winner": native_idx,
                        "replay_winner": winner.candidate_index,
                        "replay_score": score,
                    }
                )
            flipped_items += int(any(winner != native_idx for winner in replay_winners))

        item_count = len(items)
        flip_rate = flipped_items / item_count if item_count else 0.0
        agreement = agreement_count / total_replays if total_replays else 0.0
        family_results.append(
            {
                "family": family.name,
                "score_keys": family.score_keys,
                "items": item_count,
                "replays": total_replays,
                "flips": flipped_items,
                "agreements": agreement_count,
                "flip_rate": flip_rate,
                "flip_rate_percent": _percent(flip_rate),
                "flip_rate_ci95_percent": _ci_percent(flipped_items, item_count) if item_count else (0.0, 0.0),
                "agreement": agreement,
                "agreement_percent": _percent(agreement),
                "agreement_ci95_percent": _ci_percent(agreement_count, total_replays)
                if total_replays
                else (0.0, 0.0),
                "item_results": item_rows,
            }
        )

    if family_results:
        flip_avg = sum(row["flip_rate"] for row in family_results) / len(family_results)
        agreement_avg = sum(row["agreement"] for row in family_results) / len(family_results)
    else:
        flip_avg = 0.0
        agreement_avg = 0.0

    return {
        "selector": selector.name,
        "native_score_key": native_score_key,
        "items": len(items),
        "native_winners": native_winners,
        "native_scores": native_scores,
        "families": family_results,
        "flip_rate_avg": flip_avg,
        "flip_rate_avg_percent": _percent(flip_avg),
        "agreement_avg": agreement_avg,
        "agreement_avg_percent": _percent(agreement_avg),
    }
