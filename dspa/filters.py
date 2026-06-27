"""Training-time preference filters from DSPA."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping


_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_YES_RE = re.compile(r"\b(yes|yeah|yep|there is|there are|i can see|visible|appears to be)\b", re.I)
_NO_RE = re.compile(r"\b(no|not visible|cannot see|can't see|there is no|there are no|does not appear)\b", re.I)
_OBJECT_PATTERNS = [
    re.compile(r"\bis there (?:a|an|the|any)?\s*([a-z0-9 _-]+?)(?:\?| in | on | at | visible| present|$)", re.I),
    re.compile(r"\bare there (?:a|an|the|any)?\s*([a-z0-9 _-]+?)(?:\?| in | on | at | visible| present|$)", re.I),
]


@dataclass(frozen=True)
class FilterConfig:
    max_prompt_overlap_tokens: int = 12
    min_length_ratio: float = 0.8
    max_length_ratio: float = 1.2
    max_pairs_per_input: int = 2
    enable_prompt_copy_filter: bool = True
    enable_length_filter: bool = True
    enable_object_filter: bool = True
    enable_caption_tail_mask: bool = True


def tokenize_words(text: str) -> list[str]:
    return [match.group(0).lower() for match in _WORD_RE.finditer(text or "")]


def longest_contiguous_overlap(a: str, b: str) -> int:
    left = tokenize_words(a)
    right = tokenize_words(b)
    if not left or not right:
        return 0
    prev = [0] * (len(right) + 1)
    best = 0
    for left_token in left:
        curr = [0] * (len(right) + 1)
        for j, right_token in enumerate(right, 1):
            if left_token == right_token:
                curr[j] = prev[j - 1] + 1
                best = max(best, curr[j])
        prev = curr
    return best


def length_ratio(chosen: str, rejected: str) -> float:
    chosen_len = max(1, len(tokenize_words(chosen)))
    rejected_len = max(1, len(tokenize_words(rejected)))
    return chosen_len / rejected_len


def extract_object_query(prompt: str) -> str | None:
    for pattern in _OBJECT_PATTERNS:
        match = pattern.search(prompt or "")
        if match:
            obj = " ".join(tokenize_words(match.group(1)))
            return obj or None
    return None


def answer_affirms(text: str) -> bool:
    return bool(_YES_RE.search(text or "")) and not answer_denies(text)


def answer_denies(text: str) -> bool:
    return bool(_NO_RE.search(text or ""))


def _label_says_present(labels: Mapping[str, Any], obj: str) -> bool | None:
    if not labels:
        return None
    normalized = {str(key).lower(): value for key, value in labels.items()}
    if obj.lower() in normalized:
        return bool(normalized[obj.lower()])
    singular = obj.lower().rstrip("s")
    if singular in normalized:
        return bool(normalized[singular])
    return None


def candidate_passes_prompt_copy(prompt: str, answer: str, config: FilterConfig) -> bool:
    if not config.enable_prompt_copy_filter:
        return True
    return longest_contiguous_overlap(prompt, answer) <= config.max_prompt_overlap_tokens


def pair_passes_length(chosen: str, rejected: str, config: FilterConfig) -> bool:
    if not config.enable_length_filter:
        return True
    ratio = length_ratio(chosen, rejected)
    return config.min_length_ratio <= ratio <= config.max_length_ratio


def pair_passes_object_filter(
    prompt: str,
    chosen: str,
    labels: Mapping[str, Any] | None,
    config: FilterConfig,
) -> bool:
    if not config.enable_object_filter:
        return True
    obj = extract_object_query(prompt)
    if not obj:
        return True
    present = _label_says_present(labels or {}, obj)
    if present is None:
        return True
    return not (present is False and answer_affirms(chosen))


def mask_caption_tail(prompt: str, answer: str, config: FilterConfig) -> str:
    if not config.enable_caption_tail_mask:
        return answer
    prompt_tokens = tokenize_words(prompt)
    answer_tokens = answer.split()
    if len(answer_tokens) <= config.max_prompt_overlap_tokens:
        return answer
    prompt_windows = {
        tuple(prompt_tokens[i : i + config.max_prompt_overlap_tokens])
        for i in range(0, max(0, len(prompt_tokens) - config.max_prompt_overlap_tokens + 1))
    }
    lower_answer = [token.lower().strip(".,;:!?\"'()[]{}") for token in answer_tokens]
    for i in range(0, len(lower_answer) - config.max_prompt_overlap_tokens + 1):
        window = tuple(lower_answer[i : i + config.max_prompt_overlap_tokens])
        if window in prompt_windows:
            return " ".join(answer_tokens[:i]).strip()
    return answer


def filter_pair(pair: Mapping[str, Any], config: FilterConfig | None = None) -> tuple[bool, str]:
    config = config or FilterConfig()
    prompt = str(pair.get("prompt", pair.get("question", "")))
    chosen = str(pair.get("chosen", ""))
    rejected = str(pair.get("rejected", ""))
    labels = pair.get("object_labels", pair.get("labels", {}))

    if not candidate_passes_prompt_copy(prompt, chosen, config):
        return False, "chosen_prompt_copy"
    if not candidate_passes_prompt_copy(prompt, rejected, config):
        return False, "rejected_prompt_copy"
    if not pair_passes_length(chosen, rejected, config):
        return False, "length_ratio"
    if not pair_passes_object_filter(prompt, chosen, labels, config):
        return False, "object_hallucination"
    return True, "ok"


def filter_pairs(
    pairs: Iterable[Mapping[str, Any]],
    config: FilterConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    config = config or FilterConfig()
    kept: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    per_input: dict[str, int] = {}
    for pair in pairs:
        ok, reason = filter_pair(pair, config)
        counts[reason] = counts.get(reason, 0) + 1
        if not ok:
            continue
        input_id = str(pair.get("item_id", pair.get("ds_question_id", pair.get("question_id", len(per_input)))))
        if per_input.get(input_id, 0) >= config.max_pairs_per_input:
            counts["max_pairs_per_input"] = counts.get("max_pairs_per_input", 0) + 1
            continue
        per_input[input_id] = per_input.get(input_id, 0) + 1
        new_pair = dict(pair)
        pair_prompt = str(new_pair.get("prompt", new_pair.get("question", "")))
        new_pair["chosen"] = mask_caption_tail(pair_prompt, str(new_pair["chosen"]), config)
        kept.append(new_pair)
    return kept, counts
