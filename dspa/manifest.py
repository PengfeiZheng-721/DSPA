"""Fixed-pool manifest helpers.

The manifest is the contract between generation, scoring, and SRP replay. Each
JSONL row describes one image/question item and an ordered candidate pool. The
candidate order is semantically important because it defines the deterministic
tie-break used throughout DSPA: smaller candidate_index wins ties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256_text(payload)


@dataclass(frozen=True)
class Candidate:
    candidate_index: int
    text: str
    token_ids: list[int] | None = None
    scores: dict[str, Any] = field(default_factory=dict)
    candidate_hash: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Candidate":
        index = data.get("candidate_index", data.get("index", data.get("cid")))
        if index is None:
            raise ValueError("candidate is missing candidate_index")
        text = data.get("text", data.get("answer", data.get("candidate")))
        if text is None:
            raise ValueError(f"candidate {index} is missing text")
        token_ids = data.get("token_ids")
        scores = dict(data.get("scores", {}))
        candidate_hash = data.get("candidate_hash") or data.get("hash")
        return cls(
            candidate_index=int(index),
            text=str(text),
            token_ids=list(token_ids) if token_ids is not None else None,
            scores=scores,
            candidate_hash=str(candidate_hash) if candidate_hash else None,
        )

    def expected_hash(self) -> str:
        payload = {
            "candidate_index": self.candidate_index,
            "text": self.text,
            "token_ids": self.token_ids,
        }
        return sha256_json(payload)

    def to_mapping(self, include_hash: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "candidate_index": self.candidate_index,
            "text": self.text,
        }
        if self.token_ids is not None:
            data["token_ids"] = self.token_ids
        if self.scores:
            data["scores"] = self.scores
        if include_hash:
            data["candidate_hash"] = self.candidate_hash or self.expected_hash()
        return data


@dataclass(frozen=True)
class PoolItem:
    item_id: str
    prompt: str
    candidates: list[Candidate]
    dataset: str | None = None
    split: str | None = None
    image_id: str | None = None
    generation_model: str | None = None
    generation_revision: str | None = None
    generation_seed: int | None = None
    generation_config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    pool_hash: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PoolItem":
        item_id = data.get("item_id", data.get("id", data.get("question_id")))
        if item_id is None:
            raise ValueError("pool item is missing item_id")
        prompt = data.get("prompt", data.get("question"))
        if prompt is None:
            raise ValueError(f"pool item {item_id} is missing prompt")
        raw_candidates = data.get("candidates")
        if not raw_candidates:
            raise ValueError(f"pool item {item_id} has no candidates")
        candidates = [Candidate.from_mapping(candidate) for candidate in raw_candidates]
        candidates = sorted(candidates, key=lambda candidate: candidate.candidate_index)
        return cls(
            item_id=str(item_id),
            prompt=str(prompt),
            candidates=candidates,
            dataset=data.get("dataset"),
            split=data.get("split"),
            image_id=data.get("image_id"),
            generation_model=data.get("generation_model"),
            generation_revision=data.get("generation_revision"),
            generation_seed=data.get("generation_seed"),
            generation_config=dict(data.get("generation_config", {})),
            metadata=dict(data.get("metadata", {})),
            pool_hash=data.get("pool_hash"),
        )

    @property
    def pool_size(self) -> int:
        return len(self.candidates)

    def expected_hash(self) -> str:
        payload = {
            "item_id": self.item_id,
            "prompt": self.prompt,
            "candidates": [candidate.to_mapping(include_hash=True) for candidate in self.candidates],
        }
        return sha256_json(payload)

    def validate(self, require_hashes: bool = False) -> None:
        indices = [candidate.candidate_index for candidate in self.candidates]
        if len(indices) != len(set(indices)):
            raise ValueError(f"{self.item_id}: duplicate candidate_index values")
        if indices != sorted(indices):
            raise ValueError(f"{self.item_id}: candidates are not in deterministic order")
        if require_hashes:
            for candidate in self.candidates:
                if not candidate.candidate_hash:
                    raise ValueError(f"{self.item_id}/{candidate.candidate_index}: missing candidate_hash")
                if candidate.candidate_hash != candidate.expected_hash():
                    raise ValueError(f"{self.item_id}/{candidate.candidate_index}: candidate_hash mismatch")
            if not self.pool_hash:
                raise ValueError(f"{self.item_id}: missing pool_hash")
            if self.pool_hash != self.expected_hash():
                raise ValueError(f"{self.item_id}: pool_hash mismatch")

    def to_mapping(self, include_hashes: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "item_id": self.item_id,
            "dataset": self.dataset,
            "split": self.split,
            "image_id": self.image_id,
            "prompt": self.prompt,
            "pool_size": self.pool_size,
            "generation_model": self.generation_model,
            "generation_revision": self.generation_revision,
            "generation_seed": self.generation_seed,
            "generation_config": self.generation_config,
            "candidates": [candidate.to_mapping(include_hash=include_hashes) for candidate in self.candidates],
            "metadata": self.metadata,
        }
        data = {key: value for key, value in data.items() if value not in (None, {}, [])}
        if include_hashes:
            data["pool_hash"] = self.pool_hash or self.expected_hash()
        return data


def load_manifest(path: str | Path, require_hashes: bool = False) -> list[PoolItem]:
    items: list[PoolItem] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = PoolItem.from_mapping(json.loads(line))
                item.validate(require_hashes=require_hashes)
            except Exception as exc:  # noqa: BLE001 - include line number context.
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            items.append(item)
    return items


def write_manifest(items: Iterable[PoolItem], path: str | Path, include_hashes: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.to_mapping(include_hashes=include_hashes), ensure_ascii=False))
            handle.write("\n")


def manifest_digest(items: Iterable[PoolItem]) -> str:
    return sha256_json([item.to_mapping(include_hashes=True) for item in items])
