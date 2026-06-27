#!/usr/bin/env python
"""Select winners from a scored fixed-pool manifest."""

from __future__ import annotations

import argparse

from dspa.io import write_jsonl
from dspa.manifest import load_manifest
from dspa.selectors import SelectorConfig, rank_candidates, select_candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selector", default="anchored")
    parser.add_argument("--score-key", default="native")
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--include-ranking", action="store_true")
    args = parser.parse_args()

    selector = SelectorConfig(name=args.selector, epsilon=args.epsilon)
    rows = []
    for item in load_manifest(args.manifest):
        winner, score = select_candidate(item.candidates, selector, score_key=args.score_key)
        row = {
            "item_id": item.item_id,
            "selector": args.selector,
            "score_key": args.score_key,
            "winner_index": winner.candidate_index,
            "winner_text": winner.text,
            "winner_score": score,
        }
        if args.include_ranking:
            row["ranking"] = [
                {
                    "candidate_index": candidate.candidate_index,
                    "score": candidate_score,
                    "text": candidate.text,
                }
                for candidate, candidate_score in rank_candidates(item.candidates, selector, score_key=args.score_key)
            ]
        rows.append(row)
    write_jsonl(args.output, rows)


if __name__ == "__main__":
    main()
