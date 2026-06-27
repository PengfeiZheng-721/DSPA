#!/usr/bin/env python
"""Apply DSPA training-time preference filters to a JSONL pair file."""

from __future__ import annotations

import argparse
import json

from dspa.filters import FilterConfig, filter_pairs
from dspa.io import read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--max-prompt-overlap-tokens", type=int, default=12)
    parser.add_argument("--min-length-ratio", type=float, default=0.8)
    parser.add_argument("--max-length-ratio", type=float, default=1.2)
    parser.add_argument("--max-pairs-per-input", type=int, default=2)
    parser.add_argument("--disable-prompt-copy-filter", action="store_true")
    parser.add_argument("--disable-length-filter", action="store_true")
    parser.add_argument("--disable-object-filter", action="store_true")
    parser.add_argument("--disable-caption-tail-mask", action="store_true")
    args = parser.parse_args()

    config = FilterConfig(
        max_prompt_overlap_tokens=args.max_prompt_overlap_tokens,
        min_length_ratio=args.min_length_ratio,
        max_length_ratio=args.max_length_ratio,
        max_pairs_per_input=args.max_pairs_per_input,
        enable_prompt_copy_filter=not args.disable_prompt_copy_filter,
        enable_length_filter=not args.disable_length_filter,
        enable_object_filter=not args.disable_object_filter,
        enable_caption_tail_mask=not args.disable_caption_tail_mask,
    )
    rows = read_jsonl(args.input)
    kept, counts = filter_pairs(rows, config)
    write_jsonl(args.output, kept)
    summary = {"input": len(rows), "kept": len(kept), "reasons": counts}
    if args.summary:
        write_json(args.summary, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
