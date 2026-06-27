#!/usr/bin/env python
"""Run Same-pool/Same-budget Replay from a scored fixed-pool manifest."""

from __future__ import annotations

import argparse
import json

from dspa.io import write_json
from dspa.manifest import load_manifest
from dspa.selectors import SelectorConfig
from dspa.srp import ReplayFamily, run_srp


def _parse_family(value: str) -> ReplayFamily:
    if "=" not in value:
        raise argparse.ArgumentTypeError("family must have form name=score_key,score_key")
    name, raw_keys = value.split("=", 1)
    keys = [key.strip() for key in raw_keys.split(",") if key.strip()]
    if not name or not keys:
        raise argparse.ArgumentTypeError("family must have a name and at least one score key")
    return ReplayFamily(name=name, score_keys=keys)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--selector", default="anchored")
    parser.add_argument("--native-score-key", default="native")
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument(
        "--family",
        action="append",
        type=_parse_family,
        default=None,
        help="Replay family, e.g. dropout=dropout/seed0,dropout/seed1. "
        "If omitted, families are inferred from score-key prefixes.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    items = load_manifest(args.manifest)
    result = run_srp(
        items,
        selector=SelectorConfig(name=args.selector, epsilon=args.epsilon),
        families=args.family,
        native_score_key=args.native_score_key,
    )
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
