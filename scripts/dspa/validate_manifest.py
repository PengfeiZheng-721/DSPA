#!/usr/bin/env python
"""Validate a DSPA fixed-pool manifest."""

from __future__ import annotations

import argparse
import json

from dspa.manifest import load_manifest, manifest_digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--require-hashes", action="store_true")
    args = parser.parse_args()

    items = load_manifest(args.manifest, require_hashes=args.require_hashes)
    summary = {
        "manifest": args.manifest,
        "items": len(items),
        "candidates": sum(item.pool_size for item in items),
        "digest": manifest_digest(items),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
