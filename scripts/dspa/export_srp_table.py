#!/usr/bin/env python
"""Export one or more SRP result JSON files as CSV or Markdown."""

from __future__ import annotations

import argparse
import csv
import json
import sys

from dspa.reporting import rows_to_markdown, srp_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", help="SRP JSON files from scripts/dspa/run_srp.py")
    parser.add_argument("--format", choices=["csv", "md"], default="csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    payloads = []
    for path in args.results:
        with open(path, "r", encoding="utf-8") as handle:
            payloads.append(json.load(handle))
    rows = srp_rows(payloads)

    if args.format == "md":
        output = rows_to_markdown(rows)
    else:
        headers = list(rows[0].keys()) if rows else []
        from io import StringIO

        buffer = StringIO()
        writer = csv.DictWriter(buffer, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
        output = buffer.getvalue()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(output)
    else:
        sys.stdout.write(output)


if __name__ == "__main__":
    main()
