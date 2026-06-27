"""Reporting helpers for DSPA experiment outputs."""

from __future__ import annotations

from typing import Any, Iterable


def srp_rows(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        selector = result.get("selector", "unknown")
        for family in result.get("families", []):
            rows.append(
                {
                    "selector": selector,
                    "family": family["family"],
                    "flip_rate_percent": round(float(family["flip_rate_percent"]), 4),
                    "agreement_percent": round(float(family["agreement_percent"]), 4),
                    "items": family["items"],
                    "replays": family["replays"],
                    "flips": family["flips"],
                    "agreements": family["agreements"],
                }
            )
        rows.append(
            {
                "selector": selector,
                "family": "average",
                "flip_rate_percent": round(float(result.get("flip_rate_avg_percent", 0.0)), 4),
                "agreement_percent": round(float(result.get("agreement_avg_percent", 0.0)), 4),
                "items": result.get("items", 0),
                "replays": "",
                "flips": "",
                "agreements": "",
            }
        )
    return rows


def rows_to_markdown(rows: list[dict[str, Any]]) -> str:
    headers = [
        "selector",
        "family",
        "flip_rate_percent",
        "agreement_percent",
        "items",
        "replays",
        "flips",
        "agreements",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines) + "\n"
