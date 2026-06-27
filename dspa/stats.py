"""Statistical helpers for SRP reporting."""

from __future__ import annotations

from math import comb


def wilson_interval(positives: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    phat = positives / total
    den = 1.0 + z * z / total
    mid = phat + z * z / (2.0 * total)
    rad = z * ((phat * (1.0 - phat) / total + z * z / (4.0 * total * total)) ** 0.5)
    return ((mid - rad) / den, (mid + rad) / den)


def exact_mcnemar_pvalue(b: int, c: int) -> float:
    if b < 0 or c < 0:
        raise ValueError("b and c must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    observed = min(b, c)
    tail = sum(comb(n, k) for k in range(0, observed + 1)) / (2**n)
    return min(1.0, 2.0 * tail)
