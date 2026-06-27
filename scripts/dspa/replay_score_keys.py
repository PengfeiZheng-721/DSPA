#!/usr/bin/env python
"""Print the canonical SRP perturbation score-key plan."""

from __future__ import annotations

import json

from dspa.scoring import replay_score_keys


if __name__ == "__main__":
    print(json.dumps(replay_score_keys(), indent=2))
