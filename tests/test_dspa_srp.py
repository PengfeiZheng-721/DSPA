from dspa.manifest import Candidate, PoolItem
from dspa.srp import ReplayFamily, run_srp


def test_run_srp_computes_flip_rate_and_agreement():
    item_a = PoolItem(
        item_id="a",
        prompt="question a",
        candidates=[
            Candidate(
                candidate_index=0,
                text="a0",
                scores={
                    "native": {"policy_logps": [0.6], "reference_logps": [0.0]},
                    "delta_temperature/tau0.65": {"policy_logps": [0.4], "reference_logps": [0.0]},
                    "delta_temperature/tau0.75": {"policy_logps": [0.7], "reference_logps": [0.0]},
                },
            ),
            Candidate(
                candidate_index=1,
                text="a1",
                scores={
                    "native": {"policy_logps": [0.5], "reference_logps": [0.0]},
                    "delta_temperature/tau0.65": {"policy_logps": [0.8], "reference_logps": [0.0]},
                    "delta_temperature/tau0.75": {"policy_logps": [0.2], "reference_logps": [0.0]},
                },
            ),
        ],
    )
    item_b = PoolItem(
        item_id="b",
        prompt="question b",
        candidates=[
            Candidate(
                candidate_index=0,
                text="b0",
                scores={
                    "native": {"policy_logps": [0.9], "reference_logps": [0.0]},
                    "delta_temperature/tau0.65": {"policy_logps": [0.9], "reference_logps": [0.0]},
                    "delta_temperature/tau0.75": {"policy_logps": [0.9], "reference_logps": [0.0]},
                },
            ),
            Candidate(
                candidate_index=1,
                text="b1",
                scores={
                    "native": {"policy_logps": [0.1], "reference_logps": [0.0]},
                    "delta_temperature/tau0.65": {"policy_logps": [0.1], "reference_logps": [0.0]},
                    "delta_temperature/tau0.75": {"policy_logps": [0.1], "reference_logps": [0.0]},
                },
            ),
        ],
    )

    result = run_srp(
        [item_a, item_b],
        selector="anchored",
        families=[ReplayFamily("delta_temperature", ["delta_temperature/tau0.65", "delta_temperature/tau0.75"])],
    )

    family = result["families"][0]
    assert family["flips"] == 1
    assert family["replays"] == 4
    assert family["agreements"] == 3
    assert family["flip_rate"] == 0.5
    assert family["agreement"] == 0.75
