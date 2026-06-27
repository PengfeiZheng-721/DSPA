from dspa.manifest import Candidate
from dspa.selectors import SelectorConfig, anchored_score, select_candidate, sequence_log_ratio_score


def candidate(index, policy, reference=None, text=None):
    if reference is None:
        reference = [0.0 for _ in policy]
    return Candidate(
        candidate_index=index,
        text=text or f"candidate {index}",
        scores={"native": {"policy_logps": policy, "reference_logps": reference}},
    )


def test_anchored_selector_uses_per_token_average():
    concise = candidate(0, [0.15, 0.15], [0.0, 0.0])
    padded = candidate(1, [0.15, 0.15, 0.02, 0.02], [0.0, 0.0, 0.0, 0.0])

    assert sequence_log_ratio_score(padded) > sequence_log_ratio_score(concise)
    assert anchored_score(concise) > anchored_score(padded)

    winner, _ = select_candidate([padded, concise], "anchored")
    assert winner.candidate_index == 0


def test_deterministic_tie_break_uses_smallest_candidate_index():
    first = candidate(0, [0.1], [0.0])
    second = candidate(1, [0.1], [0.0])

    winner, _ = select_candidate([second, first], "anchored")
    assert winner.candidate_index == 0


def test_epsilon_margin_returns_shorter_top_two_near_tie():
    longer = Candidate(
        candidate_index=0,
        text="long answer with extra words",
        scores={"native": {"policy_logps": [0.50, 0.50, 0.50, 0.50]}},
    )
    shorter = Candidate(
        candidate_index=1,
        text="short answer",
        scores={"native": {"policy_logps": [0.49, 0.49]}},
    )

    winner, _ = select_candidate(
        [longer, shorter],
        SelectorConfig(name="epsilon-margin", epsilon=0.02),
    )
    assert winner.candidate_index == 1
