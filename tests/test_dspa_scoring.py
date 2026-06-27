import math

from dspa.scoring import token_logprobs, top_p_logprob


def test_top_p_keeps_realized_token_even_outside_nucleus():
    logits = [10.0, 0.0, -10.0]
    value = top_p_logprob(logits, token_id=2, top_p=0.5, include_realized=True)
    assert math.isfinite(value)

    dropped = top_p_logprob(logits, token_id=2, top_p=0.5, include_realized=False)
    assert dropped == float("-inf")


def test_token_logprobs_temperature_shape():
    values = token_logprobs([[1.0, 2.0], [2.0, 1.0]], [1, 0], temperature=0.7)
    assert len(values) == 2
    assert all(math.isfinite(value) for value in values)
