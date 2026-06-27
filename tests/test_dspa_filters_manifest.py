from dspa.filters import (
    FilterConfig,
    extract_object_query,
    filter_pair,
    longest_contiguous_overlap,
)
from dspa.manifest import Candidate, PoolItem, load_manifest, write_manifest


def test_longest_contiguous_overlap():
    assert longest_contiguous_overlap("a b c d e", "x b c d y") == 3


def test_filters_reject_length_ratio_and_object_hallucination():
    length_pair = {
        "question": "Describe the image.",
        "chosen": "one two three four five six",
        "rejected": "one two",
    }
    ok, reason = filter_pair(length_pair, FilterConfig())
    assert not ok
    assert reason == "length_ratio"

    object_pair = {
        "question": "Is there a dog in the image?",
        "chosen": "Yes, there is a dog.",
        "rejected": "No, there is no dog.",
        "object_labels": {"dog": False},
    }
    ok, reason = filter_pair(object_pair, FilterConfig(enable_length_filter=False))
    assert not ok
    assert reason == "object_hallucination"


def test_extract_object_query():
    assert extract_object_query("Is there a red car in the image?") == "red car"


def test_manifest_round_trip_with_hashes(tmp_path):
    item = PoolItem(
        item_id="item-1",
        prompt="What is shown?",
        candidates=[Candidate(candidate_index=0, text="A cat.", token_ids=[1, 2, 3])],
    )
    path = tmp_path / "manifest.jsonl"
    write_manifest([item], path)

    loaded = load_manifest(path, require_hashes=True)
    assert loaded[0].item_id == "item-1"
    assert loaded[0].candidates[0].candidate_index == 0
