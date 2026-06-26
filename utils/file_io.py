import json
import jsonlines
from typing import Optional

def write_jsonlines(file: str, dataset: list):
    with jsonlines.open(file, 'w') as writer:
        for data in dataset:
            writer.write(data)

def write_json(save_path, data):
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)

    return data

def read_jsonlines(file: str, index_begin: int = 0, index_all: Optional[int] = -1):
    """Read a jsonlines file with optional start and end indices.

    Parameters
    ----------
    file : str
        Path to the jsonl file.
    index_begin : int, optional
        The starting index of the records to read. Defaults to 0.
    index_all : int, optional
        The number of records to read. ``-1`` means reading to the end.
    """

    if index_all is None:
        index_all = -1

    dataset = []
    with jsonlines.open(file, 'r') as reader:
        for data in reader:
            if index_begin > 0:
                index_begin -= 1
                continue

            dataset.append(data)

            if index_all > 0:
                index_all -= 1
                if index_all == 0:
                    break

    return dataset
