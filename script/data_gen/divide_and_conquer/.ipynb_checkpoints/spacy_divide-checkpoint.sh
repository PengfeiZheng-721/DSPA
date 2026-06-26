#!/bin/bash
export PYTHONPATH=$PYTHONPATH:`realpath .`

input_path=$1
start=${2:-0}
end=${3:--1}

python utils/spacy_divide.py \
  --input ${input_path} \
  --output ${input_path}.spacy_divide.jsonl \
  --start ${start} --end ${end}
