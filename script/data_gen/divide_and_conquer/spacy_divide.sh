#!/bin/bash
export PYTHONPATH=$PYTHONPATH:`realpath .`

INPUT_FILE=$1
START=${2:-0}
END=${3:--1}

# This line correctly removes the old .jsonl and adds the new parts
OUTPUT_FILE=${INPUT_FILE%.jsonl}.spacy_divide.jsonl

echo "Input file: ${INPUT_FILE}"
echo "Output file: ${OUTPUT_FILE}"

# Call the python script with the correct --input and --output arguments
python ./utils/spacy_divide.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --start "$START" \
    --end "$END"