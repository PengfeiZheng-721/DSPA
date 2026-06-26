import argparse
import spacy
from file_io import read_jsonlines, write_jsonlines

nlp = spacy.load("en_core_web_trf")

def get_sub_clauses(text: str):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    args = parser.parse_args()

    length = None
    if args.end != -1:
        length = args.end - args.start

    if length is not None:
        data = read_jsonlines(args.input, args.start, length)
    else:
        data = read_jsonlines(args.input, args.start)
    
    # This is the new, corrected processing loop
    processed_data = []
    for item in data:
        # Case 1: The item is a list of dictionaries (the problematic case)
        if isinstance(item, list):
            for sub_item in item:
                if isinstance(sub_item, dict):
                    sub_item["sub_sents"] = get_sub_clauses(sub_item.get("answer", ""))
            processed_data.extend(item)  # Add all processed sub-items to our new list

        # Case 2: The item is a single dictionary (the normal case)
        elif isinstance(item, dict):
            item["sub_sents"] = get_sub_clauses(item.get("answer", ""))
            processed_data.append(item)

        # Case 3: The item is something else we can't handle
        else:
            print(f"Warning: Skipping a malformed item of type {type(item)}: {item}")

    # Finally, write the newly constructed list of processed data
    write_jsonlines(args.output, processed_data)

if __name__ == "__main__":
    main()
