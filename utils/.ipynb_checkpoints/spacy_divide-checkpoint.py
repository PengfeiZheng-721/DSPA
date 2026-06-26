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
    for item in data:
        item["sub_sents"] = get_sub_clauses(item.get("answer", ""))
    write_jsonlines(args.output, data)

if __name__ == "__main__":
    main()
