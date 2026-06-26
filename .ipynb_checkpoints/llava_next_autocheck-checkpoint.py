import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.file_io import read_jsonlines, write_jsonlines


def yes_probability(logits, yes_ids, no_ids):
    probs = torch.softmax(logits, dim=-1)
    yes_prob = probs[yes_ids].sum().item()
    no_prob = probs[no_ids].sum().item()
    return yes_prob / (yes_prob + no_prob + 1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--ds_name', type=str, required=True)
    parser.add_argument('--answer_file', type=str, required=True)
    parser.add_argument('--start_pos', type=int, default=0)
    parser.add_argument('--end_pos', type=int, default=-1)
    parser.add_argument('--is_yesno', action='store_true')
    args = parser.parse_args()

    length = None
    if args.end_pos != -1:
        length = args.end_pos - args.start_pos

    if length is not None:
        data = read_jsonlines(args.ds_name, args.start_pos, length)
    else:
        data = read_jsonlines(args.ds_name, args.start_pos)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, device_map='auto')
    model.eval()

    yes_ids = [tokenizer.encode(' yes')[-1], tokenizer.encode(' Yes')[-1]]
    no_ids = [tokenizer.encode(' no')[-1], tokenizer.encode(' No')[-1]]

    results = []
    for item in tqdm(data, desc='Autocheck'):
        sub_scores = []
        for sub in item.get('sub_sents', []):
            prompt = sub + '\nPlease answer yes or no.'
            inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_scores=True)
            logits = output.scores[0][0]
            score = yes_probability(logits, yes_ids, no_ids)
            sub_scores.append(score)
        item['scores'] = sub_scores
        item['score'] = sum(sub_scores)/len(sub_scores) if sub_scores else 0.0
        results.append(item)

    write_jsonlines(args.answer_file, results)

if __name__ == '__main__':
    main()
