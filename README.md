# DSPA

Code and resources for **Same Pool, Different Answer: Stable Best-of-N Selection for Vision-Language Models**.

DSPA studies a simple failure mode in Best-of-N (BoN) selection for vision-language models: even when the candidate pool is fixed, small scoring changes can select a different answer. The method separates selection stability from candidate-pool quality:

- **Anchored selector:** at test time, candidates are scored by an averaged token-level log-ratio against a fixed reference model.
- **Preference filtering:** at training time, preference pairs are filtered to reduce shortcuts from answer length, prompt copying, and object hallucination.
- **SRP audit:** Same-pool/Same-budget Replay evaluates whether the same frozen pool keeps the same winner under controlled scoring perturbations.

In the paper, standard likelihood-based BoN selection flips the winner on about 41-47% of fixed-pool test items under small scoring perturbations. DSPA reduces the average flip rate from 44.3% to 24.9% and lowers object hallucination on identical pools from 9.6% to 6.2% on POPE.

## Repository Layout

```text
.
├── muffin/                  # Core training and model utilities
├── llava/                   # LLaVA-related model code
├── omnilmm/                 # OmniLMM-related model code
├── minicpm-llama3-v-25/     # MiniCPM-V feedback utilities
├── script/
│   ├── data_gen/            # Data generation and preference-pair construction
│   ├── train/               # Training scripts
│   └── eval/                # Evaluation scripts
├── eval/                    # Evaluation data and metric scripts
├── utils/                   # Pair construction and filtering helpers
└── examples/                # Example images and small inputs
```

## Installation

```bash
git clone https://github.com/PengfeiZheng-721/DSPA.git
cd DSPA

conda create -n dspa python=3.10 -y
conda activate dspa
pip install -e .
```

Some data-construction utilities use spaCy:

```bash
wget https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.7.3/en_core_web_trf-3.7.3.tar.gz
pip install en_core_web_trf-3.7.3.tar.gz
```

## Main Entry Points

Data generation and preference construction:

```bash
bash script/data_gen/run_data_pipeline_llava15_omni.sh
bash script/data_gen/run_data_pipeline_llava15_minicpmv.sh
```

Training:

```bash
bash script/train/llava15_train.sh
bash script/train/llava15_train_lora.sh
```

Evaluation:

```bash
bash script/eval/eval_rlaifv_objhal.sh
bash script/eval/eval_rlaifv_mmhal.sh
bash script/eval/run_refomb_overall.sh
bash script/eval/run_refomb_hall.sh
```

These scripts may require local model checkpoints, benchmark files, and API keys depending on the experiment.

## DSPA Reproducibility Code

The `dspa/` package contains the code path for the paper-specific DSPA functionality:

- `dspa.selectors`: policy-only length normalization, epsilon-margin tie-break, sequence-level log-ratio, per-token PMI, co-moving reference, and the anchored selector.
- `dspa.scoring`: score-time temperature, top-p renormalization that retains the realized token, dropout seed helpers, and canonical SRP perturbation keys.
- `dspa.srp`: Same-pool/Same-budget Replay with flip-rate, agreement, Wilson intervals, and deterministic smallest-index tie-breaking.
- `dspa.filters`: training-time prompt-copy, length-ratio, object-existence hallucination, caption-tail, and max-pairs-per-input filters.
- `dspa.manifest`: fixed-pool JSONL schema helpers, candidate hashes, pool hashes, and manifest validation.

Validate a fixed-pool manifest:

```bash
python scripts/dspa/validate_manifest.py \
  --manifest artifacts/pools/llava15_7b_n8.jsonl \
  --require-hashes
```

Run the anchored selector on a scored pool:

```bash
python scripts/dspa/select_pool.py \
  --manifest artifacts/scores/llava15_7b_n8_scored.jsonl \
  --selector anchored \
  --score-key native \
  --output artifacts/results/anchored_winners.jsonl
```

Run SRP replay:

```bash
python scripts/dspa/run_srp.py \
  --manifest artifacts/scores/llava15_7b_n8_scored.jsonl \
  --selector anchored \
  --family dropout=dropout/seed0,dropout/seed1 \
  --family delta_temperature=delta_temperature/tau0.65,delta_temperature/tau0.75 \
  --family delta_top_p=delta_top_p/p0.85,delta_top_p/p0.95 \
  --output artifacts/results/srp_anchored.json
```

Apply DSPA training-time filters:

```bash
python scripts/dspa/apply_filters.py \
  --input artifacts/pairs/raw_pairs.jsonl \
  --output artifacts/pairs/dspa_filtered_pairs.jsonl \
  --summary artifacts/pairs/dspa_filter_summary.json
```

Print the canonical perturbation plan used by SRP:

```bash
python scripts/dspa/replay_score_keys.py
```

Export SRP JSON results as a table:

```bash
python scripts/dspa/export_srp_table.py \
  artifacts/results/srp_anchored.json \
  --format md \
  --output artifacts/results/table_s3_anchored.md
```

The public artifact should provide fixed-pool manifests and scored manifests. The code intentionally does not redistribute third-party model weights; use model identifiers and revisions for policy, reference, and judge checkpoints.

## Method Summary

For each input, BoN generation produces a fixed candidate pool. DSPA keeps the pool unchanged and changes only the scoring rule. Instead of ranking candidates by raw policy likelihood, the anchored selector compares each candidate token-by-token with a frozen reference model and averages the resulting log-ratio over the answer length. This reduces instability caused by length accumulation and score-scale drift.

The training-time filter is separate: it improves the quality of preference pairs by removing examples where the preference can be explained by superficial artifacts rather than better visual grounding.

## Citation

```bibtex
@inproceedings{zheng2026samepool,
  title     = {Same Pool, Different Answer: Stable Best-of-N Selection for Vision-Language Models},
  author    = {Zheng, Pengfei},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## Acknowledgement

This repository was split from the DSPA branch of `PengfeiZheng-721/RLAIF-V` and builds on the RLAIF-V/LLaVA-style training and evaluation codebase.
