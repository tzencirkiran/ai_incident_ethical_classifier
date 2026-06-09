# AI Ethics Incident Classification

Multi-label classification for AI incident reports from the AIAAIC dataset. The project predicts ethical-issue tags such as `Privacy/surveillance`, `Fairness`, `Safety`, and `Transparency` from incident text plus selected metadata.

## What is included

- A TF-IDF + logistic regression baseline
- A fine-tuned `prajjwal1/bert-tiny` model
- Split-aware preprocessing for random and temporal evaluations
- Per-label threshold tuning on validation data
- Split-specific tuning and result artifacts

## Repository Layout

- `data/` raw spreadsheet inputs
- `model/preprocessing.py` builds processed train/val/test artifacts
- `model/baseline.py` trains the TF-IDF baseline
- `model/fine_tune.py` trains TinyBERT on a processed split
- `model/tune.py` runs the TinyBERT hyperparameter sweep
- `model/evaluate.py` evaluates the saved TinyBERT checkpoint
- `model/infer.py` runs single-incident inference
- `Makefile` wraps the common commands

## Splits

The pipeline writes split artifacts under `model/processed/<split>/`.

- `random` is the default split
- `temporal` uses `Occurred` year cutoffs to simulate future generalization

Example temporal split:

```bash
make preprocess SPLIT=temporal
```

## Quick Start

Random split:

```bash
make preprocess
make baseline
make tune
make train
make test
```

Temporal split:

```bash
make preprocess SPLIT=temporal
make baseline SPLIT=temporal
make tune SPLIT=temporal
make train SPLIT=temporal ARGS="--epochs 20 --batch-size 32 --learning-rate 0.0007 --weight-decay 0.01 --warmup-ratio 0.2 --weighted-bce --seed 42"
make test SPLIT=temporal
```

## Command Reference

- `make preprocess [SPLIT=...] [ARGS="..."]`
  - Builds processed artifacts in `model/processed/<split>/`
- `make baseline [SPLIT=...]`
  - Trains TF-IDF logistic regression and saves metrics under `model/results/<split>/`
- `make tune [SPLIT=...] [TUNING_CONFIG=...]`
  - Runs the TinyBERT hyperparameter sweep for the selected split
  - Saves sweep results under `model/tuning/<split>/`
- `make train [SPLIT=...] [ARGS="..."]`
  - Fine-tunes TinyBERT and saves the checkpoint in `model/checkpoint/`
- `make test [SPLIT=...]`
  - Evaluates the saved TinyBERT checkpoint on the chosen split
- `make infer HEADLINE="..." [PURPOSE="..."] [TECHNOLOGY="..."] [SECTOR="..."]`
  - Runs a single prediction against the saved TinyBERT checkpoint

## Tuning Profiles

The sweep config is split-aware:

- `model/tuning_configs/random.json`
- `model/tuning_configs/temporal.json`

If a split-specific file exists, `make tune SPLIT=<name>` uses it automatically.

## Latest Observed Scores

These are the latest held-out results from the current repo state:

| Split | Model | Micro F1 | Macro F1 |
| --- | --- | ---: | ---: |
| random | TF-IDF baseline | 0.6057 | 0.5600 |
| random | TinyBERT | 0.5639 | 0.5048 |
| temporal | TF-IDF baseline | 0.5813 | 0.4715 |
| temporal | TinyBERT | 0.5795 | 0.4545 |

TinyBERT improved after tuning, but the baseline still wins on the held-out test sets.

## Output Locations

- `model/processed/<split>/`
  - `train.pt`, `val.pt`, `test.pt`
  - `label_classes.npy`
  - `split_metadata.json`
- `model/results/<split>/`
  - `baseline_metrics.json`
  - `baseline_per_label.csv`
- `model/tuning/<split>/`
  - `tinybert_grid_results.json`
  - `tinybert_grid_results.csv`
- `model/checkpoint/`
  - TinyBERT checkpoint, tokenizer, thresholds, and training metrics

## Notes

- The TF-IDF baseline is the current deployment candidate because it performs best on held-out test data.
- TinyBERT is kept as the neural comparison model and for future robustness experiments.
- The repository uses split-specific artifacts so random and temporal experiments do not overwrite one another.
