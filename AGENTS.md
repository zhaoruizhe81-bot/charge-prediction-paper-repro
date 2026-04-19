# Repository Guidelines

## Project Structure & Module Organization

This repository contains a paper reproduction project for charge prediction. The active codebase is `res3_repro/`.

- `charge_prediction/`: reusable Python package for data processing, models, metrics, fusion, and hierarchical classification.
- `scripts/`: command-line entry points for data preparation, training, prediction, smoke checks, and result table generation.
- `data/processed_110_paper/`: processed JSONL splits and label maps used by training scripts.
- `../2018数据集/2018数据集/`: raw CAIL-style dataset files when working from the parent folder.
- `outputs_*`: generated metrics, checkpoints, and comparison tables. Treat these as reproducible artifacts, not source code.

## Build, Test, and Development Commands

Run commands from this directory unless noted.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Installs the Python runtime dependencies. Use a CUDA-specific PyTorch build when training on GPU.

```bash
python scripts/prepare_data.py --data-dir ../2018数据集/2018数据集 --output-dir data/processed_110_paper --target-size 50000 --seed 42
```

Rebuilds processed train, validation, and test splits.

```bash
python scripts/smoke_test_flat.py
python scripts/smoke_test_hier_fc.py
```

Runs environment and model smoke checks. These expect processed data, local BERT weights, and CUDA unless adjusted.

```bash
python scripts/make_results_table.py --output-dir outputs_paper --save-path outputs_paper/results_table.csv
```

Regenerates the summary results table.

## Coding Style & Naming Conventions

Use Python 3.10+ style with 4-space indentation, type hints where practical, and `pathlib.Path` for filesystem paths. Keep reusable logic in `charge_prediction/`; keep scripts thin and argument-driven. Use `snake_case` for modules, functions, variables, and script names. Preserve existing bilingual documentation where it clarifies reproduction steps.

## Testing Guidelines

There is no formal pytest suite in this snapshot. Validate changes with the smallest relevant smoke script first, then run the affected training or table-generation command. For data changes, inspect generated `dataset_stats.json`, label maps, and a few JSONL rows before committing outputs.

## Commit & Pull Request Guidelines

The Git history uses short imperative messages such as `Add optimized training pipeline for 4060 Ti` and `Fix coarse-label normalization for hierarchical training`. Follow that style: one focused change per commit, present tense, under about 72 characters when possible.

Pull requests should describe the experiment or code path changed, list commands run, include key metric deltas or artifact paths, and mention required local assets such as `chinese-bert-wwm-ext/`.

## Security & Configuration Tips

Do not commit `.venv/`, local BERT weights, checkpoints, `.env*`, or generated output directories. Raw legal case data may contain sensitive text; keep dataset copies local and avoid adding new raw extracts unless explicitly required.
