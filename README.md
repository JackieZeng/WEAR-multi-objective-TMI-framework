# WEAR: Multi-objective Travel Mode Identification

This repository contains the implementation of **WEAR** from the paper:

> **A multi-objective framework for travel mode identification across hierarchical travel semantics**

WEAR identifies travel modes from GPS trajectories and evaluates predictions at the point, transfer, trip-leg, and mode-chain levels. The implementation combines convolutional and bidirectional recurrent encoders with CRF decoding, weighted classification loss, and adjacent contrastive loss.

## Setup

Use a Python environment (the main modules have been checked with Python 3.9). Clone the repository and install the training dependencies:

```bash
git clone https://github.com/JackieZeng/WEAR-multi-objective-TMI-framework.git
cd WEAR-multi-objective-TMI-framework
python -m pip install torch numpy pandas scipy scikit-learn pyarrow
```

For GPU training, install a CUDA-enabled build of PyTorch appropriate for your environment. The script automatically uses CUDA when available and otherwise uses the CPU.

## Data

The prepared data and three-fold test splits are included:

```text
data/
  trajseq.parquet           # Prepared trajectory data
  train_test_split/
    trajsplit/             # Test trajectory IDs for folds 0–2
    usersplit/             # Test user IDs for folds 0–2
tmi/
  main.py                  # Training and evaluation
  custom_pypkg/            # Models, losses, data utilities, and metrics
  paper_fig_param_analysis.py
```

The main script resolves data paths relative to the repository, so no path changes are needed for the included data.

## Run

From the repository root:

```bash
python tmi/main.py
```

By default, this runs 3 random seeds × 2 split strategies × 4 loss configurations × 3 folds, for **72 training runs**, each with up to 15 epochs. Each trained model is evaluated both with and without CRF decoding.

Experiment settings are defined directly in `tmi/main.py`, including `seed_list`, `split_list`, `loss_param_list`, `num_folds`, and `max_epoch`. For a shorter initial run, set `seed_list = [0]`, `split_list = ['trajsplit']`, `num_folds = 1`, and `max_epoch = 1`. Restore the defaults for the full experiments.

Results are saved under `tmi/`:

- `log/`: configuration, training logs, and evaluation summaries.
- `well-trained-model/<split>/`: model weights (`.pth`).
- `identified_trajectories/<split>/`: predicted trajectories (`.parquet`).
- `result records (auto-save)/<split>/`: evaluation metrics (`.csv`).

## Plot results (optional)

After the full default experiments finish, install the plotting dependencies and run:

```bash
python -m pip install matplotlib adjustText
python tmi/paper_fig_param_analysis.py
```

This script reads the saved evaluation records for all default seeds and configurations, writes figures to `tmi/figures/`, and saves summary CSV files in `tmi/result records (auto-save)/`.

## License

This project is released under the [MIT License](LICENSE).
