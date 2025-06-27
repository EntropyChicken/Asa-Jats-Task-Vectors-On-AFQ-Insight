# Tract Importance Evaluation for Age Prediction

This script evaluates the importance of individual brain tracts for age prediction by training and evaluating models on each tract separately. It uses a variational autoencoder (VAE) approach with staged training to identify which tracts contribute most to age prediction accuracy.

# Goal is to see how much biological signal is contained in
# each individual tract 

## Features

- Automatically processes all tracts (or a specified range)
- Supports using both FA (fractional anisotropy) and MD (mean diffusivity) data
- Trains a separate model for each tract
- Ranks tracts by their predictive power (R² and MAE)
- Generates visualizations and detailed results
- Provides both CSV summaries and per-tract detailed results

## Usage

```bash
python tract_importance_evaluation.py [OPTIONS]
```

### Options

- `--output-dir PATH`: Directory to save results (default: 'tract_importance_results')
- `--epochs-stage1 N`: Number of epochs for stage 1 training (default: 200)
- `--epochs-stage2 N`: Number of epochs for stage 2 training (default: 300)
- `--batch-size N`: Batch size for training (default: 128)
- `--learning-rate FLOAT`: Learning rate (default: 0.001)
- `--use-both-fa-md`: Use both FA and MD data (default: FA only)
- `--start-tract N`: Start from this tract index (default: 0)
- `--end-tract N`: End at this tract index (default: 47)

## Examples

### Run on all tracts using only FA data

```bash
python tract_importance_evaluation.py --output-dir results_fa_only
```

### Run on all tracts using both FA and MD data

```bash
python tract_importance_evaluation.py --output-dir results_fa_md --use-both-fa-md
```

### Run on a specific subset of tracts

```bash
python tract_importance_evaluation.py --start-tract 10 --end-tract 20
```

### Run with custom training parameters

```bash
python tract_importance_evaluation.py --epochs-stage1 150 --epochs-stage2 250 --batch-size 64 --learning-rate 0.0005
```

## Output Files

The script generates the following output structure:

```
tract_importance_results/
├── tract_names.json             # List of all tract names
├── summary_results.csv          # Summary metrics for all tracts
├── tract_ranking_by_r2.csv      # Ranking of tracts by R²
├── tract_importance_r2.png      # Visualization of tract importance
├── tract_0/                     # Results for tract 0
│   ├── experiment_details.json  # Parameters used
│   ├── training_results.json    # Detailed training metrics
│   └── ... model files ...
├── tract_1/
│   └── ...
└── ...
```

## Interpreting Results

After running the script, check the following key outputs:

1. `tract_ranking_by_r2.csv`: Provides a ranking of all tracts by their R² value
2. `tract_importance_r2.png`: Visualization showing the relative importance of each tract
3. The script will also print the top 5 most important tracts upon completion

## Requirements

- PyTorch
- NumPy
- Matplotlib
- Pandas
- AFQ-Insight package (for AFQDataset)
- tqdm for progress tracking

## Notes

- Running the full experiment on all 48 tracts can take significant time
- Consider using the `--start-tract` and `--end-tract` options to split the work
- For preliminary exploration, you can reduce the number of epochs
- GPU acceleration is recommended and will be used if available 