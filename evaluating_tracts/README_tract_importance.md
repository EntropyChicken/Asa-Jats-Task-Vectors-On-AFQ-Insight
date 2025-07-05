# Tract Importance Evaluation: Individual Tract Age Prediction Analysis

## Purpose
These experiments evaluate the importance of individual brain tracts for age prediction by training separate models on each tract. The goal is to identify which tracts contain the most biological signal for age prediction and rank them by their predictive power.

## Experimental Setup

### Data Processing
- **Individual Tract Analysis**: Train separate VAE models on each of the 48 brain tracts
- **Staged Training**: Use two-stage approach (reconstruction + age prediction)
- **Data Options**: Support for FA-only or FA+MD measurements
- **Systematic Evaluation**: Process all tracts or specified ranges

### Main Experiments
1. **Single Tract Models**: Train VAE with age prediction head for each tract individually
2. **Performance Ranking**: Rank tracts by R² and MAE for age prediction
3. **Comparative Analysis**: Identify which tracts are most predictive of biological age

## Expected Findings

### Tract Importance Ranking
- **Most Predictive Tracts**: Which individual tracts best predict age?
- **Biological Signal Distribution**: How is age-related information distributed across tracts?
- **Performance Comparison**: Relative importance of each tract for age prediction

### Outputs
- `tract_ranking_by_r2.csv`: Ranking of all tracts by R² performance
- `tract_importance_r2.png`: Visualization of tract importance
- `summary_results.csv`: Complete performance metrics for all tracts
- Individual tract results with detailed training metrics

## Usage
```bash
python tract_importance_evaluation.py --output-dir results_fa_only
python tract_importance_evaluation.py --use-both-fa-md --start-tract 10 --end-tract 20
``` 