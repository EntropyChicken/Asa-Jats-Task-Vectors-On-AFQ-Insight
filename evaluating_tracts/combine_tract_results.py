#!/usr/bin/env python
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import glob

def main():
    parser = argparse.ArgumentParser(description="Combine results from multiple tract evaluation jobs")
    parser.add_argument('--input-dir', type=str, required=True, help='Directory containing tract evaluation results')
    parser.add_argument('--output-file', type=str, default='combined_tract_ranking.csv', help='Output CSV file for combined results')
    parser.add_argument('--create-plot', action='store_true', help='Create visualization plot')
    parser.add_argument('--top-n', type=int, default=10, help='Show top N tracts in final report')
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return
    
    # Find all tract directories
    tract_dirs = glob.glob(os.path.join(args.input_dir, "tract_*"))
    if not tract_dirs:
        print(f"No tract directories found in {args.input_dir}")
        return
    
    print(f"Found {len(tract_dirs)} tract directories")
    
    # Load tract names if available
    tract_names_file = os.path.join(args.input_dir, "tract_names.json")
    tract_names = {}
    if os.path.exists(tract_names_file):
        with open(tract_names_file, 'r') as f:
            tract_list = json.load(f)
            tract_names = {i: name for i, name in enumerate(tract_list)}
    
    # Collect all results
    all_results = []
    for tract_dir in tract_dirs:
        tract_idx = int(tract_dir.split('_')[-1])
        
        # Try to load summary file
        summary_file = os.path.join(tract_dir, "tract_summary.json")
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                all_results.append(summary)
        else:
            # Look for training_results.json as fallback
            results_file = os.path.join(tract_dir, "training_results.json")
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    try:
                        results = json.load(f)
                        # Extract key metrics
                        summary = {
                            "tract_idx": tract_idx,
                            "tract_name": tract_names.get(tract_idx, f"tract_{tract_idx}"),
                            "best_r2": results.get("best_val_r2", 0),
                            "best_age_mae": results.get("best_val_mae", 0)
                        }
                        all_results.append(summary)
                    except json.JSONDecodeError:
                        print(f"Error parsing JSON from {results_file}")
    
    if not all_results:
        print("No results found in tract directories")
        return
    
    print(f"Collected results from {len(all_results)} tracts")
    
    # Convert to DataFrame and sort by R²
    df = pd.DataFrame(all_results)
    
    # Ensure we have the right columns
    if 'best_r2' not in df.columns and 'best_val_r2' in df.columns:
        df['best_r2'] = df['best_val_r2']
    
    if 'best_age_mae' not in df.columns and 'best_val_age_mae' in df.columns:
        df['best_age_mae'] = df['best_val_age_mae']
    
    # Sort by R² (descending)
    if 'best_r2' in df.columns:
        df = df.sort_values('best_r2', ascending=False)
    
    # Save combined results
    df.to_csv(args.output_file, index=False)
    print(f"Saved combined results to {args.output_file}")
    
    # Create visualization if requested
    if args.create_plot and 'best_r2' in df.columns:
        plt.figure(figsize=(12, 10))
        
        # Plot top 20 tracts or all if less than 20
        plot_df = df.head(min(20, len(df)))
        
        bars = plt.barh(plot_df['tract_name'], plot_df['best_r2'])
        plt.xlabel('Validation R²')
        plt.ylabel('Tract Name')
        plt.title('Tract Importance for Age Prediction (Ranked by R²)')
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(args.input_dir, "tract_importance_r2.png")
        plt.savefig(plot_file)
        print(f"Saved visualization to {plot_file}")
    
    # Print top N tracts
    print(f"\nTop {args.top_n} tracts by R²:")
    for i, row in df.head(args.top_n).iterrows():
        r2_value = row.get('best_r2', 0)
        mae_value = row.get('best_age_mae', 0)
        print(f"{i+1}. {row['tract_name']} (idx: {row['tract_idx']}): R²={r2_value:.4f}, MAE={mae_value:.4f}")

if __name__ == "__main__":
    main() 