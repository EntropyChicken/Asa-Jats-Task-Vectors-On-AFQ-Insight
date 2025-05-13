import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from afqinsight import AFQDataset
from afqinsight.nn.utils import prep_pytorch_data
import pandas as pd
import os
import sys
import argparse
import json
from datetime import datetime
from tqdm import tqdm
import time

# Force immediate output flushing
print("DEBUG: Script starting")
sys.stdout.flush()

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run age prediction experiment on all tracts individually')
parser.add_argument('--output-dir', type=str, default='tract_importance_results', help='Directory to save results')
parser.add_argument('--epochs-stage1', type=int, default=200, help='Number of epochs for stage 1 training')
parser.add_argument('--epochs-stage2', type=int, default=300, help='Number of epochs for stage 2 training')
parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--use-both-fa-md', action='store_true', help='Use both FA and MD data (default: FA only)')
parser.add_argument('--start-tract', type=int, default=0, help='Start from this tract index')
parser.add_argument('--end-tract', type=int, default=47, help='End at this tract index')
args = parser.parse_args()

# Adjust path as needed - update this to your path
sys.path.insert(1, os.path.join(os.getcwd(), 'Experiment_Utils'))
# Import necessary functions
try:
    print("DEBUG: Importing utility functions")
    sys.stdout.flush()
    from utils import select_device, kl_divergence_loss, prep_fa_dataset, train_vae_age_site_staged
    from models import Conv1DVariationalAutoencoder_fa, AgePredictorCNN, SitePredictorCNN, CombinedVAE_Predictors
    print("DEBUG: Successfully imported utility functions")
    sys.stdout.flush()
except Exception as e:
    print(f"ERROR importing utility functions: {str(e)}")
    import traceback
    print(traceback.format_exc())
    sys.exit(1)

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)
print(f"DEBUG: Saving results to {args.output_dir}")
sys.stdout.flush()

# Define custom function for specific tract extraction
def extract_specific_tract_data(dataset, tract_idx, batch_size=32):
    """
    Extract data for a specific tract from a PyTorch dataset.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset or tuple
        Dataset containing tract data or tuple of (dataset, train_loader, test_loader, val_loader)
    tract_idx : int
        Index of the tract to extract (0-47)
    batch_size : int
        Batch size for data loaders

    Returns
    -------
    tuple
        Specific tract train, test, and validation loaders
    """
    print(f"DEBUG: Creating dataset for tract {tract_idx}")
    
    class SingleTractDataset(torch.utils.data.Dataset):
        def __init__(self, original_dataset, tract_idx):
            self.original_dataset = original_dataset
            self.tract_idx = tract_idx

        def __len__(self):
            return len(self.original_dataset)

        def __getitem__(self, idx):
            x, y = self.original_dataset[idx]
            # Extract just the specified tract
            tract_data = x[self.tract_idx:self.tract_idx+1, :].clone()
            return tract_data, y

    # Handle different dataset formats
    if isinstance(dataset, tuple) and len(dataset) == 4:
        # If dataset is a tuple returned from prep_fa_dataset, unpack it
        torch_dataset, train_loader, test_loader, val_loader = dataset
        
        # Extract datasets from loaders
        train_dataset = train_loader.dataset
        test_dataset = test_loader.dataset
        val_dataset = val_loader.dataset
        
        print(f"DEBUG: Using datasets from data loaders, train dataset size: {len(train_dataset)}")
    elif hasattr(dataset, 'train_data') and hasattr(dataset, 'test_data') and hasattr(dataset, 'val_data'):
        # If dataset has train/test/val data attributes
        train_dataset = dataset.train_data
        test_dataset = dataset.test_data
        val_dataset = dataset.val_data
        print(f"DEBUG: Using train/test/val from dataset attributes, train dataset size: {len(train_dataset)}")
    else:
        # If dataset is a single dataset, use it for all (not ideal but fallback)
        print(f"DEBUG: Unknown dataset structure, attempting to use directly. Type: {type(dataset)}")
        if hasattr(dataset, '__getitem__') and hasattr(dataset, '__len__'):
            train_dataset = dataset
            test_dataset = dataset
            val_dataset = dataset
            print(f"DEBUG: Using dataset directly, size: {len(dataset)}")
        else:
            raise ValueError(f"Unsupported dataset type: {type(dataset)}. Cannot extract tract data.")

    # Create single tract datasets
    specific_tract_train = SingleTractDataset(train_dataset, tract_idx)
    specific_tract_test = SingleTractDataset(test_dataset, tract_idx)
    specific_tract_val = SingleTractDataset(val_dataset, tract_idx)

    # Create data loaders
    specific_tract_train_loader = torch.utils.data.DataLoader(
        specific_tract_train, batch_size=batch_size, shuffle=True
    )
    specific_tract_test_loader = torch.utils.data.DataLoader(
        specific_tract_test, batch_size=batch_size, shuffle=False
    )
    specific_tract_val_loader = torch.utils.data.DataLoader(
        specific_tract_val, batch_size=batch_size, shuffle=False
    )
    
    print(f"DEBUG: Created data loaders for tract {tract_idx}. Train size: {len(specific_tract_train)}")
    return specific_tract_train_loader, specific_tract_test_loader, specific_tract_val_loader


def run_tract_experiment(tract_idx, tract_name, base_output_dir):
    """
    Run a complete experiment for a single tract and save results.
    
    Parameters
    ----------
    tract_idx : int
        Index of the tract to analyze
    tract_name : str
        Name of the tract (for display/logs)
    base_output_dir : str
        Base directory to save all results
        
    Returns
    -------
    dict
        Dictionary containing results metrics
    """
    output_dir = os.path.join(base_output_dir, f"tract_{tract_idx}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n\n{'-'*80}")
    print(f"RUNNING EXPERIMENT ON TRACT {tract_idx}: {tract_name}")
    print(f"{'-'*80}\n")
    sys.stdout.flush()
    
    # Record experiment details
    experiment_details = {
        "tract_idx": tract_idx,
        "tract_name": tract_name,
        "epochs_stage1": args.epochs_stage1,
        "epochs_stage2": args.epochs_stage2,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "using_both_fa_md": args.use_both_fa_md,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device)
    }
    
    # Save experiment details
    with open(os.path.join(output_dir, "experiment_details.json"), "w") as f:
        json.dump(experiment_details, f, indent=2)
    
    # Extract the specific tract data
    print(f"Extracting data for tract {tract_idx}")
    sys.stdout.flush()
    
    tract_train_loader, tract_test_loader, tract_val_loader = extract_specific_tract_data(
        dataset_output, tract_idx=tract_idx, batch_size=args.batch_size
    )
    
    # Create a function to transform labels
    def transform_labels(batch_labels):
        """Transform labels to correct format for training."""
        # Extract values
        ages = batch_labels[:, age_idx].float().unsqueeze(1)
        sex_values = batch_labels[:, sex_idx].float().unsqueeze(1)
        site_values = batch_labels[:, site_idx].float()
        
        # Remap site values
        remapped_sites = torch.zeros_like(site_values)
        for i in range(len(site_values)):
            original_site = site_values[i].item()
            remapped_site = site_map.get(original_site, -1.0)
            remapped_sites[i] = remapped_site
        
        # Stack values into a single tensor
        return torch.cat([ages, sex_values, remapped_sites.unsqueeze(1)], dim=1)
    
    # Create DataLoaders with transformed labels
    class RemappedDataLoader:
        def __init__(self, original_loader):
            self.original_loader = original_loader
            
        def __iter__(self):
            for x, y in self.original_loader:
                yield x, transform_labels(y)
                
        def __len__(self):
            return len(self.original_loader)
    
    # Wrap the data loaders
    train_loader_raw = RemappedDataLoader(tract_train_loader)
    test_loader_raw = RemappedDataLoader(tract_test_loader)
    val_loader_raw = RemappedDataLoader(tract_val_loader)
    
    # Get a sample batch to determine input dimensions
    for x_batch, _ in train_loader_raw:
        break
    
    input_channels = x_batch.shape[1]
    sequence_length = x_batch.shape[2]
    print(f"Input shape: channels={input_channels}, sequence_length={sequence_length}")
    
    # Set parameters for the experiment
    latent_dim = 64  # Choose the larger latent dim
    dropout = 0.0  # VAE dropout
    age_dropout = 0.1
    site_dropout = 0.2
    w_recon = 1.0
    w_kl = 0.001
    w_age = 15.0  # Higher weight for age prediction
    w_site = 5.0  # Higher weight for site adversarial training
    
    # Get the number of unique sites from the training data
    unique_sites = set()
    for i, (_, labels) in enumerate(train_loader_raw):
        if i < 5:  # Just check a few batches for debugging
            print(f"Batch {i} site values: {labels[:, 2].tolist()}")
        unique_sites.update(labels[:, 2].tolist())
    num_sites = len(unique_sites)
    print(f"Detected {num_sites} unique site IDs in the data: {sorted(unique_sites)}")
    
    # Create models
    try:
        vae = Conv1DVariationalAutoencoder_fa(latent_dims=latent_dim, dropout=dropout, input_length=sequence_length)
        
        age_predictor = AgePredictorCNN(input_channels=input_channels, 
                                        sequence_length=sequence_length, 
                                        dropout=age_dropout)
        
        site_predictor = SitePredictorCNN(num_sites=num_sites, 
                                         input_channels=input_channels, 
                                         sequence_length=sequence_length, 
                                         dropout=site_dropout)
    except Exception as e:
        print(f"ERROR creating models: {str(e)}")
        import traceback
        print(traceback.format_exc())
        sys.stdout.flush()
        return {"error": str(e)}
    
    # Train the models using staged training
    print(f"Starting staged training for tract {tract_idx}")
    sys.stdout.flush()
    
    start_time = time.time()
    
    try:
        results = train_vae_age_site_staged(
            vae_model=vae,
            age_predictor=age_predictor,
            site_predictor=site_predictor,
            train_data=train_loader_raw,
            val_data=val_loader_raw,
            epochs_stage1=args.epochs_stage1,
            epochs_stage2=args.epochs_stage2,
            lr=args.learning_rate,
            device=device,
            max_grad_norm=1.0,
            w_recon=w_recon,
            w_kl=w_kl,
            w_age=w_age,
            w_site=w_site,
            save_dir=output_dir
        )
        
        training_time = time.time() - start_time
        
        # Add additional metadata to results
        results["tract_idx"] = tract_idx
        results["tract_name"] = tract_name
        results["training_time"] = training_time
        results["training_time_formatted"] = f"{training_time/60:.2f} minutes"
        
        # Save detailed results
        with open(os.path.join(output_dir, "training_results.json"), "w") as f:
            # Convert any non-serializable values to strings first
            serializable_results = {}
            for k, v in results.items():
                if isinstance(v, (int, float, str, bool, list, dict)) and not isinstance(v, (np.int32, np.int64, np.float32, np.float64)):
                    serializable_results[k] = v
                else:
                    try:
                        # Try to convert numpy values to Python native types
                        if isinstance(v, (np.int32, np.int64)):
                            serializable_results[k] = int(v)
                        elif isinstance(v, (np.float32, np.float64)):
                            serializable_results[k] = float(v)
                        else:
                            serializable_results[k] = str(v)
                    except:
                        serializable_results[k] = str(v)
            
            json.dump(serializable_results, f, indent=2)
        
        print(f"Training for tract {tract_idx} completed in {training_time/60:.2f} minutes")
        print(f"Best validation R²: {results.get('best_val_r2', 'N/A')}")
        print(f"Best validation MAE: {results.get('best_val_mae', 'N/A')}")
        sys.stdout.flush()
        
        return results
        
    except Exception as e:
        print(f"ERROR during training: {str(e)}")
        import traceback
        print(traceback.format_exc())
        sys.stdout.flush()
        return {"error": str(e), "tract_idx": tract_idx, "tract_name": tract_name}


# Main script execution
try:
    print("DEBUG: Selecting device")
    sys.stdout.flush()
    device = select_device()
    print(f"DEBUG: Selected device: {device}")
    sys.stdout.flush()

    print("DEBUG: Loading dataset")
    sys.stdout.flush()
    dataset = AFQDataset.from_study('hbn')
    print(f"DEBUG: Loaded dataset with shape: {dataset.X.shape}")
    print(f"DEBUG: Target columns: {dataset.target_cols}")
    sys.stdout.flush()
    
    # Get the tract names for reference
    print("DEBUG: Getting tract names")
    tract_names = []
    for feature_name in dataset.feature_names:
        parts = feature_name.split('_')
        # Extract tract name (format is typically "dki_fa_tract_name_node0")
        if len(parts) >= 4 and parts[2] == 'node0':
            tract_name = parts[1]
            if tract_name not in tract_names:
                tract_names.append(tract_name)
    
    if len(tract_names) > 0:
        print(f"DEBUG: Found {len(tract_names)} tract names")
        # Save tract names to file for later reference
        with open(os.path.join(args.output_dir, "tract_names.json"), "w") as f:
            json.dump(tract_names, f, indent=2)
    else:
        print("WARNING: Could not extract tract names from feature names")
        # Create default tract names
        tract_names = [f"tract_{i}" for i in range(48)]
    
    print("DEBUG: Getting age and site indices")
    sys.stdout.flush()
    age_idx = dataset.target_cols.index('age')
    site_idx = dataset.target_cols.index('scan_site_id')
    sex_idx = dataset.target_cols.index('sex')
    print(f"DEBUG: Found age_idx={age_idx}, site_idx={site_idx}, sex_idx={sex_idx}")
    sys.stdout.flush()

    # Create a site mapping
    site_map = {0.0: 0.0, 1.0: 1.0, 3.0: 2.0, 4.0: 3.0}
    print(f"Using site map: {site_map}")
    
    # Prepare dataset based on chosen modality (FA only or FA+MD)
    print(f"DEBUG: Preparing dataset with {'both FA and MD' if args.use_both_fa_md else 'FA only'}")
    sys.stdout.flush()
    
    if args.use_both_fa_md:
        # Create a combined FA+MD dataset
        dataset_output = prep_fa_dataset(dataset, target_labels=["dki_fa", "dki_md"], batch_size=args.batch_size)
    else:
        # Create an FA-only dataset
        dataset_output = prep_fa_dataset(dataset, target_labels=["dki_fa"], batch_size=args.batch_size)
    
    # Run experiments for each tract in the specified range
    all_results = []
    print(f"Starting experiments for tracts {args.start_tract} to {args.end_tract}")
    
    # Create summary file to track progress
    summary_file = os.path.join(args.output_dir, "summary_results.csv")
    with open(summary_file, 'w') as f:
        f.write("tract_idx,tract_name,best_val_r2,best_val_mae,training_time_minutes,status\n")
    
    for tract_idx in range(args.start_tract, args.end_tract + 1):
        if tract_idx < len(tract_names):
            tract_name = tract_names[tract_idx]
        else:
            tract_name = f"tract_{tract_idx}"
        
        print(f"\nProcessing tract {tract_idx}/{args.end_tract}: {tract_name}")
        
        # Run the experiment
        result = run_tract_experiment(tract_idx, tract_name, args.output_dir)
        all_results.append(result)
        
        # Update summary file
        with open(summary_file, 'a') as f:
            if "error" in result:
                f.write(f"{tract_idx},{tract_name},0,0,0,error\n")
            else:
                f.write(f"{tract_idx},{tract_name},{result.get('best_val_r2', 0)},{result.get('best_val_mae', 0)},{result.get('training_time', 0)/60:.2f},completed\n")
    
    # Create a final summary with tract ranking
    print("\nAnalyzing results...")
    valid_results = [r for r in all_results if "error" not in r]
    
    if valid_results:
        # Sort tracts by R² (descending)
        sorted_by_r2 = sorted(valid_results, key=lambda x: x.get('best_val_r2', 0), reverse=True)
        r2_ranking = pd.DataFrame([
            {
                'tract_idx': r['tract_idx'],
                'tract_name': r['tract_name'],
                'best_val_r2': r.get('best_val_r2', 0),
                'best_val_mae': r.get('best_val_mae', 0)
            } 
            for r in sorted_by_r2
        ])
        
        # Save rankings
        r2_ranking.to_csv(os.path.join(args.output_dir, "tract_ranking_by_r2.csv"), index=False)
        
        # Create visualization of tract importance
        plt.figure(figsize=(12, 8))
        bars = plt.barh(r2_ranking['tract_name'], r2_ranking['best_val_r2'])
        plt.xlabel('Validation R²')
        plt.ylabel('Tract Name')
        plt.title('Tract Importance for Age Prediction (Ranked by R²)')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "tract_importance_r2.png"))
        
        # Print top 5 tracts
        print("\nTop 5 tracts by R²:")
        top5 = r2_ranking.head(5)
        for i, row in top5.iterrows():
            print(f"{i+1}. {row['tract_name']} (idx: {row['tract_idx']}): R²={row['best_val_r2']:.4f}, MAE={row['best_val_mae']:.4f}")
    
    print("\nExperiment completed!")
    
except Exception as e:
    print(f"ERROR: {str(e)}")
    import traceback
    print(traceback.format_exc())
    sys.exit(1) 