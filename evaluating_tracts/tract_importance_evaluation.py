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

"""
Tract Importance Evaluation Script

This script evaluates the importance of individual brain tracts for age prediction
using deep learning models. It runs a separate neural network for each tract and 
compares their performance.

TRACT NAMING CONVENTION:
- In this code, each tract has a unique name that includes its modality (dki_fa or dki_md) 
  followed by the actual tract name.
- For example: "dki_faSLF_left" for the left Superior Longitudinal Fasciculus using
  Fractional Anisotropy data.
- This naming scheme allows us to distinguish between the same tract measured with
  different modalities (FA vs MD).
- Each unique combination is treated as a separate tract in the analysis.

When using both Fractional Anisotropy (FA) and Mean Diffusivity (MD) data:
- We can evaluate up to 48 distinct tracts (24 FA + 24 MD tracts)
- Results will show which tracts and which metrics (FA or MD) are most predictive
- Direct comparisons between FA and MD versions of the same tract will be generated

The output includes:
- Rankings of all tracts by R² and MAE
- Separate rankings for FA and MD tracts
- Direct comparisons between FA and MD for the same tracts
- Visualizations showing the most age-predictive brain regions and measurements

RUNNING THE SCRIPT:
- Use --use-both-fa-md to include both FA and MD measurements
- Use --start-tract and --end-tract to specify which tracts to analyze
"""

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
# sys.path.insert(1, os.path.join(os.getcwd(), 'Experiment_Utils'))
sys.path.insert(1, '/mmfs1/gscratch/nrdg/samchou/AFQ-Insight-Autoencoder-Experiments/Experiment_Utils')
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
        Name of the tract (for display/logs) including modality
    base_output_dir : str
        Base directory to save all results
        
    Returns
    -------
    dict
        Dictionary containing results metrics
    """
    # Extract modality from the tract name if available
    modality = "unknown"
    if "dki_fa" in tract_name:
        modality = "dki_fa"
        base_name = tract_name.replace("dki_fa", "")
    elif "dki_md" in tract_name:
        modality = "dki_md"
        base_name = tract_name.replace("dki_md", "")
    else:
        base_name = tract_name
    
    # Create a directory name that includes both the index and modality
    output_dir = os.path.join(base_output_dir, f"tract_{tract_idx}_{modality}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n\n{'-'*80}")
    print(f"RUNNING EXPERIMENT ON TRACT {tract_idx}: {tract_name} (modality: {modality})")
    print(f"{'-'*80}\n")
    sys.stdout.flush()
    
    # Record experiment details
    experiment_details = {
        "tract_idx": tract_idx,
        "tract_name": tract_name,
        "modality": modality,
        "base_tract_name": base_name,
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
        
        # Extract important metrics for the summary
        best_val_r2 = None
        best_val_mae = None
        best_site_acc = None
        
        # Extract R² and MAE from age predictor results
        if 'age_predictor' in results and 'val_r2_epoch' in results['age_predictor']:
            age_val_r2 = results['age_predictor']['val_r2_epoch']
            age_best_index = np.argmax(age_val_r2) if len(age_val_r2) > 0 else -1
            if age_best_index >= 0:
                best_val_r2 = age_val_r2[age_best_index]
                
            if 'val_loss_epoch' in results['age_predictor']:
                age_val_mae = results['age_predictor']['val_loss_epoch']  # MAE is the loss for age predictor
                best_val_mae = min(age_val_mae) if len(age_val_mae) > 0 else None
        
        # Extract site accuracy from site predictor results
        if 'site_predictor' in results and 'val_acc_epoch' in results['site_predictor']:
            site_val_acc = results['site_predictor']['val_acc_epoch']
            best_site_acc = max(site_val_acc) if len(site_val_acc) > 0 else None
        
        # Also check combined model results
        if 'combined' in results:
            combined_results = results['combined']
            
            # Check for R² in combined results
            if 'val_age_r2_epoch' in combined_results:
                combined_val_r2 = combined_results['val_age_r2_epoch']
                combined_best_r2 = max(combined_val_r2) if len(combined_val_r2) > 0 else None
                if combined_best_r2 is not None and (best_val_r2 is None or combined_best_r2 > best_val_r2):
                    best_val_r2 = combined_best_r2
            
            # Check for MAE in combined results
            if 'val_age_mae_epoch' in combined_results:
                combined_val_mae = combined_results['val_age_mae_epoch']
                combined_best_mae = min(combined_val_mae) if len(combined_val_mae) > 0 else None
                if combined_best_mae is not None and (best_val_mae is None or combined_best_mae < best_val_mae):
                    best_val_mae = combined_best_mae
            
            # Check for site accuracy in combined results
            if 'val_site_acc_epoch' in combined_results:
                combined_site_acc = combined_results['val_site_acc_epoch']
                combined_best_site_acc = max(combined_site_acc) if len(combined_site_acc) > 0 else None
                if combined_best_site_acc is not None and (best_site_acc is None or combined_best_site_acc > best_site_acc):
                    best_site_acc = combined_best_site_acc
        
        # Add best metrics to the top level of results
        results['best_val_r2'] = best_val_r2
        results['best_val_mae'] = best_val_mae
        results['best_site_acc'] = best_site_acc
        
        # Save training history as CSV files for easier analysis
        # Save VAE training history
        if 'vae' in results:
            vae_history = pd.DataFrame({
                'epoch': list(range(1, len(results['vae']['train_loss_epoch']) + 1)),
                'train_loss': results['vae']['train_loss_epoch'],
                'val_loss': results['vae']['val_loss_epoch'],
                'train_recon_loss': results['vae']['train_recon_loss_epoch'],
                'val_recon_loss': results['vae']['val_recon_loss_epoch'],
                'train_kl_loss': results['vae']['train_kl_loss_epoch'],
                'val_kl_loss': results['vae']['val_kl_loss_epoch'],
                'beta': results['vae']['current_beta_epoch']
            })
            vae_history.to_csv(os.path.join(output_dir, 'vae_training_history.csv'), index=False)
        
        # Save Age Predictor training history
        if 'age_predictor' in results:
            age_history = pd.DataFrame({
                'epoch': list(range(1, len(results['age_predictor']['train_loss_epoch']) + 1)),
                'train_mae': results['age_predictor']['train_loss_epoch'],
                'val_mae': results['age_predictor']['val_loss_epoch'],
                'train_r2': results['age_predictor']['train_r2_epoch'],
                'val_r2': results['age_predictor']['val_r2_epoch']
            })
            age_history.to_csv(os.path.join(output_dir, 'age_predictor_training_history.csv'), index=False)
        
        # Save Site Predictor training history
        if 'site_predictor' in results:
            site_history = pd.DataFrame({
                'epoch': list(range(1, len(results['site_predictor']['train_loss_epoch']) + 1)),
                'train_loss': results['site_predictor']['train_loss_epoch'],
                'val_loss': results['site_predictor']['val_loss_epoch'],
                'train_acc': results['site_predictor']['train_acc_epoch'],
                'val_acc': results['site_predictor']['val_acc_epoch']
            })
            site_history.to_csv(os.path.join(output_dir, 'site_predictor_training_history.csv'), index=False)
            
        # Save Combined Model training history
        if 'combined' in results:
            combined_cols = {
                'epoch': list(range(1, len(results['combined']['train_loss_epoch']) + 1)),
                'train_loss': results['combined']['train_loss_epoch'],
                'val_loss': results['combined']['val_loss_epoch']
            }
            
            # Add other metrics if they exist
            for metric in ['train_recon_loss_epoch', 'val_recon_loss_epoch', 
                          'train_kl_loss_epoch', 'val_kl_loss_epoch',
                          'train_age_loss_epoch', 'val_age_loss_epoch',
                          'train_site_loss_epoch', 'val_site_loss_epoch',
                          'train_age_mae_epoch', 'val_age_mae_epoch',
                          'train_site_acc_epoch', 'val_site_acc_epoch',
                          'train_age_r2_epoch', 'val_age_r2_epoch',
                          'current_beta_epoch', 'current_grl_alpha_epoch']:
                if metric in results['combined']:
                    # Create a friendlier column name by removing _epoch suffix
                    col_name = metric.replace('_epoch', '')
                    combined_cols[col_name] = results['combined'][metric]
            
            combined_history = pd.DataFrame(combined_cols)
            combined_history.to_csv(os.path.join(output_dir, 'combined_model_training_history.csv'), index=False)
            
            # Create a dedicated summary file for adversarial training (stage 2) results
            # Find the best R² and MAE values from the combined model training
            combined_val_r2 = results['combined'].get('val_age_r2_epoch', [])
            combined_val_mae = results['combined'].get('val_age_mae_epoch', [])
            combined_val_site_acc = results['combined'].get('val_site_acc_epoch', [])
            
            best_combined_r2 = max(combined_val_r2) if combined_val_r2 else None
            best_combined_r2_epoch = combined_val_r2.index(best_combined_r2) + 1 if best_combined_r2 is not None else None
            
            best_combined_mae = min(combined_val_mae) if combined_val_mae else None
            best_combined_mae_epoch = combined_val_mae.index(best_combined_mae) + 1 if best_combined_mae is not None else None
            
            # Find the best site accuracy, which is interesting for adversarial training
            # Lower site accuracy can indicate better site-invariant features
            best_combined_site_acc = max(combined_val_site_acc) if combined_val_site_acc else None
            worst_combined_site_acc = min(combined_val_site_acc) if combined_val_site_acc else None
            
            # Create a summary DataFrame for stage 2 (adversarial training)
            adversarial_summary = {
                'tract_idx': tract_idx,
                'tract_name': tract_name,
                'best_val_r2': best_combined_r2,
                'best_val_r2_epoch': best_combined_r2_epoch,
                'best_val_mae': best_combined_mae,
                'best_val_mae_epoch': best_combined_mae_epoch,
                'best_site_acc': best_combined_site_acc,
                'worst_site_acc': worst_combined_site_acc,
                'best_epoch': results['combined'].get('best_epoch', None),
                'best_metric_value': results['combined'].get(f'best_{val_metric_to_monitor}', None),
                'total_epochs': len(combined_val_r2)
            }
            
            # Save the adversarial training summary
            pd.DataFrame([adversarial_summary]).to_csv(
                os.path.join(output_dir, 'adversarial_training_summary.csv'), index=False
            )
            print(f"Saved adversarial training (stage 2) summary to {os.path.join(output_dir, 'adversarial_training_summary.csv')}")
        
        # Save detailed results as JSON
        try:
            # Convert numpy arrays and other non-serializable types to lists or primitives
            serializable_results = {}
            
            def make_serializable(obj):
                if isinstance(obj, (np.ndarray, list, tuple)):
                    return [float(x) if isinstance(x, (np.float32, np.float64)) else 
                            int(x) if isinstance(x, (np.int32, np.int64)) else x 
                            for x in obj]
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                else:
                    return obj
            
            # Process each top-level key
            for key, value in results.items():
                if isinstance(value, dict):
                    serializable_results[key] = make_serializable(value)
                else:
                    serializable_results[key] = make_serializable(value)
                    
            with open(os.path.join(output_dir, "training_results.json"), "w") as f:
                json.dump(serializable_results, f, indent=2)
                
        except Exception as e:
            print(f"WARNING: Error serializing full results: {str(e)}")
            # Fall back to saving a simplified version
            with open(os.path.join(output_dir, "training_results_simple.json"), "w") as f:
                simple_results = {
                    "tract_idx": tract_idx,
                    "tract_name": tract_name,
                    "best_val_r2": best_val_r2,
                    "best_val_mae": best_val_mae,
                    "best_site_acc": best_site_acc,
                    "training_time": training_time,
                    "training_time_formatted": f"{training_time/60:.2f} minutes"
                }
                json.dump(simple_results, f, indent=2)
        
        print(f"Training for tract {tract_idx} completed in {training_time/60:.2f} minutes")
        print(f"Best validation R²: {best_val_r2}")
        print(f"Best validation MAE: {best_val_mae}")
        print(f"Best site accuracy: {best_site_acc}%")
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
    print(f"DEBUG: First few feature_names: {dataset.feature_names[:5]}")
    print(f"DEBUG: Type of first feature_name: {type(dataset.feature_names[0])}")
    
    # Attempt to get tract information based on feature structure
    tract_names = []
    tract_data_types = {}  # Store data type (FA/MD) for each tract
    
    # Check how the feature names are structured and extract tracts accordingly
    if len(dataset.feature_names) > 0:
        if isinstance(dataset.feature_names[0], tuple):
            print("DEBUG: Feature names are tuples, examining structure...")
            # Print the first tuple to see its structure
            print(f"DEBUG: First feature tuple: {dataset.feature_names[0]}")
            
            # Get unique tract names from the tuples
            for feature_tuple in dataset.feature_names:
                # Most likely the tract name is the second element
                if len(feature_tuple) >= 2:
                    # Combine modality and tract name as in the example
                    tract_name = feature_tuple[0] + feature_tuple[1]
                    if tract_name not in tract_names:
                        tract_names.append(tract_name)
                        tract_data_types[tract_name] = feature_tuple[0]  # Store the modality
        else:
            print("DEBUG: Feature names are strings, using split method...")
            for feature_name in dataset.feature_names:
                parts = feature_name.split('_')
                # Extract tract name (format is typically "dki_fa_tract_name_node0")
                if len(parts) >= 4 and parts[2] == 'node0':
                    modality = f"{parts[0]}_{parts[1]}"  # dki_fa or dki_md
                    tract_name = modality + parts[3]     # combine modality with tract name
                    if tract_name not in tract_names:
                        tract_names.append(tract_name)
                        tract_data_types[tract_name] = modality
    
    # After attempted tract name extraction, check if we got any names
    if len(tract_names) > 0:
        print(f"DEBUG: Found {len(tract_names)} tract names: {tract_names}")
        # Save tract names to file for later reference
        with open(os.path.join(args.output_dir, "tract_names.json"), "w") as f:
            json.dump(tract_names, f, indent=2)
        
        # Also save tract data types for reference
        with open(os.path.join(args.output_dir, "tract_data_types.json"), "w") as f:
            json.dump(tract_data_types, f, indent=2)
    else:
        print("WARNING: Could not extract tract names from feature names")
        print("DEBUG: Creating default tract names")
        # Create default tract names - with FA/MD prefixes if using both
        if args.use_both_fa_md:
            tract_names = []
            tract_data_types = {}
            for i in range(24):
                # Create FA tracts
                fa_tract = f"dki_fatract_{i}"
                tract_names.append(fa_tract)
                tract_data_types[fa_tract] = "dki_fa"
                
                # Create MD tracts
                md_tract = f"dki_mdtract_{i}"
                tract_names.append(md_tract)
                tract_data_types[md_tract] = "dki_md"
        else:
            # Just create generic tract names if only using FA
            tract_names = [f"dki_fatract_{i}" for i in range(48)]
            tract_data_types = {name: "dki_fa" for name in tract_names}
            
        print(f"DEBUG: Created {len(tract_names)} default tract names")
        
        # Save these default names for consistency
        with open(os.path.join(args.output_dir, "tract_names.json"), "w") as f:
            json.dump(tract_names, f, indent=2)
            
        # Save the data types
        with open(os.path.join(args.output_dir, "tract_data_types.json"), "w") as f:
            json.dump(tract_data_types, f, indent=2)
    
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
        f.write("tract_idx,tract_name,modality,stage1_best_r2,stage1_best_mae,stage1_best_site_acc," +
                "stage2_best_r2,stage2_best_mae,stage2_best_site_acc,stage2_worst_site_acc," +
                "training_time_minutes,status\n")
    
    for tract_idx in range(args.start_tract, args.end_tract + 1):
        if tract_idx < len(tract_names):
            tract_name = tract_names[tract_idx]
            
            # Determine modality from tract name
            if "dki_fa" in tract_name:
                modality = "dki_fa"
            elif "dki_md" in tract_name:
                modality = "dki_md"
            else:
                modality = "unknown"
            
            print(f"\nProcessing tract {tract_idx}/{args.end_tract}: {tract_name}")
        else:
            # Create default name if index is out of range
            if args.use_both_fa_md:
                # Determine if this is an FA or MD tract based on index
                if tract_idx % 2 == 0:
                    modality = "dki_fa"
                    tract_name = f"dki_fatract_{tract_idx//2}"
                else:
                    modality = "dki_md"
                    tract_name = f"dki_mdtract_{tract_idx//2}"
            else:
                modality = "dki_fa"
                tract_name = f"dki_fatract_{tract_idx}"
            
            print(f"\nProcessing tract {tract_idx}/{args.end_tract}: {tract_name}")
        
        # Run the experiment
        result = run_tract_experiment(tract_idx, tract_name, args.output_dir)
        
        # Make sure modality is in the result
        if "modality" not in result:
            result["modality"] = modality
        
        all_results.append(result)
        
        # Update summary file
        with open(summary_file, 'a') as f:
            if "error" in result:
                f.write(f"{tract_idx},{tract_name},{modality},0,0,0,0,0,0,0,0,error\n")
            else:
                # Extract metrics for stage 1 (independent training)
                # These would come from the individual predictors
                stage1_r2 = 0
                stage1_mae = 0
                stage1_site_acc = 0
                
                # Check for age predictor metrics first
                if 'age_predictor' in result and 'val_r2_epoch' in result['age_predictor']:
                    stage1_r2 = max(result['age_predictor']['val_r2_epoch']) if result['age_predictor']['val_r2_epoch'] else 0
                    stage1_mae = min(result['age_predictor']['val_loss_epoch']) if result['age_predictor']['val_loss_epoch'] else 0
                
                # Check for site predictor metrics
                if 'site_predictor' in result and 'val_acc_epoch' in result['site_predictor']:
                    stage1_site_acc = max(result['site_predictor']['val_acc_epoch']) if result['site_predictor']['val_acc_epoch'] else 0
                
                # Extract metrics for stage 2 (adversarial training)
                # These would come from the combined model
                stage2_r2 = 0
                stage2_mae = 0
                stage2_site_acc = 0
                stage2_worst_site_acc = 0
                
                if 'combined' in result:
                    if 'val_age_r2_epoch' in result['combined']:
                        stage2_r2 = max(result['combined']['val_age_r2_epoch']) if result['combined']['val_age_r2_epoch'] else 0
                    
                    if 'val_age_mae_epoch' in result['combined']:
                        stage2_mae = min(result['combined']['val_age_mae_epoch']) if result['combined']['val_age_mae_epoch'] else 0
                    
                    if 'val_site_acc_epoch' in result['combined']:
                        site_acc_values = result['combined']['val_site_acc_epoch']
                        if site_acc_values:
                            stage2_site_acc = max(site_acc_values)
                            stage2_worst_site_acc = min(site_acc_values)
                
                training_time_mins = result.get('training_time', 0) / 60
                
                f.write(f"{tract_idx},{tract_name},{modality}," +
                        f"{stage1_r2},{stage1_mae},{stage1_site_acc}," +
                        f"{stage2_r2},{stage2_mae},{stage2_site_acc},{stage2_worst_site_acc}," +
                        f"{training_time_mins:.2f},completed\n")
    
    # Create a final summary with tract ranking
    print("\nAnalyzing results...")
    valid_results = [r for r in all_results if "error" not in r]
    
    if valid_results:
        # Create a comprehensive DataFrame with all metrics
        metrics_df = pd.DataFrame([
            {
                'tract_idx': r['tract_idx'],
                'tract_name': r['tract_name'],
                'modality': r.get('modality', 'unknown'),
                'base_tract_name': r.get('base_tract_name', r['tract_name']),
                'best_val_r2': r.get('best_val_r2', 0),
                'best_val_mae': r.get('best_val_mae', 0),
                'best_site_acc': r.get('best_site_acc', 0),
                'training_time_minutes': r.get('training_time', 0) / 60
            } 
            for r in valid_results
        ])
        
        # Save complete metrics table
        metrics_df.to_csv(os.path.join(args.output_dir, "all_tract_metrics.csv"), index=False)
        
        # Sort tracts by R² (descending) - overall ranking
        r2_ranking = metrics_df.sort_values('best_val_r2', ascending=False).reset_index(drop=True)
        r2_ranking.to_csv(os.path.join(args.output_dir, "tract_ranking_by_r2.csv"), index=False)
        
        # Sort tracts by MAE (ascending) - overall ranking
        mae_ranking = metrics_df.sort_values('best_val_mae', ascending=True).reset_index(drop=True)
        mae_ranking.to_csv(os.path.join(args.output_dir, "tract_ranking_by_mae.csv"), index=False)
        
        # Create separate rankings for each modality if both types are present
        fa_metrics = metrics_df[metrics_df['modality'] == 'dki_fa']
        md_metrics = metrics_df[metrics_df['modality'] == 'dki_md']
        
        if not fa_metrics.empty:
            # FA Rankings
            fa_r2_ranking = fa_metrics.sort_values('best_val_r2', ascending=False).reset_index(drop=True)
            fa_r2_ranking.to_csv(os.path.join(args.output_dir, "fa_tract_ranking_by_r2.csv"), index=False)
            
            fa_mae_ranking = fa_metrics.sort_values('best_val_mae', ascending=True).reset_index(drop=True)
            fa_mae_ranking.to_csv(os.path.join(args.output_dir, "fa_tract_ranking_by_mae.csv"), index=False)
            
            # Print top FA tracts
            print("\nTop 5 FA tracts by R²:")
            for i, row in fa_r2_ranking.head(5).iterrows():
                print(f"{i+1}. {row['base_tract_name']} (idx: {row['tract_idx']}): " + 
                      f"R²={row['best_val_r2']:.4f}, MAE={row['best_val_mae']:.4f} years")
        
        if not md_metrics.empty:
            # MD Rankings
            md_r2_ranking = md_metrics.sort_values('best_val_r2', ascending=False).reset_index(drop=True)
            md_r2_ranking.to_csv(os.path.join(args.output_dir, "md_tract_ranking_by_r2.csv"), index=False)
            
            md_mae_ranking = md_metrics.sort_values('best_val_mae', ascending=True).reset_index(drop=True)
            md_mae_ranking.to_csv(os.path.join(args.output_dir, "md_tract_ranking_by_mae.csv"), index=False)
            
            # Print top MD tracts
            print("\nTop 5 MD tracts by R²:")
            for i, row in md_r2_ranking.head(5).iterrows():
                print(f"{i+1}. {row['base_tract_name']} (idx: {row['tract_idx']}): " + 
                      f"R²={row['best_val_r2']:.4f}, MAE={row['best_val_mae']:.4f} years")
        
        # Create a consolidated summary of adversarial training results
        adversarial_summaries = []
        for tract_idx in range(args.start_tract, args.end_tract + 1):
            # Look for files with modality in the directory name
            for modality in ['dki_fa', 'dki_md', 'unknown']:
                adversarial_file = os.path.join(args.output_dir, f"tract_{tract_idx}_{modality}", "adversarial_training_summary.csv")
                if os.path.exists(adversarial_file):
                    try:
                        adv_data = pd.read_csv(adversarial_file)
                        if not adv_data.empty:
                            # Add modality information
                            adv_row = adv_data.iloc[0].to_dict()
                            adv_row['modality'] = modality
                            adversarial_summaries.append(adv_row)
                    except Exception as e:
                        print(f"Error reading adversarial summary for tract {tract_idx} ({modality}): {e}")
                    break  # Found file for this tract, no need to check other modalities
        
        if adversarial_summaries:
            adv_df = pd.DataFrame(adversarial_summaries)
            # Save the consolidated adversarial training summary
            adv_df.to_csv(os.path.join(args.output_dir, "all_tracts_adversarial_summary.csv"), index=False)
            
            # Create rankings based on adversarial training
            adv_r2_ranking = adv_df.sort_values('best_val_r2', ascending=False).reset_index(drop=True)
            adv_r2_ranking.to_csv(os.path.join(args.output_dir, "adversarial_tract_ranking_by_r2.csv"), index=False)
            
            adv_mae_ranking = adv_df.sort_values('best_val_mae', ascending=True).reset_index(drop=True)
            adv_mae_ranking.to_csv(os.path.join(args.output_dir, "adversarial_tract_ranking_by_mae.csv"), index=False)
            
            # Create separate rankings for each modality
            fa_adv = adv_df[adv_df['modality'] == 'dki_fa']
            md_adv = adv_df[adv_df['modality'] == 'dki_md']
            
            if not fa_adv.empty:
                fa_adv_r2 = fa_adv.sort_values('best_val_r2', ascending=False).reset_index(drop=True)
                fa_adv_r2.to_csv(os.path.join(args.output_dir, "fa_adversarial_ranking_by_r2.csv"), index=False)
                
                fa_adv_mae = fa_adv.sort_values('best_val_mae', ascending=True).reset_index(drop=True)
                fa_adv_mae.to_csv(os.path.join(args.output_dir, "fa_adversarial_ranking_by_mae.csv"), index=False)
            
            if not md_adv.empty:
                md_adv_r2 = md_adv.sort_values('best_val_r2', ascending=False).reset_index(drop=True)
                md_adv_r2.to_csv(os.path.join(args.output_dir, "md_adversarial_ranking_by_r2.csv"), index=False)
                
                md_adv_mae = md_adv.sort_values('best_val_mae', ascending=True).reset_index(drop=True)
                md_adv_mae.to_csv(os.path.join(args.output_dir, "md_adversarial_ranking_by_mae.csv"), index=False)
            
            # Create a visualization of adversarial training effectiveness
            plt.figure(figsize=(14, 10))
            
            # Top 20 tracts by R² from adversarial training
            top20_adv_r2 = adv_r2_ranking.head(20).copy()
            
            # Create x positions for bars
            x = np.arange(len(top20_adv_r2))
            width = 0.35
            
            # Normalize MAE to 0-1 scale for better visualization alongside R²
            max_mae = top20_adv_r2['best_val_mae'].max()
            min_mae = top20_adv_r2['best_val_mae'].min()
            normalized_mae = 1 - ((top20_adv_r2['best_val_mae'] - min_mae) / (max_mae - min_mae))
            
            # Create bar plot
            fig, ax = plt.subplots(figsize=(14, 8))
            rects1 = ax.bar(x - width/2, top20_adv_r2['best_val_r2'], width, label='R² (higher is better)')
            rects2 = ax.bar(x + width/2, normalized_mae, width, 
                            label='Normalized MAE (higher means lower error)')
            
            # Color bars by modality
            colors = {'dki_fa': 'skyblue', 'dki_md': 'salmon', 'unknown': 'gray'}
            for i, (_, row) in enumerate(top20_adv_r2.iterrows()):
                if 'modality' in row:
                    modality = row['modality']
                    rects1[i].set_color(colors.get(modality, 'gray'))
            
            ax.set_ylabel('Score')
            ax.set_title('Top 20 Tracts from Adversarial Training: R² and Normalized MAE')
            ax.set_xticks(x)
            ax.set_xticklabels(top20_adv_r2['tract_name'], rotation=45, ha='right')
            
            # Add legend for modality colors
            from matplotlib.patches import Patch
            legend_elements = []
            if 'dki_fa' in top20_adv_r2['modality'].values:
                legend_elements.append(Patch(facecolor=colors['dki_fa'], label='FA'))
            if 'dki_md' in top20_adv_r2['modality'].values:
                legend_elements.append(Patch(facecolor=colors['dki_md'], label='MD'))
            if 'unknown' in top20_adv_r2['modality'].values:
                legend_elements.append(Patch(facecolor=colors['unknown'], label='Unknown'))
            
            # Add two legends: one for metrics, one for modality
            first_legend = ax.legend(title="Metrics", loc='upper left')
            ax.add_artist(first_legend)
            ax.legend(handles=legend_elements, title="Modality", loc='upper right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "adversarial_top_tracts_metrics.png"))
            
            # Print top 5 tracts from adversarial training
            print("\nTop 5 tracts from adversarial training (by R²):")
            for i, row in adv_r2_ranking.head(5).iterrows():
                modality = row['modality'] if 'modality' in row else 'unknown'
                # Clean up modality display
                display_modality = 'FA' if modality == 'dki_fa' else 'MD' if modality == 'dki_md' else modality
                print(f"{i+1}. {display_modality} {row['tract_name']} (idx: {row['tract_idx']}): " + 
                      f"R²={row['best_val_r2']:.4f}, MAE={row['best_val_mae']:.4f} years")
                      
        # Compare FA and MD performance if both are present
        if not fa_metrics.empty and not md_metrics.empty:
            # Try to match tracts by their base name
            fa_base_tracts = set([name.replace('dki_fa', '') for name in fa_metrics['tract_name']])
            md_base_tracts = set([name.replace('dki_md', '') for name in md_metrics['tract_name']])
            
            # Find common tracts
            common_base_tracts = fa_base_tracts.intersection(md_base_tracts)
            
            if common_base_tracts:
                # Create comparison data
                comparison_data = []
                for base_tract in common_base_tracts:
                    fa_name = 'dki_fa' + base_tract
                    md_name = 'dki_md' + base_tract
                    
                    fa_row = fa_metrics[fa_metrics['tract_name'] == fa_name].iloc[0]
                    md_row = md_metrics[md_metrics['tract_name'] == md_name].iloc[0]
                    
                    # Calculate differences
                    r2_diff = fa_row['best_val_r2'] - md_row['best_val_r2']
                    mae_diff = fa_row['best_val_mae'] - md_row['best_val_mae']
                    
                    comparison_data.append({
                        'base_tract': base_tract,
                        'fa_tract_idx': fa_row['tract_idx'],
                        'md_tract_idx': md_row['tract_idx'],
                        'fa_r2': fa_row['best_val_r2'],
                        'md_r2': md_row['best_val_r2'],
                        'r2_diff': r2_diff,  # Positive means FA is better
                        'fa_mae': fa_row['best_val_mae'],
                        'md_mae': md_row['best_val_mae'],
                        'mae_diff': mae_diff,  # Negative means FA is better (lower MAE)
                        'better_r2': 'FA' if r2_diff > 0 else 'MD',
                        'better_mae': 'FA' if mae_diff < 0 else 'MD'
                    })
                
                # Create comparison DataFrame and save
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df.to_csv(os.path.join(args.output_dir, "fa_md_comparison.csv"), index=False)
                
                # Print comparison summary
                fa_better_r2 = sum(comparison_df['r2_diff'] > 0)
                md_better_r2 = sum(comparison_df['r2_diff'] < 0)
                fa_better_mae = sum(comparison_df['mae_diff'] < 0)
                md_better_mae = sum(comparison_df['mae_diff'] > 0)
                
                print(f"\nComparison of FA vs MD tracts ({len(common_base_tracts)} matched tracts):")
                print(f"  Better R² with FA: {fa_better_r2} tracts")
                print(f"  Better R² with MD: {md_better_r2} tracts")
                print(f"  Better MAE with FA: {fa_better_mae} tracts")
                print(f"  Better MAE with MD: {md_better_mae} tracts")
                
                # Print top 5 tracts where FA is significantly better than MD by R²
                print("\nTop 5 tracts where FA outperforms MD (by R² difference):")
                for i, row in comparison_df.sort_values('r2_diff', ascending=False).head(5).iterrows():
                    print(f"{i+1}. {row['base_tract']}: " + 
                          f"FA R²={row['fa_r2']:.4f} vs MD R²={row['md_r2']:.4f} " + 
                          f"(diff: {row['r2_diff']:.4f})")
                
                # Print top 5 tracts where MD is significantly better than FA by R²
                print("\nTop 5 tracts where MD outperforms FA (by R² difference):")
                for i, row in comparison_df.sort_values('r2_diff', ascending=True).head(5).iterrows():
                    print(f"{i+1}. {row['base_tract']}: " + 
                          f"MD R²={row['md_r2']:.4f} vs FA R²={row['fa_r2']:.4f} " + 
                          f"(diff: {-row['r2_diff']:.4f})")
                
                # Create visualization of FA vs MD comparison for R²
                plt.figure(figsize=(14, 8))
                # Sort by absolute R² difference
                comparison_df['abs_r2_diff'] = comparison_df['r2_diff'].abs()
                top_diff = comparison_df.sort_values('abs_r2_diff', ascending=False).head(15)
                
                x = np.arange(len(top_diff))
                width = 0.35
                
                fig, ax = plt.subplots(figsize=(14, 8))
                fa_bars = ax.bar(x - width/2, top_diff['fa_r2'], width, label='FA R²', color='skyblue')
                md_bars = ax.bar(x + width/2, top_diff['md_r2'], width, label='MD R²', color='salmon')
                
                # Add markers for better modality
                for i, row in enumerate(top_diff.iterrows()[1]):
                    better = row['better_r2']
                    if better == 'FA':
                        ax.annotate('FA', xy=(i - width/2, row['fa_r2']), xytext=(0, 5), 
                                   textcoords='offset points', ha='center', va='bottom',
                                   weight='bold', color='green')
                    else:
                        ax.annotate('MD', xy=(i + width/2, row['md_r2']), xytext=(0, 5), 
                                   textcoords='offset points', ha='center', va='bottom',
                                   weight='bold', color='green')
                
                ax.set_ylabel('Validation R²')
                ax.set_title('FA vs MD: R² Comparison for Top 15 Tracts with Largest Difference')
                ax.set_xticks(x)
                ax.set_xticklabels(top_diff['base_tract'], rotation=45, ha='right')
                ax.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, "fa_md_r2_comparison.png"))
                
                # Create visualization of FA vs MD comparison for MAE
                plt.figure(figsize=(14, 8))
                # Sort by absolute MAE difference
                comparison_df['abs_mae_diff'] = comparison_df['mae_diff'].abs()
                top_mae_diff = comparison_df.sort_values('abs_mae_diff', ascending=False).head(15)
                
                x = np.arange(len(top_mae_diff))
                
                fig, ax = plt.subplots(figsize=(14, 8))
                fa_bars = ax.bar(x - width/2, top_mae_diff['fa_mae'], width, label='FA MAE', color='skyblue')
                md_bars = ax.bar(x + width/2, top_mae_diff['md_mae'], width, label='MD MAE', color='salmon')
                
                # Add markers for better modality (lower MAE is better)
                for i, row in enumerate(top_mae_diff.iterrows()[1]):
                    better = row['better_mae']
                    if better == 'FA':
                        ax.annotate('FA', xy=(i - width/2, row['fa_mae']), xytext=(0, -15), 
                                   textcoords='offset points', ha='center', va='top',
                                   weight='bold', color='green')
                    else:
                        ax.annotate('MD', xy=(i + width/2, row['md_mae']), xytext=(0, -15), 
                                   textcoords='offset points', ha='center', va='top',
                                   weight='bold', color='green')
                
                ax.set_ylabel('Validation MAE (years)')
                ax.set_title('FA vs MD: MAE Comparison for Top 15 Tracts with Largest Difference (lower is better)')
                ax.set_xticks(x)
                ax.set_xticklabels(top_mae_diff['base_tract'], rotation=45, ha='right')
                ax.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, "fa_md_mae_comparison.png"))
        
        # Print top 5 tracts overall
        print("\nTop 5 tracts overall by R²:")
        for i, row in r2_ranking.head(5).iterrows():
            # Clean up display
            modality = row['modality']
            display_modality = 'FA' if modality == 'dki_fa' else 'MD' if modality == 'dki_md' else modality
            base_name = row['base_tract_name'] if 'base_tract_name' in row else row['tract_name'].replace('dki_fa', '').replace('dki_md', '')
            
            print(f"{i+1}. {display_modality} {base_name} (idx: {row['tract_idx']}): " + 
                  f"R²={row['best_val_r2']:.4f}, MAE={row['best_val_mae']:.4f} years")
    
    print("\nExperiment completed!")
    
except Exception as e:
    print(f"ERROR: {str(e)}")
    import traceback
    print(traceback.format_exc())
    sys.exit(1) 