import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from afqinsight import AFQDataset
from afqinsight.nn.utils import prep_pytorch_data
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
from sklearn.decomposition import PCA
import afqinsight.augmentation as aug
from afqinsight.nn.pt_models import Conv1DAutoencoder
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

# Force immediate output flushing
print("DEBUG: Script starting")
print(f"DEBUG: Arguments: {sys.argv}")
print(f"DEBUG: Current working directory: {os.getcwd()}")
sys.stdout.flush()

# Adjust path as needed
sys.path.insert(1, '/mmfs1/gscratch/nrdg/samchou/AFQ-Insight-Autoencoder-Experiments/Experiment_Utils')
# sys.path.insert(1, '/Users/samchou/AFQ-Insight-Autoencoder-Experiments/AFQ-Insight-Autoencoder-Experiments/Experiment_Utils')
# Import necessary functions, including the new one
try:
    print("DEBUG: Importing utility functions")
    sys.stdout.flush()
    from utils import select_device, kl_divergence_loss,prep_fa_flattned_data, prep_fa_flattened_remapped_data, train_vae_age_site_staged
    from models import Conv1DVariationalAutoencoder_fa, AgePredictorCNN, SitePredictorCNN, CombinedVAE_Predictors
    print("DEBUG: Successfully imported utility functions")
    sys.stdout.flush()
except Exception as e:
    print(f"ERROR importing utility functions: {str(e)}")
    import traceback
    print(traceback.format_exc())
    sys.exit(1)

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
    
    torch_dataset, train_loader, test_loader, val_loader = prep_pytorch_data(dataset,batch_size=128)  
    print("DEBUG: Prepared initial PyTorch data loaders")
    sys.stdout.flush()

    print("DEBUG: Getting age and site indices")
    sys.stdout.flush()
    age_idx = dataset.target_cols.index('age')
    site_idx = dataset.target_cols.index('scan_site_id')
    print(f"DEBUG: Found age_idx={age_idx}, site_idx={site_idx}")
    sys.stdout.flush()

    print("DEBUG: Preparing flattened data")
    sys.stdout.flush()
    torch_dataset, all_tracts_train_loader, all_tracts_test_loader, all_tracts_val_loader = prep_fa_flattned_data(dataset, batch_size=128)
    print("DEBUG: Prepared flattened data")
    sys.stdout.flush()

    print("DEBUG: Preparing remapped data")
    sys.stdout.flush()
    print("Preparing initial PyTorch data loaders...")
    try:
        # Assuming prep_pytorch_data returns torch_dataset, train_loader, test_loader, val_loader
        # If it returns datasets, create loaders here.
        # Adapt this call based on the actual signature and return values of your prep_pytorch_data
        prep_output = prep_fa_flattened_remapped_data(dataset, batch_size=128)
        if len(prep_output) == 4:
            _, train_loader_raw, test_loader_raw, val_loader_raw = prep_output
        else:
            raise ValueError(f"Expected 4 return values from prep_pytorch_data, got {len(prep_output)}")

        print("Initial data loaders prepared.")
        sys.stdout.flush()
    except Exception as e:
         print(f"Error calling prep_pytorch_data: {e}")
         print("Ensure the function exists and returns DataLoaders or required components.")
         import traceback
         print(traceback.format_exc())
         sys.exit(1)

    print("DEBUG: Examining data structure")
    sys.stdout.flush()
    # Examine first batch of data
    try:
        x_sample, labels_sample = next(iter(train_loader_raw))
        print(f"DEBUG: First batch x shape: {x_sample.shape}, labels shape: {labels_sample.shape}")
        print(f"DEBUG: First example label: {labels_sample[0]}")
        sys.stdout.flush()
    except Exception as e:
        print(f"ERROR examining first batch: {str(e)}")
        import traceback
        print(traceback.format_exc())
        sys.stdout.flush()

    if 'x_batch' in locals() and x_batch is not None:
        input_channels = x_batch.shape[1]
        sequence_length = x_batch.shape[2]
        print(f"Detected input shape: channels={input_channels}, sequence_length={sequence_length}")
    else:
        print("Warning: Could not get sample batch to determine input shape.")
        # Set defaults or exit if necessary
        input_channels = 1 # Set manually if needed
        sequence_length = 50 # Set manually if needed (MUST MATCH VAE DECODER OUTPUT)
        print(f"Using default/manual input shape: channels={input_channels}, sequence_length={sequence_length}")
    sys.stdout.flush()

    # ... existing code ...

    # ================================================================================
    # STAGED TRAINING EXPERIMENT
    # ================================================================================
    print("\n\n" + "="*80)
    print("RUNNING STAGED TRAINING EXPERIMENT")
    print("="*80 + "\n")
    sys.stdout.flush()

    # Set parameters for the staged experiment
    latent_dim = 64  # Choose the larger latent dim
    dropout = 0.0  # VAE dropout
    age_dropout = 0.2
    site_dropout = 0.2
    w_recon = 1.0
    w_kl = 0.1
    w_age = 10.0  # Higher weight for age prediction
    w_site = 5.0  # Higher weight for site adversarial training
    
    print("DEBUG: Creating models")
    sys.stdout.flush()

    # Create models
    try:
        vae = Conv1DVariationalAutoencoder_fa(latent_dims=latent_dim, dropout=dropout)
        
        # AgePredictorCNN now accepts sex information to improve age prediction
        age_predictor = AgePredictorCNN(input_channels=input_channels, 
                                       sequence_length=sequence_length, 
                                       dropout=age_dropout)
                                       
        site_predictor = SitePredictorCNN(num_sites=4, 
                                         input_channels=input_channels, 
                                         sequence_length=sequence_length, 
                                         dropout=site_dropout)
        print("DEBUG: Successfully created models")
        sys.stdout.flush()
    except Exception as e:
        print(f"ERROR creating models: {str(e)}")
        import traceback
        print(traceback.format_exc())
        sys.stdout.flush()
        raise

    # Define directory for saving staged models
    staged_save_directory = "staged_experiment_results"
    os.makedirs(staged_save_directory, exist_ok=True)
    print(f"Staged experiment results will be saved in: {staged_save_directory}")
    sys.stdout.flush()


    print("DEBUG: Starting staged training")
    sys.stdout.flush()
    try:   
        staged_results = train_vae_age_site_staged(
            vae_model=vae,
            age_predictor=age_predictor,
            site_predictor=site_predictor,
            train_data=train_loader_raw,
            val_data=val_loader_raw,
            epochs_stage1=500,  # For individual training
            epochs_stage2=1000,  # For adversarial training
            lr=0.001,
            device=device,
            max_grad_norm=1.0,
            w_recon=w_recon,
            w_kl=w_kl,
            w_age=w_age,
            w_site=w_site,
            kl_annealing_start_epoch=200,
            kl_annealing_duration=200,
            kl_annealing_start=0.001,
            grl_alpha_start=0.0,
            grl_alpha_end=2.5,
            grl_alpha_epochs=150,
            save_dir=staged_save_directory,
            val_metric_to_monitor="val_age_mae"
        )
        print("DEBUG: Staged training completed successfully")
        sys.stdout.flush()
    except Exception as e:
        print(f"ERROR during staged training: {str(e)}")
        import traceback
        print(traceback.format_exc()) 
        sys.stdout.flush()
        sys.exit(1)

    print("DEBUG: Processing results")
    sys.stdout.flush()
    def process_metrics(metrics_dict, keys_to_convert):
        processed_results = {}
        for key in keys_to_convert:
            metric_list = metrics_dict.get(key, [])
            new_list = []
            if isinstance(metric_list, (list, tuple)):
                for val in metric_list:
                    if isinstance(val, torch.Tensor):
                        new_list.append(float(val.cpu().item()))
                    elif isinstance(val, (int, float, np.number)):
                        new_list.append(float(val))
                    else:
                        new_list.append(float('nan'))
            processed_results[key] = new_list
        return processed_results

    # Process and save VAE Stage 1 results
    if staged_results and "vae" in staged_results:
        vae_results = staged_results["vae"]
        
        # Keys to convert for VAE metrics
        vae_keys = [
            "train_loss_epoch", "val_loss_epoch",
            "train_recon_loss_epoch", "val_recon_loss_epoch",
            "train_kl_loss_epoch", "val_kl_loss_epoch",
            "current_beta_epoch", "current_lr_epoch"
        ]
        
        vae_processed = process_metrics(vae_results, vae_keys)
        
        # Create DataFrame for VAE metrics
        vae_epochs = len(vae_processed.get("train_loss_epoch", []))
        if vae_epochs > 0:
            vae_df_data = {"epoch": range(1, vae_epochs + 1)}
            for k in vae_keys:
                col_name = k.replace('_epoch', '')
                vae_df_data[col_name] = vae_processed.get(k, [float('nan')] * vae_epochs)
            
            vae_df = pd.DataFrame(vae_df_data)
            vae_metrics_file = os.path.join(staged_save_directory, "vae_metrics.csv")
            vae_df.to_csv(vae_metrics_file, index=False)
            print(f"Saved VAE training metrics to {vae_metrics_file}")
        

    # Process and save Age Predictor Stage 1 results
    if staged_results and "age_predictor" in staged_results:
        age_results = staged_results["age_predictor"]
        
        # Keys to convert for Age Predictor metrics
        age_keys = ["train_loss_epoch", "val_loss_epoch", "current_lr_epoch"]
        
        age_processed = process_metrics(age_results, age_keys)
        
        # Create DataFrame for Age Predictor metrics
        age_epochs = len(age_processed.get("train_loss_epoch", []))
        if age_epochs > 0:
            age_df_data = {"epoch": range(1, age_epochs + 1)}
            for k in age_keys:
                col_name = k.replace('_epoch', '')
                age_df_data[col_name] = age_processed.get(k, [float('nan')] * age_epochs)
            
            age_df = pd.DataFrame(age_df_data)
            age_metrics_file = os.path.join(staged_save_directory, "age_predictor_metrics.csv")
            age_df.to_csv(age_metrics_file, index=False)
            print(f"Saved Age Predictor training metrics to {age_metrics_file}")

    # Process and save Site Predictor Stage 1 results
    if staged_results and "site_predictor" in staged_results:
        site_results = staged_results["site_predictor"]
        
        # Keys to convert for Site Predictor metrics
        site_keys = ["train_loss_epoch", "val_loss_epoch", "train_acc_epoch", "val_acc_epoch", "current_lr_epoch"]
        
        site_processed = process_metrics(site_results, site_keys)
        
        # Create DataFrame for Site Predictor metrics
        site_epochs = len(site_processed.get("train_loss_epoch", []))
        if site_epochs > 0:
            site_df_data = {"epoch": range(1, site_epochs + 1)}
            for k in site_keys:
                col_name = k.replace('_epoch', '')
                site_df_data[col_name] = site_processed.get(k, [float('nan')] * site_epochs)
            
            site_df = pd.DataFrame(site_df_data)
            site_metrics_file = os.path.join(staged_save_directory, "site_predictor_metrics.csv")
            site_df.to_csv(site_metrics_file, index=False)
            print(f"Saved Site Predictor training metrics to {site_metrics_file}")

    # Process and save combined stage results
    if staged_results and "combined" in staged_results:
        combined_results = staged_results["combined"]
        
        # Convert metrics to CPU floats
        keys_to_convert = [
            "train_loss_epoch", "val_loss_epoch", 
            "train_recon_loss_epoch", "val_recon_loss_epoch",
            "train_kl_loss_epoch", "val_kl_loss_epoch", 
            "train_age_loss_epoch", "val_age_loss_epoch",
            "train_site_loss_epoch", "val_site_loss_epoch", 
            "train_age_mae_epoch", "val_age_mae_epoch",
            "train_site_acc_epoch", "val_site_acc_epoch", 
            "current_beta_epoch", "current_grl_alpha_epoch",
            "current_lr_epoch"
        ]
        
        processed_results = process_metrics(combined_results, keys_to_convert)
        
        # Create DataFrame
        num_epochs = len(processed_results.get("train_loss_epoch", []))
        if num_epochs > 0:
            df_data = {"epoch": range(1, num_epochs + 1)}
            for k in keys_to_convert:
                col_name = k.replace('_epoch', '')
                metric_data = processed_results.get(k, [])
                if len(metric_data) != num_epochs:
                    metric_data = [float('nan')] * num_epochs
                df_data[col_name] = metric_data
            
            df_epochs = pd.DataFrame(df_data)
            metrics_file = os.path.join(staged_save_directory, "staged_combined_metrics.csv")
            df_epochs.to_csv(metrics_file, index=False)
            
            # Save summary metrics
            best_mae_key = "best_val_age_mae"
            best_mae = combined_results.get(best_mae_key, float('nan'))
            if isinstance(best_mae, torch.Tensor):
                best_mae = float(best_mae.cpu().item())
            
            df_summary = pd.DataFrame([{
                best_mae_key: best_mae,
                "best_epoch": combined_results.get("best_epoch", float('nan')),
                "model_path": combined_results.get("model_path", "N/A")
            }])
            summary_file = os.path.join(staged_save_directory, "staged_combined_summary.csv")
            df_summary.to_csv(summary_file, index=False)
            
            print(f"Saved staged training metrics to {metrics_file}")
            print(f"Saved staged training summary to {summary_file}")

    print("\n" + "="*80)
    print("Staged training experiment complete!")
    print("="*80 + "\n")
except Exception as e:
    print(f"UNHANDLED ERROR: {str(e)}")
    import traceback
    print(traceback.format_exc())
    sys.stdout.flush()
    sys.exit(1)
