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
import pandas as pd
# from afqinsight.nn.pt_models import Conv1DVariationalAutoencoder
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Function to create and save confusion matrices
def create_confusion_matrices(save_dir, site_names=None):
    """
    Create and save confusion matrices from saved site prediction data.
    
    Parameters
    ----------
    save_dir : str
        Directory where site prediction data is saved
    site_names : list, optional
        List of site names for axis labels
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    # Define directory with saved predictions
    data_dir = os.path.join(save_dir, 'site_predictions')
    
    if not os.path.exists(data_dir):
        print(f"No site prediction data found in {data_dir}")
        return
        
    # Create directory for confusion matrix plots
    plot_dir = os.path.join(save_dir, 'confusion_matrices')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Get all saved files
    true_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy') and 'true_sites' in f])
    
    # Group files by prefix and epoch
    file_groups = {}
    for f in true_files:
        # Extract prefix and epoch from filename
        parts = f.split('_')
        prefix = parts[0]
        epoch = int(parts[-1].replace('.npy', ''))
        
        # Add to groups
        if (prefix, epoch) not in file_groups:
            file_groups[(prefix, epoch)] = {}
        
        file_groups[(prefix, epoch)]['true'] = os.path.join(data_dir, f)
        # Find corresponding predictions file
        pred_file = f.replace('true_sites', 'pred_sites')
        if os.path.exists(os.path.join(data_dir, pred_file)):
            file_groups[(prefix, epoch)]['pred'] = os.path.join(data_dir, pred_file)
    
    print(f"Found {len(file_groups)} pairs of true/predicted site data")
    
    # Create confusion matrices
    for (prefix, epoch), files in file_groups.items():
        if 'true' in files and 'pred' in files:
            # Load data
            true_sites = np.load(files['true'])
            pred_sites = np.load(files['pred'])
            
            # Compute confusion matrix
            cm = confusion_matrix(true_sites, pred_sites, normalize='true')
            
            # Create display with labels if provided
            if site_names is not None:
                labels = site_names
            else:
                num_sites = len(np.unique(np.concatenate([true_sites, pred_sites])))
                labels = [f"Site {i}" for i in range(num_sites)]
                
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            
            # Create figure and plot
            fig, ax = plt.subplots(figsize=(10, 8))
            disp.plot(ax=ax, cmap='Blues', values_format='.2f')
            plt.title(f'{prefix.capitalize()} Confusion Matrix - Epoch {epoch}')
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'{prefix}_confusion_matrix_epoch_{epoch}.png'))
            plt.close()
            
            print(f"Created confusion matrix for {prefix} data at epoch {epoch}")
            
            # Also save a CSV of the confusion matrix for detailed analysis
            df = pd.DataFrame(cm, index=labels, columns=labels)
            df.to_csv(os.path.join(plot_dir, f'{prefix}_confusion_matrix_epoch_{epoch}.csv'))
            
    # Create a final summary visualization of confusion matrices over time
    create_confusion_matrix_summary(plot_dir, file_groups)
    
def create_confusion_matrix_summary(plot_dir, file_groups):
    """Create a summary visualization showing how confusion matrices evolve over time."""
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    
    # Only process validation data for the summary
    val_epochs = sorted([epoch for prefix, epoch in file_groups.keys() if prefix == 'val'])
    
    if len(val_epochs) <= 1:
        print("Not enough epochs for time series visualization")
        return
        
    # Calculate metrics over time
    epochs = []
    accuracies = []
    chance_divergences = []  # How far from chance (1/num_classes)
    
    for epoch in val_epochs:
        if ('val', epoch) in file_groups:
            files = file_groups[('val', epoch)]
            
            # Load data
            true_sites = np.load(files['true'])
            pred_sites = np.load(files['pred'])
            
            # Calculate metrics
            cm = confusion_matrix(true_sites, pred_sites, normalize='true')
            accuracy = np.trace(cm) / np.sum(cm)
            
            # Chance level is 1/num_classes
            num_classes = cm.shape[0]
            chance_level = 1.0 / num_classes
            
            # Calculate divergence from chance (positive: better than chance, negative: worse than chance)
            chance_divergence = accuracy - chance_level
            
            epochs.append(epoch)
            accuracies.append(accuracy)
            chance_divergences.append(chance_divergence)
    
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.plot(epochs, accuracies, color=color, marker='o', label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Add horizontal line for chance level
    chance_level = 1.0 / num_classes
    ax1.axhline(y=chance_level, color='gray', linestyle='--', label=f'Chance Level ({chance_level:.2f})')
    
    # Add second axis for divergence
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Divergence from Chance', color=color)
    ax2.plot(epochs, chance_divergences, color=color, marker='x', label='Divergence from Chance')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add horizontal line at zero divergence
    ax2.axhline(y=0, color='lightgray', linestyle=':')
    
    # Add combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.title('Site Classification Performance Over Training')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'site_performance_over_time.png'))
    plt.close()
    
    # Save the data as CSV for further analysis
    df = pd.DataFrame({
        'epoch': epochs,
        'accuracy': accuracies,
        'chance_level': [chance_level] * len(epochs),
        'divergence_from_chance': chance_divergences
    })
    df.to_csv(os.path.join(plot_dir, 'site_performance_over_time.csv'), index=False)
    print(f"Created site classification performance summary")

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
    from utils import select_device, kl_divergence_loss, prep_fa_dataset, train_vae_age_site_staged
    from models import Conv1DVariationalAutoencoder_fa_unflattened, AgePredictorCNN, SitePredictorCNN, Conv1DVariationalAutoencoder_fa, BaseConv1DEncode_fa_unflattened
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
    
    torch_dataset, train_loader, test_loader, val_loader = prep_pytorch_data(dataset, batch_size=128)  
    print("DEBUG: Prepared initial PyTorch data loaders")
    sys.stdout.flush()

    print("DEBUG: Getting age and site indices")
    sys.stdout.flush()
    age_idx = dataset.target_cols.index('age')
    site_idx = dataset.target_cols.index('scan_site_id')
    sex_idx = dataset.target_cols.index('sex')
    print(f"DEBUG: Found age_idx={age_idx}, site_idx={site_idx}, sex_idx={sex_idx}")
    sys.stdout.flush()

    # First, get a combined FA and MD dataset
    print("DEBUG: Preparing FA+MD dataset")
    sys.stdout.flush()
    # Create a combined FA+MD dataset
    torch_dataset, train_loader, test_loader, val_loader = prep_fa_dataset(dataset, target_labels=["dki_fa"], batch_size=128)
    
    print("DEBUG: Prepared first tract data")
    sys.stdout.flush()

    print("DEBUG: Examining data structure")
    sys.stdout.flush()
    # Examine first batch of data
    try:
        x_sample, labels_sample = next(iter(train_loader))
        print(f"DEBUG: First batch x shape: {x_sample.shape}, labels shape: {labels_sample.shape}")
        print(f"DEBUG: First example label: {labels_sample[0]}")
        print(f"DEBUG: First example data min/max: {x_sample.min().item():.4f}/{x_sample.max().item():.4f}")
        
        # Extract dimensions
        sequence_length = x_sample.shape[2]
        input_channels = x_sample.shape[1]
        print(f"DEBUG: Detected input shape: channels={input_channels}, sequence_length={sequence_length}")
        sys.stdout.flush()
    except Exception as e:
        print(f"ERROR examining first batch: {str(e)}")
        import traceback
        print(traceback.format_exc())
        sys.stdout.flush()
        # Set defaults if detection fails
        input_channels = 48
        sequence_length = 50
        print(f"Using default input shape: channels={input_channels}, sequence_length={sequence_length}")
    
    # Create a remapped dataset to handle the site mapping correctly
    print("DEBUG: Preparing remapped data")
    site_map = {0.0: 0.0, 1.0: 1.0, 3.0: 2.0, 4.0: 3.0}
    print(f"Using site map: {site_map}")

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
    train_loader_raw = RemappedDataLoader(train_loader)
    test_loader_raw = RemappedDataLoader(test_loader)
    val_loader_raw = RemappedDataLoader(val_loader)
    
    # ================================================================================
    # STAGED TRAINING EXPERIMENT
    # ================================================================================
    print("\n\n" + "="*80)
    print("RUNNING STAGED TRAINING EXPERIMENT - FA TRACTS ONLY")
    print("="*80 + "\n")
    sys.stdout.flush()

    # Set parameters for the staged experiment
    latent_dim = 64  # Choose the larger latent dim
    dropout = 0.0  # VAE dropout
    age_dropout = 0.1
    site_dropout = 0.2
    w_recon = 5.0
    w_kl = 0.0001
    w_age = 15.0  # Higher weight for age prediction
    w_site = 1.0  # Higher weight for site adversarial training
    
    print("DEBUG: Creating models")
    sys.stdout.flush()

    # Get the number of unique sites from the training data
    unique_sites = set()
    for i, (_, labels) in enumerate(train_loader_raw):
        if i < 5:  # Just check a few batches for debugging
            print(f"Batch {i} site values: {labels[:, 2].tolist()}")
        unique_sites.update(labels[:, 2].tolist())
    num_sites = len(unique_sites)
    print(f"Detected {num_sites} unique site IDs in the data: {sorted(unique_sites)}")

    # Add debug to check actual dimensions
    dummy_input = torch.zeros(1, 48, 50, device=device)
    dummy_enc = BaseConv1DEncode_fa_unflattened(num_tracts=48, latent_dims=64, dropout=0.0)
    dummy_enc.to(device)
    with torch.no_grad():
        out = dummy_enc._encode_base(dummy_input)
        print(f"DEBUG: Encoder output shape with 48x50 input: {out.shape}")
        flattened_size = out.numel()
        print(f"DEBUG: Flattened size: {flattened_size}, which is {flattened_size//64}*64")

    # Create models
    try:
        vae = Conv1DVariationalAutoencoder_fa_unflattened(num_tracts=48,latent_dims=latent_dim, dropout=dropout)
        
        age_predictor = AgePredictorCNN(input_channels=input_channels, 
                                        sequence_length=sequence_length, 
                                        dropout=age_dropout)
        
        site_predictor = SitePredictorCNN(num_sites=num_sites, 
                                         input_channels=input_channels, 
                                         sequence_length=sequence_length, 
                                         dropout=site_dropout)
        print(f"DEBUG: Successfully created models (num_sites={num_sites})")
        sys.stdout.flush()
    except Exception as e:
        print(f"ERROR creating models: {str(e)}")
        import traceback
        print(traceback.format_exc())
        sys.stdout.flush()
        raise

    # Define directory for saving staged models
    staged_save_directory = "fa_tracts_experiment_results"
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
            lr=0.0001,
            device=device,
            max_grad_norm=1.0,
            w_recon=w_recon,
            w_kl=w_kl,
            w_age=w_age,
            w_site=w_site,
            kl_annealing_start_epoch=250,
            kl_annealing_duration=500,
            kl_annealing_start=0.0001,
            grl_alpha_start=0.0,
            grl_alpha_end=7.5,
            grl_alpha_epochs=300,
            save_dir=staged_save_directory,
            val_metric_to_monitor="val_age_mae",
            save_predictions_interval=50  # Save site predictions every 50 epochs
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
        age_keys = ["train_loss_epoch", "val_loss_epoch", "train_r2_epoch", "val_r2_epoch", "current_lr_epoch"]
        
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
            "train_age_r2_epoch", "val_age_r2_epoch",
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

    # Create confusion matrices from saved prediction data
    print("\n" + "="*80)
    print("GENERATING CONFUSION MATRICES FROM SAVED DATA")
    print("="*80 + "\n")
    sys.stdout.flush()

    try:
        # Define site names if known, otherwise will use generic labels
        site_names = [f"Site {i}" for i in range(num_sites)]
        
        # Generate confusion matrices from saved data
        create_confusion_matrices(staged_save_directory, site_names)
        
        print("\nConfusion matrix generation complete!")
        sys.stdout.flush()
    except Exception as e:
        print(f"ERROR during confusion matrix generation: {str(e)}")
        import traceback
        print(traceback.format_exc())
        sys.stdout.flush()

    print("\n" + "="*80)
    print("FA UNFLATTENED tract staged training experiment complete!")
    print("="*80 + "\n")
except Exception as e:
    print(f"UNHANDLED ERROR: {str(e)}")
    import traceback
    print(traceback.format_exc())
    sys.stdout.flush()
    sys.exit(1) 