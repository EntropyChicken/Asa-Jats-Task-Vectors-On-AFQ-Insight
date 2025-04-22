import torch
import torch.nn.functional as F
from afqinsight.datasets import AFQDataset
from afqinsight.nn.utils import prep_pytorch_data
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from torch.utils.data import Dataset, DataLoader

def get_beta(current_epoch, total_epochs, start_epoch=100):
    if current_epoch < start_epoch:
        return 0.0
    
    progress = (current_epoch - start_epoch) / (total_epochs - start_epoch)
    
    beta = 1.0 / (1.0 + np.exp(-10 * (progress - 0.5)))

    return beta


def train_variational_autoencoder(model, train_data, val_data, epochs=500, lr=0.001, device = 'cuda',
                                beta=1.0, max_grad_norm=1.0, 
                                kl_annealing_start_epoch=200, kl_annealing_duration=200, kl_annealing_start=0.0001):
    """
    Training loop for variational autoencoder with delayed sigmoid KL annealing.
    KL term has zero weight until kl_annealing_start_epoch, then anneals over kl_annealing_duration.
    """
    torch.backends.cudnn.benchmark = True

    latent_dim = model.latent_dims if hasattr(model, 'latent_dims') else "unknown"
    dropout = model.encoder.dropout.p if hasattr(model, 'encoder') and hasattr(model.encoder, 'dropout') else "unknown"
    
    model_filename = f"best_vae_model_ld{latent_dim}_dr{dropout}.pth"

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=5, factor=0.5)
    
    scaler = torch.amp.GradScaler(device=device)  

    train_rmse_per_epoch = []
    val_rmse_per_epoch = []
    train_recon_per_epoch = []
    val_recon_per_epoch = []
    train_kl_per_epoch = []
    val_kl_per_epoch = []
    train_loss_per_epoch = []
    val_loss_per_epoch = []
    
    best_val_rmse = float('inf')  
    best_model_state = None  
    best_epoch = 0
    
    for epoch in range(epochs):
        if epoch < kl_annealing_start_epoch:
            kl_annealing_factor = 0.0
        elif epoch < kl_annealing_start_epoch + kl_annealing_duration:

            progress = (epoch - kl_annealing_start_epoch) / kl_annealing_duration 

            kl_annealing_factor = kl_annealing_start + (1.0 - kl_annealing_start) * (
                1 / (1 + np.exp(-10 * (progress - 0.5))) 
            )
        else:
            kl_annealing_factor = 1.0
            
        current_beta = beta * kl_annealing_factor
        
        model.train()
        running_loss = 0
        running_rmse = 0
        running_kl = 0
        items = 0
        running_recon_loss = 0
        
        for x, _ in train_data:
            batch_size = x.size(0)
            tract_data = x.to(device, non_blocking=True)
            
            opt.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type= "cuda"):
                x_hat, mean, logvar = model(tract_data)
                
                loss, recon_loss, kl_loss = vae_loss(tract_data, x_hat, mean, logvar, current_beta, reduction="sum")
                
                batch_rmse = torch.sqrt(F.mse_loss(tract_data, x_hat, reduction="mean"))
            
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            
            scaler.step(opt)
            scaler.update()
              
            items += batch_size
            running_loss += loss.item()
            running_rmse += batch_rmse.item() * batch_size 
            if current_beta > 0: 
                running_kl += kl_loss.item()
            running_recon_loss += recon_loss.item() # Average recon loss per item
        
        avg_train_rmse = running_rmse / items
        avg_train_recon_loss = running_recon_loss / items
        # Calculate average KL loss carefully, avoiding division by zero if beta was 0
        avg_train_kl = (running_kl / items) if current_beta > 0 else 0.0 
        avg_train_loss = running_loss / items
        
        train_rmse_per_epoch.append(avg_train_rmse)
        train_kl_per_epoch.append(avg_train_kl)
        train_recon_per_epoch.append(avg_train_recon_loss)
        train_loss_per_epoch.append(avg_train_loss)

        # Validation
        model.eval()
        val_rmse = 0
        val_kl = 0
        val_items = 0
        val_recon_loss = 0
        val_loss_total = 0  # Track total validation loss
        
        with torch.no_grad():
            for x, *_ in val_data:
                batch_size = x.size(0)
                tract_data = x.to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type = "cuda"):
                    x_hat, mean, logvar = model(tract_data)
                    
                    # Use current_beta for validation loss calculation too
                    val_loss, val_recon_loss_batch, val_kl_loss_batch = vae_loss(tract_data, x_hat, mean, logvar, current_beta, reduction="sum")
                    batch_val_rmse = torch.sqrt(F.mse_loss(tract_data, x_hat, reduction="mean"))
                
                val_items += batch_size
                val_loss_total += val_loss.item()
                val_rmse += batch_val_rmse.item() * batch_size
                # Only add KL loss if it has non-zero weight
                if current_beta > 0:
                    val_kl += val_kl_loss_batch.item()
                val_recon_loss += val_recon_loss_batch.item()
        
        avg_val_recon_loss = val_recon_loss / val_items
        avg_val_rmse = val_rmse / val_items
        # Calculate average KL loss carefully
        avg_val_kl = (val_kl / val_items) if current_beta > 0 else 0.0
        avg_val_loss = val_loss_total / val_items
        
        val_rmse_per_epoch.append(avg_val_rmse)
        val_kl_per_epoch.append(avg_val_kl)
        val_recon_per_epoch.append(avg_val_recon_loss)
        val_loss_per_epoch.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        # Check and save the best model state if current validation loss is lower
        if avg_val_rmse < best_val_rmse:
            print(f"Saving best model state with RMSE: {avg_val_rmse:.4f} at epoch {epoch+1}")
            best_val_rmse = avg_val_rmse
            best_model_state = model.state_dict().copy()  # Make a copy to ensure it's preserved
            best_epoch = epoch + 1  # Make a copy to ensure it's preserved
            
            torch.save(best_model_state, model_filename)
            print(f"Best model saved to: {model_filename}")
        
        print(f"Epoch {epoch+1}, KL Weight: {current_beta:.6f}, Train RMSE: {avg_train_rmse:.4f}, Val RMSE: {avg_val_rmse:.4f}, KL (Train): {avg_train_kl:.4f}, KL (Val): {avg_val_kl:.4f}, "
              f"Recon (Train): {avg_train_recon_loss:.4f}, Recon (Val): {avg_val_recon_loss:.4f}")
    
    print(f"Training complete. Best model was from epoch {best_epoch} with validation RMSE: {best_val_rmse:.4f}")
    
    return {
        "train_rmse_per_epoch": train_rmse_per_epoch,
        "val_rmse_per_epoch": val_rmse_per_epoch,
        "train_kl_per_epoch": train_kl_per_epoch,
        "val_kl_per_epoch": val_kl_per_epoch,
        "train_recon_per_epoch": train_recon_per_epoch,
        "val_recon_per_epoch": val_recon_per_epoch,
        "train_loss_per_epoch": train_loss_per_epoch,
        "val_loss_per_epoch": val_loss_per_epoch,
        "best_val_rmse": best_val_rmse,
        "best_epoch": best_epoch,
        "model_path": model_filename
    }

def train_autoencoder(model, train_data, val_data, epochs=500, lr=0.001, device = 'cuda', max_grad_norm=1.0):
    """
    Training loop for standard autoencoder
    """
    torch.backends.cudnn.benchmark = True

    # Get latent dimensions and dropout from the model
    latent_dim = model.latent_dims if hasattr(model, 'latent_dims') else "unknown"
    # Get dropout from the encoder component
    dropout = model.encoder.dropout.p if hasattr(model, 'encoder') and hasattr(model.encoder, 'dropout') else "unknown"
    
    # Create a unique model filename
    model_filename = f"best_ae_model_ld{latent_dim}_dr{dropout}.pth"

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=5, factor=0.5)
    
    scaler = torch.amp.GradScaler(device=device)  # For mixed precision training

    train_rmse_per_epoch = []
    val_rmse_per_epoch = []
    train_recon_loss_per_epoch = []
    val_recon_loss_per_epoch = []
    train_loss_per_epoch = []
    val_loss_per_epoch = []
    
    best_val_rmse = float('inf')  # Track the best (lowest) validation RMSE
    best_model_state = None  # Save the best model state
    best_epoch = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0
        running_rmse = 0
        items = 0
        running_recon_loss = 0
        
        for x, _ in train_data:
            batch_size = x.size(0)
            tract_data = x.to(device, non_blocking=True)
            
            opt.zero_grad(set_to_none=True)
            
            # Forward pass
            with torch.amp.autocast(device_type= "cuda"):
                x_hat = model(tract_data)
                
                # Compute loss
                recon_loss = F.mse_loss(tract_data, x_hat, reduction="sum")
                loss = recon_loss
                
                # Calculate RMSE (primarily for logging)
                batch_rmse = torch.sqrt(F.mse_loss(tract_data, x_hat, reduction="mean"))
            
            # Scale the total loss for backward pass
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            
            # Unscale gradients for clipping
            scaler.unscale_(opt)
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            
            # Step optimizer with scaled gradients
            scaler.step(opt)
            scaler.update()
              
            #increasing by batch size
            items += batch_size
            running_loss += loss.item()
            running_rmse += batch_rmse.item() * batch_size  # Weighted sum
            running_recon_loss += recon_loss.item() # Average recon loss per item
        
        avg_train_rmse = running_rmse / items
        avg_train_recon_loss = running_recon_loss / items
        avg_train_loss = running_loss / items
        train_rmse_per_epoch.append(avg_train_rmse)
        train_recon_loss_per_epoch.append(avg_train_recon_loss)
        train_loss_per_epoch.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_rmse = 0
        val_recon_loss = 0
        val_items = 0
        
        with torch.no_grad():
            for x, _ in val_data:
                batch_size = x.size(0)
                tract_data = x.to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type="cuda"):
                    # Forward pass
                    x_hat = model(tract_data)
                    
                    loss = F.mse_loss(tract_data, x_hat, reduction="sum")
                    batch_val_rmse = torch.sqrt(F.mse_loss(tract_data, x_hat, reduction="mean"))
                
                val_items += batch_size
                val_recon_loss += loss.item()
                val_rmse += batch_val_rmse.item() * batch_size
        
        avg_val_rmse = val_rmse / val_items
        avg_val_recon_loss = val_recon_loss / val_items
        avg_val_loss = avg_val_recon_loss
        
        val_rmse_per_epoch.append(avg_val_rmse)
        val_recon_loss_per_epoch.append(avg_val_recon_loss)
        val_loss_per_epoch.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        # Check and save the best model state if current validation RMSE is lower
        if avg_val_rmse < best_val_rmse:
            print(f"Epoch {epoch+1}: Saving best model state with RMSE: {avg_val_rmse:.4f}")
            best_val_rmse = avg_val_rmse
            best_epoch = epoch + 1
            
            # Save the best model weights to disk
            torch.save(model.state_dict(), model_filename)
            print(f"Best model saved to: {model_filename}")
        
        print(f"Epoch {epoch+1}, Train RMSE: {avg_train_rmse:.4f}, Val RMSE: {avg_val_rmse:.4f}, " +
              f"Recon Loss (Train): {avg_train_recon_loss:.4f}, Recon Loss (Val): {avg_val_recon_loss:.4f}")
        
    print(f"Training complete. Best model was from epoch {best_epoch} with validation RMSE: {best_val_rmse:.4f}")
    print(f"Best model saved to: {model_filename}")
    
    return {
        "train_rmse_per_epoch": train_rmse_per_epoch,
        "val_rmse_per_epoch": val_rmse_per_epoch,
        "train_recon_loss_per_epoch": train_recon_loss_per_epoch,
        "val_recon_loss_per_epoch": val_recon_loss_per_epoch,
        "train_loss_per_epoch": train_loss_per_epoch,
        "val_loss_per_epoch": val_loss_per_epoch,
        "best_val_rmse": best_val_rmse,
        "best_epoch": best_epoch,
        "model_path": model_filename
    }

def kl_divergence_loss(mean, logvar):
    """
    KL divergence between the learned distribution and a standard Gaussian.
    """
    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

def vae_loss(x, x_hat, mean, logvar, kl_weight=1.0, reduction="sum"):
    """
    Combined VAE loss: reconstruction + KL divergence
    """
    if reduction == "sum":
        recon_loss = F.mse_loss(x, x_hat, reduction="sum")
    else:
        recon_loss = F.mse_loss(x, x_hat, reduction="mean")

    kl_loss = kl_divergence_loss(mean, logvar)

    total_loss = recon_loss + kl_weight * kl_loss

    return total_loss, recon_loss, kl_loss

def select_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print("Using device:", device)
    print()

    if device.type == 'cuda':
        print("CUDA Device Name:", torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("  Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
        print("  Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")
    elif device.type == 'mps':
        print("Using MPS backend on macOS. (Detailed memory info may not be available.)")

    return device

def prep_fa_flattned_data(dataset, batch_size=64):
    """
    Prepares PyTorch dataloaders for training, testing, and validation.
    These dataloaders select the fa tracts ONLY and flatten them to "create" more data.

    Parameters
    ----------
    dataset : AFQDataset
        The dataset to extract fa tracts from and flatten
    batch_size : int
        The batch size to be used.

    Returns
    -------
    tuple:
        The FA dataset,
        New Training data loader,
        New Test data loader,
        New Validation data loader.
    """
    torch_dataset_fa, train_loader_fa, test_loader_fa, val_loader_fa = prep_fa_dataset(
        dataset, batch_size=batch_size
    )

    class AllTractsDataset(torch.utils.data.Dataset):
        def __init__(self, original_dataset):
            self.original_dataset = original_dataset
            self.sample_count = len(original_dataset)
            self.tract_count = original_dataset[0][0].shape[0]

        def __len__(self):
            return self.sample_count * self.tract_count

        def __getitem__(self, idx):
            sample_idx = idx // self.tract_count
            tract_idx = idx % self.tract_count

            x, y = self.original_dataset[sample_idx]

            tract_data = x[tract_idx : tract_idx + 1, :].clone()

            return tract_data, y

    all_tracts_train_dataset = AllTractsDataset(train_loader_fa.dataset)
    all_tracts_test_dataset = AllTractsDataset(test_loader_fa.dataset)
    all_tracts_val_dataset = AllTractsDataset(val_loader_fa.dataset)

    all_tracts_train_loader = torch.utils.data.DataLoader(
        all_tracts_train_dataset, batch_size=batch_size, shuffle=True
    )
    all_tracts_test_loader = torch.utils.data.DataLoader(
        all_tracts_test_dataset, batch_size=batch_size, shuffle=False
    )
    all_tracts_val_loader = torch.utils.data.DataLoader(
        all_tracts_val_dataset, batch_size=batch_size, shuffle=False
    )

    return (
        torch_dataset_fa,
        all_tracts_train_loader,
        all_tracts_test_loader,
        all_tracts_val_loader,
    )

def prep_fa_dataset(dataset, target_labels="dki_fa", batch_size=32):
    """
    Extracts features that match the specified label from the provided dataset and
    prepares the dataset for training.
    """
    # Can be single target or a list of targets
    if isinstance(target_labels, str):
        features = [target_labels]
    else:
        features = target_labels

    fa_indices = []
    for i, fname in enumerate(dataset.feature_names):
        if any(feature in fname for feature in features):
            fa_indices.append(i)

    if not fa_indices:
        available_features = sorted(
            {fname.split("_")[0] for fname in dataset.feature_names}
        )
        raise ValueError(
            f"No features found matching patterns: {features}. "
            f"Features found: {available_features}"
        )

    X_fa = dataset.X[:, fa_indices]
    feature_names_fa = [dataset.feature_names[i] for i in fa_indices]
    dataset_fa = AFQDataset(
        X=X_fa,
        y=dataset.y,
        groups=dataset.groups,
        feature_names=feature_names_fa,
        group_names=dataset.group_names,
        target_cols=dataset.target_cols,
        subjects=dataset.subjects,
        sessions=dataset.sessions,
        classes=dataset.classes,
    )
    return prep_pytorch_data(dataset_fa, batch_size=batch_size)

def prep_first_tract_data(dataset, batch_size=32):
    """
    Prepares PyTorch dataloaders for training, testing, and validation.
    These dataloaders select the first tract ONLY.

    Parameters
    ----------
    dataset : AFQDataset
        The dataset to extract the first tract from.
    batch_size : int
        The batch size to be used.

    Returns
    -------
    tuple:
        PyTorch dataset,
        New Training data loader,
        New Test data loader,
        New Validation data loader.
    """
    torch_dataset, train_loader, test_loader, val_loader = prep_pytorch_data(
        dataset, batch_size=batch_size
    )

    class FirstTractDataset(torch.utils.data.Dataset):
        def __init__(self, original_dataset):
            self.original_dataset = original_dataset

        def __len__(self):
            return len(self.original_dataset)

        def __getitem__(self, idx):
            x, y = self.original_dataset[idx]
            tract_data = x[0:1, :].clone()
            return tract_data, y

    first_tract_train_dataset = FirstTractDataset(train_loader.dataset)
    first_tract_test_dataset = FirstTractDataset(test_loader.dataset)
    first_tract_val_dataset = FirstTractDataset(val_loader.dataset)

    # Create first tract data loaders
    first_tract_train_loader = torch.utils.data.DataLoader(
        first_tract_train_dataset, batch_size=batch_size, shuffle=True
    )
    first_tract_test_loader = torch.utils.data.DataLoader(
        first_tract_test_dataset, batch_size=batch_size, shuffle=False
    )
    first_tract_val_loader = torch.utils.data.DataLoader(
        first_tract_val_dataset, batch_size=batch_size, shuffle=False
    )

    return (
        torch_dataset,
        first_tract_train_loader,
        first_tract_test_loader,
        first_tract_val_loader,
    )

# === Adversarial Training Components ===

# Gradient Reversal Layer
class GradReverse(torch.autograd.Function):
    """
    Gradient Reversal Layer: Acts as identity during forward pass,
    reverses gradient sign scaled by alpha during backward pass.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def grad_reverse(x, alpha=1.0):
    """Helper function to apply the Gradient Reversal Layer."""
    return GradReverse.apply(x, alpha)

def train_variational_autoencoder_age_site(
    combined_model,
    train_data,
    val_data,
    epochs=500,
    lr=0.001,
    device="cuda",
    max_grad_norm=1.0,
    w_recon=1.0,
    w_kl=1.0,
    w_age=1.0,
    w_site=1.0,
    kl_annealing_start_epoch=0,
    kl_annealing_duration=200,
    kl_annealing_start=0.0001,
    grl_alpha_start=0.0,
    grl_alpha_end=1.0,
    grl_alpha_epochs=100,
    save_prefix="best_combined_model",
    val_metric_to_monitor="val_age_mae"
):
    torch.backends.cudnn.benchmark = True

    opt = torch.optim.Adam(combined_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=10, factor=0.5, verbose=True)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    train_loss_epoch = []
    val_loss_epoch = []
    train_recon_loss_epoch = []
    val_recon_loss_epoch = []
    train_kl_loss_epoch = []
    val_kl_loss_epoch = []
    train_age_loss_epoch = []
    val_age_loss_epoch = []
    train_site_loss_epoch = []
    val_site_loss_epoch = []
    train_age_mae_epoch = []
    val_age_mae_epoch = []
    train_site_acc_epoch = []
    val_site_acc_epoch = []
    current_beta_epoch = []
    current_grl_alpha_epoch = []
    current_lr_epoch = []

    best_val_metric_value = float("inf") 
    best_model_state = None
    best_epoch = 0
    model_filename = f"{save_prefix}.pth"

    # --- Loss Criteria --- (Defined outside the loop)
    recon_criterion = torch.nn.MSELoss(reduction="mean")
    age_criterion = torch.nn.L1Loss(reduction="mean") # MAE
    site_criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    if not hasattr(combined_model, "forward") or len(combined_model.forward.__code__.co_varnames) < 3:
        print("Warning: combined_model.forward signature should ideally accept (self, x, grl_alpha).")

    print(f"Starting combined training on {device}... Monitoring ", val_metric_to_monitor)
    num_train_batches = len(train_data)
    # num_val_batches = len(val_data)

    for epoch in range(epochs):
        current_lr_epoch.append(opt.param_groups[0]["lr"])

        # --- GRL Alpha Calculation ---
        if grl_alpha_epochs > 0 and epoch < grl_alpha_epochs:
            progress = epoch / grl_alpha_epochs
            current_grl_alpha = grl_alpha_start + (grl_alpha_end - grl_alpha_start) * progress
        elif epoch >= grl_alpha_epochs:
            current_grl_alpha = grl_alpha_end
        else:
            current_grl_alpha = grl_alpha_end
        current_grl_alpha_epoch.append(current_grl_alpha)

        # --- KL Beta Calculation ---
        if kl_annealing_duration > 0 and epoch >= kl_annealing_start_epoch:
            annealing_epoch = epoch - kl_annealing_start_epoch
            if annealing_epoch < kl_annealing_duration:
                progress = annealing_epoch / kl_annealing_duration
                sigmoid_val = 1 / (1 + np.exp(-10 * (progress - 0.5)))
                kl_annealing_factor = kl_annealing_start + (1.0 - kl_annealing_start) * sigmoid_val
            else:
                kl_annealing_factor = 1.0
        else:
            kl_annealing_factor = 0.0 if epoch < kl_annealing_start_epoch else 1.0
        current_beta = w_kl * kl_annealing_factor
        current_beta_epoch.append(current_beta)

        current_w_recon = w_recon
        current_w_age = w_age
        current_w_site = w_site

        # =================== TRAINING ==================
        combined_model.train()
        running_loss = 0.0
        running_recon_loss = 0.0
        running_kl_loss = 0.0
        running_age_loss = 0.0
        running_site_loss = 0.0
        running_age_mae_sum = 0.0 
        running_site_correct = 0.0
        train_items = 0

        for i, (x, labels) in enumerate(train_data):
            batch_size = x.size(0)
            tract_data = x.to(device, non_blocking=True)



            age_true = labels[:, 0].float().unsqueeze(1).to(device) #for some reason non_blocking=True causes nan values
            site_true = labels[:, 1].long().to(device, non_blocking=True) # Get remapped site index as Long

            if torch.isnan(age_true).any(): print(f"train NaN found in age_true! Batch indices: {torch.where(torch.isnan(age_true))[0].tolist()}")

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                model_out = combined_model(tract_data, grl_alpha=current_grl_alpha)
                x_hat, mean, logvar, age_pred, site_pred = model_out

                recon_loss = recon_criterion(x_hat, tract_data)
                kl_loss_unreduced = kl_divergence_loss(mean, logvar)
                kl_loss = kl_loss_unreduced / batch_size # Normalize per batch item
                age_loss = age_criterion(age_pred, age_true)
                site_loss = site_criterion(site_pred, site_true)

                if torch.isnan(recon_loss): print(f"NaN found in recon_loss!")
                if torch.isnan(kl_loss): print(f"NaN found in kl_loss! mean={mean.mean().item():.2f}, logvar={logvar.mean().item():.2f}")

                if torch.isnan(age_loss): print(f"NaN found in age_loss! age_pred mean: {age_pred.mean().item():.2f}, age_true mean: {age_true.nanmean().item():.2f}, any age_true NaN: {torch.isnan(age_true).any()}")
                if torch.isnan(site_loss): print(f"NaN found in site_loss!")
                # -------------------------------------------

                total_loss = (current_w_recon * recon_loss +
                              current_beta * kl_loss +
                              current_w_age * age_loss +
                              current_w_site * site_loss)

                # --- DEBUG: Check total loss ---
                if torch.isnan(total_loss): print("train NaN found in total_loss BEFORE backward!")
                # ---------------------------------

            scaler.scale(total_loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(combined_model.parameters(), max_norm=max_grad_norm)
            scaler.step(opt)
            scaler.update()

            train_items += batch_size
            running_loss += total_loss.item() * batch_size
            running_recon_loss += recon_loss.item() * batch_size
            running_kl_loss += kl_loss.item() * batch_size
            running_age_loss += age_loss.item() * batch_size
            running_site_loss += site_loss.item() * batch_size
            running_age_mae_sum += age_loss.item() * batch_size # L1 loss is MAE
            _, predicted_sites = torch.max(site_pred.data, 1)
            # Use correct target variable
            running_site_correct += (predicted_sites == site_true).sum().item()

            if (i + 1) % 10 == 0 or (i + 1) == num_train_batches:
                print(f"\rEpoch {epoch+1}/{epochs} | Batch {i+1}/{num_train_batches} | Train Loss: {total_loss.item():.4f}", end="")

        # Calculate average training metrics for the epoch
        avg_train_loss = running_loss / train_items
        avg_train_recon_loss = running_recon_loss / train_items
        avg_train_kl_loss = running_kl_loss / train_items
        avg_train_age_loss = running_age_loss / train_items
        avg_train_site_loss = running_site_loss / train_items
        avg_train_age_mae = running_age_mae_sum / train_items
        avg_train_site_acc = (running_site_correct / train_items) * 100

        # Append training metrics to lists
        train_loss_epoch.append(avg_train_loss)
        train_recon_loss_epoch.append(avg_train_recon_loss)
        train_kl_loss_epoch.append(avg_train_kl_loss)
        train_age_loss_epoch.append(avg_train_age_loss)
        train_site_loss_epoch.append(avg_train_site_loss)
        train_age_mae_epoch.append(avg_train_age_mae)
        train_site_acc_epoch.append(avg_train_site_acc)

        # =================== VALIDATION ==================
        combined_model.eval()
        running_val_loss = 0.0
        running_val_recon_loss = 0.0
        running_val_kl_loss = 0.0
        running_val_age_loss = 0.0
        running_val_site_loss = 0.0
        running_val_age_mae_sum = 0.0
        running_val_site_correct = 0.0
        val_items = 0

        with torch.no_grad():
            for x, labels in val_data:
                batch_size = x.size(0)
                tract_data = x.to(device, non_blocking=True)
                # Revert to correct batch-wise slicing
                age_true = labels[:, 0].float().unsqueeze(1).to(device)
                site_true = labels[:, 1].long().to(device, non_blocking=True)

                # --- DEBUG: Check for NaNs in true labels ---
                if torch.isnan(age_true).any(): print(f"NaN found in val age_true! Batch indices: {torch.where(torch.isnan(age_true))[0].tolist()}")
                # ---------------------------------------------

                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    # Removed print
                    model_out = combined_model(tract_data, grl_alpha=current_grl_alpha)
                    x_hat, mean, logvar, age_pred, site_pred = model_out
                    # Removed print
                    # --- DEBUG: Check for NaNs in model outputs ---

                    recon_loss = recon_criterion(x_hat, tract_data)
                    kl_loss = kl_divergence_loss(mean, logvar) / batch_size

                    age_loss = age_criterion(age_pred, age_true)
                    site_loss = site_criterion(site_pred, site_true)

                    # --- DEBUG: Check individual loss values ---
                    if torch.isnan(recon_loss): print(f"NaN found in recon_loss!")
                    if torch.isnan(kl_loss): print(f"NaN found in kl_loss! mean={mean.mean().item():.2f}, logvar={logvar.mean().item():.2f}")
                    # Use correct target variable in debug message
                    if torch.isnan(age_loss): print(f"NaN found in age_loss! age_pred mean: {age_pred.mean().item():.2f}, age_true mean: {age_true.nanmean().item():.2f}, any age_true NaN: {torch.isnan(age_true).any()}")
                    if torch.isnan(site_loss): print(f"NaN found in site_loss!")
                    # -------------------------------------------

                    total_loss = (current_w_recon * recon_loss +
                                  current_beta * kl_loss +
                                  current_w_age * age_loss +
                                  current_w_site * site_loss)

                    # --- DEBUG: Check total loss ---
                    if torch.isnan(total_loss): print("NaN found in total_loss BEFORE backward!")
                    # ---------------------------------

                val_items += batch_size
                running_val_loss += total_loss.item() * batch_size
                running_val_recon_loss += recon_loss.item() * batch_size
                running_val_kl_loss += kl_loss.item() * batch_size
                running_val_age_loss += age_loss.item() * batch_size
                running_val_site_loss += site_loss.item() * batch_size
                running_val_age_mae_sum += age_loss.item() * batch_size
                _, predicted_sites = torch.max(site_pred.data, 1)
                # Use correct target variable
                running_val_site_correct += (predicted_sites == site_true).sum().item()

        # Calculate average validation metrics
        avg_val_loss = running_val_loss / val_items
        avg_val_recon_loss = running_val_recon_loss / val_items
        avg_val_kl_loss = running_val_kl_loss / val_items
        avg_val_age_loss = running_val_age_loss / val_items
        avg_val_site_loss = running_val_site_loss / val_items
        avg_val_age_mae = running_val_age_mae_sum / val_items
        avg_val_site_acc = (running_val_site_correct / val_items) * 100

        # Append validation metrics to lists
        val_loss_epoch.append(avg_val_loss)
        val_recon_loss_epoch.append(avg_val_recon_loss)
        val_kl_loss_epoch.append(avg_val_kl_loss)
        val_age_loss_epoch.append(avg_val_age_loss)
        val_site_loss_epoch.append(avg_val_site_loss)
        val_age_mae_epoch.append(avg_val_age_mae)
        val_site_acc_epoch.append(avg_val_site_acc)

        # --- Scheduler Step --- (Step based on the chosen metric)
        # Create a temporary dict to easily access the metric value by key
        current_epoch_val_metrics = {
            "val_loss": avg_val_loss,
            "val_recon_loss": avg_val_recon_loss,
            "val_kl_loss": avg_val_kl_loss,
            "val_age_loss": avg_val_age_loss,
            "val_site_loss": avg_val_site_loss,
            "val_age_mae": avg_val_age_mae,
            "val_site_acc": avg_val_site_acc # Accuracy isn't usually used for ReduceLROnPlateau min mode
        }
        metric_to_schedule = current_epoch_val_metrics.get(val_metric_to_monitor)
        if metric_to_schedule is None:
            print(f"Warning: Metric '{val_metric_to_monitor}' not found for scheduler. Defaulting to 'val_loss'.")
            metric_to_schedule = avg_val_loss
        scheduler.step(metric_to_schedule)

        # --- Checkpoint Best Model (similar to train_vae) ---
        current_val_metric = metric_to_schedule # Use the same metric value as scheduler
        if current_val_metric < best_val_metric_value:
            best_val_metric_value = current_val_metric
            best_epoch = epoch + 1
            best_model_state = combined_model.state_dict().copy() # Store state dict
            print(f"\nEpoch {epoch+1}: New best model found! {val_metric_to_monitor}: {best_val_metric_value:.4f}. State stored.")

        # --- Print Epoch Summary --- (Using average metrics)
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  LR: {current_lr_epoch[-1]:.1e} | Beta: {current_beta:.4f} | GRL Alpha: {current_grl_alpha:.4f}")
        print(f"  Loss (Train/Val): {avg_train_loss:.4f} / {avg_val_loss:.4f}")
        print(f"  Recon Loss (Train/Val): {avg_train_recon_loss:.4f} / {avg_val_recon_loss:.4f}")
        print(f"  KL Loss (Train/Val): {avg_train_kl_loss:.6f} / {avg_val_kl_loss:.6f}")
        print(f"  Age MAE (Train/Val): {avg_train_age_mae:.4f} / {avg_val_age_mae:.4f}")
        print(f"  Site Acc (Train/Val): {avg_train_site_acc:.2f}% / {avg_val_site_acc:.2f}%")
        print("-" * 60)

    # --- Save Best Model State After Loop --- (similar to train_vae)
    if best_model_state is not None:
        print(f"\nTraining complete. Saving best model from epoch {best_epoch} ({val_metric_to_monitor}: {best_val_metric_value:.4f}) to {model_filename}")
        torch.save(best_model_state, model_filename)
    else:
        print("\nTraining complete. No best model state was saved (no improvement found or error occurred).")
        model_filename = None # Indicate no model was saved

    # --- Return Results Dictionary (similar to train_vae) ---
    results = {
        "train_loss_epoch": train_loss_epoch,
        "val_loss_epoch": val_loss_epoch,
        "train_recon_loss_epoch": train_recon_loss_epoch,
        "val_recon_loss_epoch": val_recon_loss_epoch,
        "train_kl_loss_epoch": train_kl_loss_epoch,
        "val_kl_loss_epoch": val_kl_loss_epoch,
        "train_age_loss_epoch": train_age_loss_epoch,
        "val_age_loss_epoch": val_age_loss_epoch,
        "train_site_loss_epoch": train_site_loss_epoch,
        "val_site_loss_epoch": val_site_loss_epoch,
        "train_age_mae_epoch": train_age_mae_epoch,
        "val_age_mae_epoch": val_age_mae_epoch,
        "train_site_acc_epoch": train_site_acc_epoch,
        "val_site_acc_epoch": val_site_acc_epoch,
        "current_beta_epoch": current_beta_epoch,
        "current_grl_alpha_epoch": current_grl_alpha_epoch,
        "current_lr_epoch": current_lr_epoch,
        f"best_{val_metric_to_monitor}": best_val_metric_value,
        "best_epoch": best_epoch,
        "model_path": model_filename
    }

    return results

# Ensure kl_divergence_loss is available in the scope
# If not, uncomment or add:
# def kl_divergence_loss(mean, logvar):
#     return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=[i for i in range(1, mean.dim())])

def prep_fa_flattened_remapped_data(dataset, batch_size=64, site_col_name='scan_site_id', age_col_name='age'):
    """
    Prepares PyTorch dataloaders like prep_fa_flattned_data but also remaps site IDs.
    Selects FA tracts ONLY, flattens them per tract, and remaps site IDs.

    Parameters
    ----------
    dataset : AFQDataset
        The dataset to extract fa tracts from and flatten.
    batch_size : int
        The batch size to be used.
    site_col_name : str
        Name of the site column in dataset.target_cols.
    age_col_name : str
        Name of the age column in dataset.target_cols.

    Returns
    -------
    tuple:
        Original FA dataset (subsetted features),
        New Training data loader (yields x_tract, [age, remapped_site]),
        New Test data loader,
        New Validation data loader.
    """
    # 1. Prepare FA-only dataset first (reuse existing logic)
    # Assuming target_labels="dki_fa" is appropriate for selecting FA features
    torch_dataset_fa, train_loader_fa, test_loader_fa, val_loader_fa = prep_fa_dataset(
        dataset, target_labels="dki_fa", batch_size=batch_size
    )

    # 2. Get necessary info for remapping from the original dataset
    try:
        original_target_cols = dataset.target_cols
        age_idx = original_target_cols.index(age_col_name)
        site_idx = original_target_cols.index(site_col_name)
        print(f"Remapping prep: Using age index {age_idx}, site index {site_idx} from {original_target_cols}")
    except (AttributeError, ValueError) as e:
        print(f"Error finding columns '{age_col_name}' or '{site_col_name}' in dataset.target_cols: {e}")
        raise ValueError("Could not find required columns for remapping.") from e

    # Define the site map {original_id: new_id}
    site_map = {0.0: 0.0, 1.0: 1.0, 3.0: 2.0, 4.0: 3.0}
    print(f"Using site map: {site_map}")

    # 3. Define the modified AllTractsDataset with remapping
    class AllTractsRemappedDataset(Dataset):
        def __init__(self, original_fa_dataset, age_idx, site_idx, site_map):
            self.original_fa_dataset = original_fa_dataset # This is the FA-only TensorDataset
            self.sample_count = len(original_fa_dataset)
            # Assuming original_fa_dataset[0][0] gives fa_tract data [num_tracts, num_nodes]
            sample_x, _ = original_fa_dataset[0]
            self.tract_count = sample_x.shape[0]
            self.age_idx = age_idx
            self.site_idx = site_idx
            self.site_map = site_map
            if not isinstance(original_fa_dataset, torch.utils.data.Dataset):
                 raise TypeError("original_fa_dataset must be an instance of torch.utils.data.Dataset")

        def __len__(self):
            # Each sample yields tract_count individual items
            return self.sample_count * self.tract_count

        def __getitem__(self, idx):
            # Map flattened index back to original sample and tract
            sample_idx = idx // self.tract_count
            tract_idx = idx % self.tract_count

            # Get data from the original FA-only dataset
            # base_dataset yields (fa_data, original_y)
            fa_data, original_y = self.original_fa_dataset[sample_idx]

            # Extract the specific tract profile
            # Ensure it has shape [1, num_nodes] for Conv1D
            tract_data = fa_data[tract_idx : tract_idx + 1, :].clone()

            # Extract age and original site from the original y tensor
            age = original_y[self.age_idx].item()
            original_site = original_y[self.site_idx].item()

            # Remap the site ID
            remapped_site = self.site_map.get(original_site, -1.0) # Default to -1.0 if not found
            if remapped_site == -1.0:
                 print(f"Warning: Site value {original_site} at original index {sample_idx} not in map!")

            # Create the new label tensor [age, remapped_site]
            new_labels = torch.tensor([age, remapped_site], dtype=torch.float32)

            return tract_data, new_labels

    # 4. Create instances of the new Dataset using the base FA datasets
    print("Creating remapped datasets...")
    all_tracts_train_dataset = AllTractsRemappedDataset(train_loader_fa.dataset, age_idx, site_idx, site_map)
    all_tracts_test_dataset = AllTractsRemappedDataset(test_loader_fa.dataset, age_idx, site_idx, site_map)
    all_tracts_val_dataset = AllTractsRemappedDataset(val_loader_fa.dataset, age_idx, site_idx, site_map)

    # 5. Create the final DataLoaders
    print("Creating final DataLoaders...")
    # Use torch.utils.data.DataLoader explicitly
    all_tracts_train_loader = torch.utils.data.DataLoader(
        all_tracts_train_dataset, batch_size=batch_size, shuffle=True
    )
    all_tracts_test_loader = torch.utils.data.DataLoader(
        all_tracts_test_dataset, batch_size=batch_size, shuffle=False
    )
    all_tracts_val_loader = torch.utils.data.DataLoader(
        all_tracts_val_dataset, batch_size=batch_size, shuffle=False
    )

    print("prep_fa_flattened_remapped_data complete.")
    return (
        torch_dataset_fa, # Return original subsetted FA dataset for reference
        all_tracts_train_loader,
        all_tracts_test_loader,
        all_tracts_val_loader,
    )