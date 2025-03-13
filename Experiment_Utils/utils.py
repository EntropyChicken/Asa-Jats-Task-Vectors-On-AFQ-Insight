import torch
import torch.nn.functional as F

def train_variational_autoencoder(model, train_data, val_data, epochs=500, lr=0.001, kl_weight=0.001, device = 'cuda'):
    """
    Training loop for variational autoencoder with KL annealing
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=5, factor=0.5)
    
    train_rmse_per_epoch = []
    val_rmse_per_epoch = []
    train_kl_per_epoch = []
    val_kl_per_epoch = []
    train_recon_per_epoch = []
    val_recon_per_epoch = []
    
    best_val_rmse = float('inf')  # Track the best (lowest) validation RMSE
    best_model_state = None  # Save the best model state

    #lets try KL annealing
    beta_start = 0.0
    beta_end = 1.0
    slope = (beta_end - beta_start) / epochs
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0
        running_rmse = 0
        running_kl = 0
        items = 0
        running_recon_loss = 0 
        beta = beta_start + slope * epoch
        
        for x, _ in train_data:
            batch_size = x.size(0)
            tract_data = x.to(device)
            
            opt.zero_grad()
            
            # Forward pass returns reconstructed x, mean and logvar
            x_hat, mean, logvar = model(tract_data)
            
            # Compute loss with KL divergence
            loss, recon_loss, kl_loss = vae_loss(tract_data, x_hat, mean, logvar, beta, reduction="sum")
            #recon loss here is the sum of the MSE
            #loss is the sum of the KL and the recon loss
            #kl loss is the sum of the KL
            #none are normalized yet 

            # Calculate RMSE (primarily for logging)
            batch_rmse = torch.sqrt(F.mse_loss(tract_data, x_hat, reduction="mean"))
            
            loss.backward()
            opt.step()
              
            #increasing by batch size
            items += batch_size
            running_loss += loss.item()
            running_rmse += batch_rmse.item() * batch_size  # Weighted sum
            running_kl += kl_loss.item() # Average KL per item
            running_recon_loss += recon_loss.item() # Average recon loss per item
        
        scheduler.step(running_loss / items)
        avg_train_rmse = running_rmse / items
        avg_train_kl = running_kl / items 
        avg_train_recon_loss = running_recon_loss / items
        train_rmse_per_epoch.append(avg_train_rmse)
        train_kl_per_epoch.append(avg_train_kl)
        train_recon_per_epoch.append(avg_train_recon_loss)

        # Validation
        model.eval()
        val_rmse = 0
        val_kl = 0
        val_items = 0
        val_recon_loss = 0
        
        with torch.no_grad():
            for x, *_ in val_data:
                batch_size = x.size(0)
                tract_data = x.to(device)
                
                x_hat, mean, logvar = model(tract_data)
                
                val_loss, val_recon_loss, val_kl_loss = vae_loss(tract_data, x_hat, mean, logvar, beta, reduction="sum")
                
                batch_val_rmse = torch.sqrt(F.mse_loss(tract_data, x_hat, reduction="mean"))

                val_items += batch_size
                val_loss += val_loss.item()
                val_rmse += batch_val_rmse.item() * tract_data.size(0)
                val_kl += val_kl_loss.item()
                val_recon_loss += val_recon_loss.item()
        
        avg_val_recon_loss = val_recon_loss / val_items
        avg_val_rmse = val_rmse / val_items
        avg_val_kl = val_kl / val_items
        val_rmse_per_epoch.append(avg_val_rmse)
        val_kl_per_epoch.append(avg_val_kl)
        val_recon_per_epoch.append(avg_val_recon_loss)
        
        # Check and save the best model state if current validation loss is lower
        if avg_val_rmse < best_val_rmse:
            print("Saving best model state with RMSE:", avg_val_rmse)
            best_val_rmse = avg_val_rmse
            best_model_state = model.state_dict().copy()  # Make a copy to ensure it's preserved
        
        print(f"Epoch {epoch+1}, Train RMSE: {avg_train_rmse:.4f}, Val RMSE: {avg_val_rmse:.4f}, KL: {avg_train_kl:.4f}," ,
              f"Recon Loss (Train): {avg_train_recon_loss:.4f}, Recon Loss (Val): {avg_val_recon_loss:.4f}")
    
    # Load the best model state back into the model
    model.load_state_dict(best_model_state)
    
    return {
        "train_rmse_per_epoch": train_rmse_per_epoch,
        "val_rmse_per_epoch": val_rmse_per_epoch,
        "train_kl_per_epoch": train_kl_per_epoch,
        "val_kl_per_epoch": val_kl_per_epoch,
        "train_recon_per_epoch": train_recon_per_epoch,
        "val_recon_per_epoch": val_recon_per_epoch,
        "best_val_rmse": best_val_rmse,
    }

def kl_divergence_loss(mean, logvar):
    """
    Compute KL divergence loss for VAE
    """
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return kl_loss


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