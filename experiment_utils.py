import torch
from afqinsight.nn.utils import prep_fa_dataset, reconstruction_loss, vae_loss, kl_divergence_loss, prep_first_tract_data

def train_autoencoder_fixed(model, train_data, val_data, epochs=500, lr=0.001, device = 'cuda'):    
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=5, factor=0.5)
    
    train_rmse_per_epoch = []
    val_rmse_per_epoch = []
    best_val_loss = float('inf')  # Track the best (lowest) validation RMSE
    best_model_state = None  # Save the best model state
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0
        running_rmse = 0
        items = 0
        
        for x, _ in train_data:
            batch_size = x.size(0)
            tract_data = x.to(device)  # Shape: (batch_size, 100)
            
            opt.zero_grad()
            x_hat = model(tract_data)
            loss = reconstruction_loss(tract_data, x_hat, kl_div=0, reduction="sum")
            batch_rmse = torch.sqrt(F.mse_loss(tract_data, x_hat, reduction="mean"))
            
            loss.backward()
            opt.step()
            
            items += tract_data.size(0)
            running_loss += loss.item()
            running_rmse += batch_rmse.item() * tract_data.size(0)  # Weighted sum
            
        scheduler.step(running_loss / items)
        avg_train_rmse = running_rmse / items
        train_rmse_per_epoch.append(avg_train_rmse)
        
        # Validation
        model.eval()
        val_rmse = 0
        val_items = 0
        
        with torch.no_grad():
            for x, _ in val_data:
                tract_data = x.to(device)
                x_hat = model(tract_data)
                batch_val_rmse = torch.sqrt(F.mse_loss(tract_data, x_hat, reduction="mean"))
                val_items += tract_data.size(0)
                val_rmse += batch_val_rmse.item() * tract_data.size(0)
                
        avg_val_rmse = val_rmse / val_items
        val_rmse_per_epoch.append(avg_val_rmse)
        
        # Check and save the best model state if current validation loss is lower
        if avg_val_rmse < best_val_loss:
            best_val_loss = avg_val_rmse
            best_model_state = model.state_dict().copy()  # Make a copy to ensure it's preserved
            
        print(f"Epoch {epoch+1}, Train RMSE: {avg_train_rmse:.4f}, Val RMSE: {avg_val_rmse:.4f}")
    
    # Load the best model state back into the model
    model.load_state_dict(best_model_state)
    
    return {
        "train_rmse_per_epoch": train_rmse_per_epoch,
        "val_rmse_per_epoch": val_rmse_per_epoch,
        "best_val_loss": best_val_loss
    }


