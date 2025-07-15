#!/usr/bin/env python
# coding: utf-8

# In[5]:


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
#from afqinsight.nn.pt_models import Conv1DAutoencoder
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:





# In[6]:


# FA FLATTENED DATASET
# NON VARIATIONAL 
# CONVOLUTIONAL AUTOENCODER
# TESTING LATENT AND DROPOUT SIMULTATENTOUSLY 


# In[7]:


import sys 
# sys.path.insert(1, '/Users/samchou/AFQ-Insight-Autoencoder-Experiments/AFQ-Insight-Autoencoder-Experiments/Experiment_Utils')
sys.path.insert(1, './Experiment_Utils') # code should be run from root path of this project "AFQ-Insight-Autoencoder-Experiments"
from utils import train_variational_autoencoder, train_autoencoder, select_device, prep_fa_dataset, prep_first_tract_data, prep_fa_flattned_data
from models import Conv1DAutoencoder_fa


# In[8]:


device = select_device()



# In[9]:


dataset = AFQDataset.from_study('hbn')
#torch_dataset, train_loader, test_loader, val_loader = prep_pytorch_data(dataset,batch_size=256)  
torch_dataset, train_loader, test_loader, val_loader = prep_pytorch_data(dataset)  
gt_shape = torch_dataset[0][1].size()[0]
sequence_length = torch_dataset[0][0].size()[0]  # 48
in_channels = torch_dataset[0][0].size()[1]  # 100


# In[10]:


torch_dataset, all_tracts_train_loader, all_tracts_test_loader, all_tracts_val_loader = prep_fa_flattned_data(dataset, batch_size=128)


# In[11]:


import pandas as pd
import torch

latent_dims = [2, 4, 8, 16, 32, 64, 100]
dropout_values = [0.0, 0.1, 0.5]
models = {}
results = {}

for dropout in dropout_values:
    for latent_dim in latent_dims:

        print(f"Training Autoencoder with Latent Dimension: {latent_dim} and Dropout: {dropout}")
        
        # 1) Train the model and store results
        test_model = Conv1DAutoencoder_fa(latent_dims=latent_dim, dropout=dropout).to(device)
        training_results = train_autoencoder(
            model=test_model,
            train_data=all_tracts_train_loader,
            val_data=all_tracts_val_loader,
            epochs=50,   # or more epochs
            device=device
        )
        
        models[latent_dim, dropout] = test_model
        results[latent_dim, dropout] = training_results
        
        print(f"Completed training for latent_dim={latent_dim}, Best Val RMSE: {training_results['best_val_rmse']:.4f}")

        # 2) Convert Tensors in training_results to CPU floats
        list_keys = [
            "train_rmse_per_epoch",
            "val_rmse_per_epoch",
            "train_recon_loss_per_epoch",
            "val_recon_loss_per_epoch",
            "train_loss_per_epoch",
            "val_loss_per_epoch"
        ]
        
        scalar_keys = [
            "best_val_rmse",
        ]
        
        # Process list values
        for key in list_keys:
            if key in training_results:
                new_list = []
                for val in training_results[key]:  # Iterate only for list values
                    if isinstance(val, torch.Tensor):
                        new_list.append(float(val.cpu().item()))
                    else:
                        new_list.append(float(val))
                training_results[key] = new_list
        
        # Process scalar values separately
        for key in scalar_keys:
            if key in training_results and isinstance(training_results[key], torch.Tensor):
                training_results[key] = float(training_results[key].cpu().item())

        # 3) Build your DataFrames and save them with unique filenames
        num_epochs = len(training_results["train_rmse_per_epoch"])
        
        # Per-epoch metrics
        df_epochs = pd.DataFrame({
            "epoch": range(1, num_epochs + 1),
            "train_rmse": training_results["train_rmse_per_epoch"],
            "val_rmse": training_results["val_rmse_per_epoch"],
            "train_recon_loss": training_results["train_recon_loss_per_epoch"],
            "val_recon_loss": training_results["val_recon_loss_per_epoch"],
            "train_loss": training_results["train_loss_per_epoch"],
            "val_loss": training_results["val_loss_per_epoch"],
            "best_val_rmse": training_results["best_val_rmse"],
        })
        
        # Use f-strings to create unique filenames
        df_epochs.to_csv(f"ae_per_epoch_metrics_ld{latent_dim}_dr{dropout}.csv", index=False)
        
        # Summary row
        df_summary = pd.DataFrame([
            {"best_val_rmse": training_results["best_val_rmse"]}
        ])
        df_summary.to_csv(f"ae_summary_ld{latent_dim}_dr{dropout}.csv", index=False)


# In[ ]:


latent_dims = [2, 4, 8, 16, 32, 64, 100]
dropout_values = [0.0, 0.1, 0.5]
df_best_val = pd.DataFrame(index=dropout_values, columns=latent_dims)

# Populate the DataFrame with the best validation RMSE values
for latent_dim in latent_dims:
    for dropout in dropout_values:
        print(f"Latent Dim: {latent_dim}, Dropout: {dropout}")  
        key = (latent_dim, dropout)
        if key in results:
            best_val_loss = results[key]["best_val_rmse"]
            df_best_val.loc[dropout, latent_dim] = best_val_loss
        else:
            print(f"Key {key} not found in results.")
            # Handle missing key, e.g., set a default value
            df_best_val.loc[dropout, latent_dim] = np.nan  # or any other default


# Convert the values to floats (if they aren't already)
df_best_val = df_best_val.astype(float)

# Plot the heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(df_best_val, annot=True, fmt=".3f", cmap="viridis")
# plt.xlabel("Latent Dimensions")
# plt.ylabel("Dropout Rate")
# plt.title("Best Validation RMSE for Each Dropout and Latent Dimension Combination")
# plt.show()

#Saving the heatmap to a file, for batch scripting 
plt.figure(figsize=(10, 8))
sns.heatmap(df_best_val, annot=True, fmt=".3f", cmap="viridis")
plt.xlabel("Latent Dimensions")
plt.ylabel("Dropout Rate")
plt.title("Best Validation RMSE for Each Dropout and Latent Dimension Combination")
plt.savefig("heatmap_conv_vae_combined_fa_flattened.png")  
plt.close()  


# In[ ]:


#selecting the model with 16 latent dimensions, 0.1 dropout
sample = all_tracts_test_loader.dataset[0][0][0:1].unsqueeze(0).to(device)
output = models[16, 0.1](sample)

# Assuming the first element of the tuple is the reconstruction:
reconstructed = output[0]

orig = sample.cpu().detach().numpy()
recon = reconstructed.cpu().detach().numpy()

plt.plot(orig.flatten()[0:100], color='blue', label='Original')
plt.plot(recon.flatten()[0:100], color='red', label='Reconstructed')
# plt.figure()
# plt.plot(orig.flatten()[0:100], label="Original")
# plt.plot(recon.flatten()[0:100], label="Reconstructed")
# plt.legend()
# plt.title("Original vs. Reconstructed (First 100 Points)")
# plt.savefig("ogvsrecon_conv_vae_combined_fa_flattened.png")  
# plt.close()

