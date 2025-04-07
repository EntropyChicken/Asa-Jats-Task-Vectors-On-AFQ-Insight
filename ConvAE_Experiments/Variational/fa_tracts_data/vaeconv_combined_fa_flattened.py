#!/usr/bin/env python
# coding: utf-8

# In[32]:


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
from afqinsight.nn.utils import prep_fa_dataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
from sklearn.decomposition import PCA
import afqinsight.augmentation as aug
from afqinsight.nn.pt_models import Conv1DAutoencoder
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:





# In[33]:


# FA FLATTENED DATASET
# NON VARIATIONAL 
# CONVOLUTIONAL AUTOENCODER
# TESTING LATENT AND DROPOUT SIMULTATENTOUSLY 


# In[34]:


import sys 
# sys.path.insert(1, '/Users/samchou/AFQ-Insight-Autoencoder-Experiments/AFQ-Insight-Autoencoder-Experiments/Experiment_Utils')
sys.path.insert(1, '/mmfs1/gscratch/nrdg/samchou/AFQ-Insight-Autoencoder-Experiments/Experiment_Utils')
from utils import train_variational_autoencoder, train_autoencoder, select_device, prep_fa_dataset, prep_first_tract_data, prep_fa_flattned_data
from models import Conv1DVariationalAutoencoder_fa


# In[35]:


device = select_device()


# In[36]:


dataset = AFQDataset.from_study('hbn')
torch_dataset, train_loader, test_loader, val_loader = prep_pytorch_data(dataset,batch_size=64)  
gt_shape = torch_dataset[0][1].size()[0]
sequence_length = torch_dataset[0][0].size()[0]  # 48
in_channels = torch_dataset[0][0].size()[1]  # 100


# In[37]:


torch_dataset, all_tracts_train_loader, all_tracts_test_loader, all_tracts_val_loader = prep_fa_flattned_data(dataset, batch_size=64)


# In[38]:


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
        test_model = Conv1DVariationalAutoencoder_fa(latent_dims=latent_dim, dropout=dropout).to(device)
        training_results = train_variational_autoencoder(
            model=test_model,
            train_data=all_tracts_train_loader,
            val_data=all_tracts_val_loader,
            epochs=500,   # or more epochs
            kl_weight=0.001,
            device=device
        )
        
        models[latent_dim, dropout] = test_model
        results[latent_dim, dropout] = training_results
        
        print(f"Completed training for latent_dim={latent_dim}, Best Val RMSE: {training_results['best_val_rmse']:.4f}")

        # 2) Convert Tensors in training_results to CPU floats
        keys_to_convert = [
            "train_rmse_per_epoch",
            "val_rmse_per_epoch",
            "train_kl_per_epoch",
            "val_kl_per_epoch",
            "train_recon_per_epoch",
            "val_recon_per_epoch",
        ]
        for key in keys_to_convert:
            if key in training_results:
                new_list = []
                for val in training_results[key]:
                    if isinstance(val, torch.Tensor):
                        new_list.append(float(val.cpu().item()))
                    else:
                        new_list.append(float(val))
                training_results[key] = new_list

        # If best_val_rmse might be a tensor:
        if isinstance(training_results.get("best_val_rmse"), torch.Tensor):
            training_results["best_val_rmse"] = float(training_results["best_val_rmse"].cpu().item())

        # 3) Build your DataFrames and save them with unique filenames
        num_epochs = len(training_results["train_rmse_per_epoch"])
        
        # Per-epoch metrics
        df_epochs = pd.DataFrame({
            "epoch": range(1, num_epochs + 1),
            "train_rmse": training_results["train_rmse_per_epoch"],
            "val_rmse": training_results["val_rmse_per_epoch"],
            "train_kl": training_results["train_kl_per_epoch"],
            "val_kl": training_results["val_kl_per_epoch"],
            "train_recon_loss": training_results["train_recon_per_epoch"],
            "val_recon_loss": training_results["val_recon_per_epoch"],
        })
        
        # Use f-strings to create unique filenames
        df_epochs.to_csv(f"vae_per_epoch_metrics_ld{latent_dim}_dr{dropout}.csv", index=False)
        
        # Summary row
        df_summary = pd.DataFrame([
            {"best_val_rmse": training_results["best_val_rmse"]}
        ])
        df_summary.to_csv(f"vae_summary_ld{latent_dim}_dr{dropout}.csv", index=False)


# In[41]:


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


# In[42]:


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

