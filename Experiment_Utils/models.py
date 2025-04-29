import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1DVariationalEncoder_fa(nn.Module):
    def __init__(self, latent_dims=20, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        
        # Instead of directly mapping to latent space, we'll produce two outputs:
        # mean and log variance (each of size latent_dims)
        # self.conv4_mean = nn.Conv1d(64, latent_dims, kernel_size=5, stride=2, padding=2)
        # self.conv4_logvar = nn.Conv1d(64, latent_dims, kernel_size=5, stride=2, padding=2)

        self.fc_mean = nn.Linear(64*7, latent_dims)
        self.fc_logvar = nn.Linear(64*7, latent_dims)
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x = torch.flatten(x, 1)
        x = F.relu(self.conv1(x)) # [64, 16, 25]
        x = self.dropout(x)
        x = F.relu(self.conv2(x)) # [64, 32, 13]
        x = self.dropout(x)
        x = F.relu(self.conv3(x)) # [64, 64, 7]
        x = self.dropout(x)
        
        # mean = self.conv4_mean(x)
        # logvar = self.conv4_logvar(x)
        x = self.flatten(x) # [64, 64*7]
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)    

        return mean, logvar

class Conv1DVariationalDecoder_fa(nn.Module):
    def __init__(self, latent_dims=20):
        super().__init__()
        self.fc = nn.Linear(latent_dims, 64 * 7)
        # self.deconv1 = nn.ConvTranspose1d(latent_dims, 64, kernel_size=5, stride=2, padding=2, output_padding=0)
        self.deconv2 = nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=0)
        self.deconv3 = nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=2, output_padding=1)
        self.deconv4 = nn.ConvTranspose1d(16, 1, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.fc(x)
        x = x.view(batch_size, 64, 7)
        # x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = self.deconv4(x)
        return x
        # return x.view(batch_size, -1)

class Conv1DVariationalAutoencoder_fa(nn.Module):
    def __init__(self, latent_dims=20, dropout=0.0):
        super().__init__()
        self.encoder = Conv1DVariationalEncoder_fa(latent_dims, dropout=dropout)
        self.decoder = Conv1DVariationalDecoder_fa(latent_dims)
        self.latent_dims = latent_dims
        
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z
    
    def forward(self, x):
        mean, logvar = self.encoder(x)
        
        z = self.reparameterize(mean, logvar)
        
        x_prime = self.decoder(z)
        
        return x_prime, mean, logvar
    
class Conv1DEncoder_fa(nn.Module):
    def __init__(self, latent_dims=20, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        
        # Instead of directly mapping to latent space, we'll produce two outputs:
        # mean and log variance (each of size latent_dims)
        self.conv4 = nn.Conv1d(64, latent_dims, kernel_size=5, stride=2, padding=2)
        # self.conv4_logvar = nn.Conv1d(64, latent_dims, kernel_size=5, stride=2, padding=2)

        # self.fc_mean = nn.Linear(64*7, latent_dims)
        # self.fc_logvar = nn.Linear(64*7, latent_dims)
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x = torch.flatten(x, 1)
        x = F.relu(self.conv1(x)) # [64, 16, 25]
        x = self.dropout(x)
        x = F.relu(self.conv2(x)) # [64, 32, 13]
        x = self.dropout(x)
        x = F.relu(self.conv3(x)) # [64, 64, 7]
        x = self.dropout(x)
        x = self.conv4(x) # [64, 64, 4]

        return x

class Conv1DDecoder_fa(nn.Module):
    def __init__(self, latent_dims=20):
        super().__init__()
        # self.fc = nn.Linear(latent_dims, 64 * 7)
        self.deconv1 = nn.ConvTranspose1d(latent_dims, 64, kernel_size=5, stride=2, padding=2, output_padding=0)
        self.deconv2 = nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=0)
        self.deconv3 = nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=2, output_padding=1)
        self.deconv4 = nn.ConvTranspose1d(16, 1, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):

        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = self.deconv4(x)
        return x

class Conv1DAutoencoder_fa(nn.Module):
    def __init__(self, latent_dims=20, dropout=0.0):
        super().__init__()
        self.encoder = Conv1DEncoder_fa(latent_dims, dropout=dropout)
        self.decoder = Conv1DDecoder_fa(latent_dims)
        self.latent_dims = latent_dims
        
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# --- Predictor Models ---

class AgePredictorCNN(nn.Module):
    def __init__(self, input_channels=1, sequence_length=50, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        _dummy_input = torch.randn(1, input_channels, sequence_length)
        _conv_output_shape = self._get_conv_output_shape(_dummy_input)
        flat_size = _conv_output_shape[1] * _conv_output_shape[2]

        # Add an extra input for sex in the first fully connected layer
        self.fc1 = nn.Linear(flat_size + 1, 64)  # +1 for sex feature
        self.fc_out = nn.Linear(64, 1)

    def _get_conv_output_shape(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x.shape

    def forward(self, x, sex=None):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)

        x = self.flatten(x)
        
        # Concatenate sex information if provided
        if sex is not None:
            # Ensure sex has the right shape [batch_size, 1]
            if len(sex.shape) == 1:
                sex = sex.unsqueeze(1)
            x = torch.cat([x, sex], dim=1)
        else:
            # If sex not provided, add a zero column as placeholder
            zeros = torch.zeros(x.size(0), 1, device=x.device)
            x = torch.cat([x, zeros], dim=1)
            
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        age_pred = self.fc_out(x)
        return age_pred

class SitePredictorCNN(nn.Module):
    def __init__(self, num_sites=4, input_channels=1, sequence_length=50, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        _dummy_input = torch.randn(1, input_channels, sequence_length)
        _conv_output_shape = self._get_conv_output_shape(_dummy_input)
        flat_size = _conv_output_shape[1] * _conv_output_shape[2]

        self.fc1 = nn.Linear(flat_size, 64)
        self.fc_out = nn.Linear(64, num_sites)

    def _get_conv_output_shape(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x.shape

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)

        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        site_pred = self.fc_out(x)
        return site_pred


# --- Combined Model ---

try:
    from .utils import grad_reverse
except ImportError:
    try:
        from utils import grad_reverse 
    except ImportError:
        print("Warning: grad_reverse function not found. Define or import it for CombinedVAE_Predictors.")
        def grad_reverse(x, alpha=1.0):
            print("Warning: Using dummy grad_reverse!")
            return x

class CombinedVAE_Predictors(nn.Module):
    def __init__(self, vae_model, age_predictor, site_predictor):
        super().__init__()
        self.vae = vae_model
        self.age_predictor = age_predictor
        self.site_predictor = site_predictor

    def forward(self, x, sex=None, grl_alpha=1.0):
        x_hat, mean, logvar = self.vae(x)
        age_pred = self.age_predictor(x_hat, sex)
        x_hat_reversed = grad_reverse(x_hat, grl_alpha)
        site_pred = self.site_predictor(x_hat_reversed)
        return x_hat, mean, logvar, age_pred, site_pred