import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1DVariationalEncoder_fa(nn.Module):
    def __init__(self, latent_dims=20, dropout=0.2, input_length=50):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2)
        self.conv2_50 = nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=2)
        self.conv2_100 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        
        # Calculate the output size dynamically
        self._dummy_input = torch.zeros(1, 1, input_length)
        self._conv_output = self._get_conv_output_shape(self._dummy_input)
        self.flattened_size = self._conv_output[1] * self._conv_output[2]
        
        self.fc_mean = nn.Linear(self.flattened_size, latent_dims)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dims)
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def _get_conv_output_shape(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.shape
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2_100(x))
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        
        x = self.flatten(x)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)    

        return mean, logvar

class Conv1DVariationalDecoder_fa(nn.Module):
    def __init__(self, latent_dims=20, conv_output_shape=None):
        super().__init__()
        # Store the expected shape of the conv features
        self.conv_channels = conv_output_shape[1]  # 64
        self.conv_length = conv_output_shape[2]    # 7 for input_length=50, 14 for input_length=100
        self.flattened_size = self.conv_channels * self.conv_length
        
        self.fc = nn.Linear(latent_dims, self.flattened_size)
        self.deconv2 = nn.ConvTranspose1d(self.conv_channels, 32, kernel_size=5, stride=2, padding=2, output_padding=0)
        self.deconv3_100 = nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv3_50 = nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=2, output_padding=1)
        self.deconv4 = nn.ConvTranspose1d(16, 1, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.fc(x)
        x = x.view(batch_size, self.conv_channels, self.conv_length)
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3_100(x))
        x = self.deconv4(x)
        return x

class Conv1DVariationalAutoencoder_fa(nn.Module):
    def __init__(self, latent_dims=20, dropout=0.0, input_length=50):
        super().__init__()
        self.encoder = Conv1DVariationalEncoder_fa(latent_dims, dropout=dropout, input_length=input_length)
        # Pass the shape information from encoder to decoder
        self.decoder = Conv1DVariationalDecoder_fa(latent_dims, self.encoder._conv_output)
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

        # Remove the extra input for sex in the first fully connected layer
        self.fc1 = nn.Linear(flat_size, 64) 
        self.fc_out = nn.Linear(64, 1)

    def _get_conv_output_shape(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x.shape

    def forward(self, x): # Removed sex=None from arguments
        # Input shape: [1, 50]
        # Need to reshape to [1, 1, 50] for conv1d (batch_size, channels, sequence_length)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Shape becomes [1, 1, 50]
            
        # Conv1: kernel=5, stride=2, padding=2
        x = self.relu(self.bn1(self.conv1(x)))  # Shape: [1, 32, 25]
        x = self.dropout(x)
        
        # Conv2: kernel=3, stride=2, padding=1
        x = self.relu(self.bn2(self.conv2(x)))  # Shape: [1, 64, 13]
        x = self.dropout(x)
        
        # Conv3: kernel=3, stride=2, padding=1
        x = self.relu(self.bn3(self.conv3(x)))  # Shape: [1, 128, 7]
        x = self.dropout(x)

        x = self.flatten(x)  # Shape: [1, 128*7] = [1, 896]
        
        x = self.relu(self.fc1(x))  # Shape: [1, 64]
        x = self.dropout(x)
        age_pred = self.fc_out(x)  # Shape: [1, 1]
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
        age_pred = self.age_predictor(x_hat)
        x_hat_reversed = grad_reverse(x_hat, grl_alpha)
        site_pred = self.site_predictor(x_hat_reversed)
        return x_hat, mean, logvar, age_pred, site_pred

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class SelfAttention1D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv1d(in_channels, in_channels//8, kernel_size=1)
        self.key = nn.Conv1d(in_channels, in_channels//8, kernel_size=1)
        self.value = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, L = x.size()
        proj_query = self.query(x).permute(0, 2, 1)
        proj_key = self.key(x)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value(x)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = self.gamma * out + x
        return out

class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        each_channel = out_channels // 3
        remainder = out_channels - (each_channel * 3)
        
        self.conv3 = nn.Conv1d(in_channels, each_channel, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, each_channel, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, each_channel + remainder, kernel_size=7, padding=3)
        
    def forward(self, x):
        return torch.cat([self.conv3(x), self.conv5(x), self.conv7(x)], dim=1)

class ImprovedAgePredictorCNN(nn.Module):
    def __init__(self, input_channels=1, sequence_length=50, dropout=0.3):
        super().__init__()
        
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.multi_scale1 = MultiScaleConv(32, 63) 
        self.bn_ms1 = nn.BatchNorm1d(63)
        
        self.res1 = ResidualBlock(63)
        self.res2 = ResidualBlock(63)
        
        self.attention = SelfAttention1D(63)
        
        self.conv2 = nn.Conv1d(63, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.res3 = ResidualBlock(128)
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        _dummy_input = torch.randn(1, input_channels, sequence_length)
        _conv_output_shape = self._get_conv_output_shape(_dummy_input)
        flat_size = _conv_output_shape[1] * _conv_output_shape[2]
        
        self.sex_embedding = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32)
        )
        
        self.fc1 = nn.Linear(flat_size + 32, 256)  
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn_fc3 = nn.BatchNorm1d(64)
        
        self.fc_out = nn.Linear(64, 1)
    
    def _get_conv_output_shape(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn_ms1(self.multi_scale1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.attention(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.res3(x)
        return x.shape
    
    def forward(self, x, sex=None):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        
        x = self.relu(self.bn_ms1(self.multi_scale1(x)))
        x = self.dropout(x)
        
        x = self.res1(x)
        x = self.res2(x)
        x = self.attention(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        x = self.res3(x)
        x = self.flatten(x)
        
        if sex is not None:
            if len(sex.shape) == 1:
                sex = sex.unsqueeze(1)
            sex_features = self.sex_embedding(sex)
        else:
            sex_features = torch.zeros(x.size(0), 32, device=x.device)
        
        combined = torch.cat([x, sex_features], dim=1)
        
        combined = self.relu(self.bn_fc1(self.fc1(combined)))
        combined = self.dropout(combined)
        
        combined = self.relu(self.bn_fc2(self.fc2(combined)))
        combined = self.dropout(combined)
        
        combined = self.relu(self.bn_fc3(self.fc3(combined)))
        combined = self.dropout(combined)
        
        age_pred = self.fc_out(combined)
        
        return age_pred

class SimpleAgePredictorCNN(nn.Module):
    """
    A simpler age predictor model that may be more robust for our task.
    """
    def __init__(self, input_channels=1, sequence_length=50, dropout=0.2):
        super().__init__()
        # Fewer layers and simpler architecture
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Calculate output size after convolutions
        _dummy_input = torch.randn(1, input_channels, sequence_length)
        _conv_output = self._get_conv_output_shape(_dummy_input)
        flat_size = _conv_output[1] * _conv_output[2]
        
        # Simpler fully connected layers
        self.fc1 = nn.Linear(flat_size + 1, 64)  # +1 for sex feature
        self.fc2 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, 1)
    
    def _get_conv_output_shape(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x.shape
    
    def forward(self, x, sex=None):
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        
        x = self.flatten(x)
        
        # Concatenate sex information
        if sex is not None:
            if len(sex.shape) == 1:
                sex = sex.unsqueeze(1)
            x = torch.cat([x, sex], dim=1)
        else:
            zeros = torch.zeros(x.size(0), 1, device=x.device)
            x = torch.cat([x, zeros], dim=1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        age_pred = self.fc_out(x)
        
        return age_pred