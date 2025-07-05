#!/usr/bin/env python3
"""
Test script to verify that the AFQ-Insight-Autoencoder-Experiments package is installed correctly.
"""

import torch
import numpy as np

def test_models_import():
    """Test that all main model classes can be imported."""
    from Experiment_Utils.models import (
        Conv1DAutoencoder_fa,
        Conv1DVariationalAutoencoder_fa,
        AgePredictorCNN,
        SitePredictorCNN,
        CombinedVAE_Predictors
    )
    print("✓ All main models can be imported")

def test_utils_import():
    """Test that core utility functions can be imported."""
    from Experiment_Utils.utils import (
        select_device,
        get_beta,
        kl_divergence_loss,
        vae_loss,
        calculate_r2_score
    )
    print("✓ Core utility functions can be imported")

def test_device_selection():
    """Test device selection functionality."""
    from Experiment_Utils.utils import select_device
    
    device = select_device()
    assert device is not None
    assert str(device) in ['cpu', 'cuda', 'mps']
    print(f"✓ Device selection works: {device}")

def test_model_creation():
    """Test that models can be instantiated."""
    from Experiment_Utils.models import Conv1DAutoencoder_fa, AgePredictorCNN
    
    # Test autoencoder creation
    model = Conv1DAutoencoder_fa(latent_dims=64, dropout=0.1)
    assert model is not None
    print("✓ Autoencoder model can be created")
    
    # Test age predictor creation
    age_model = AgePredictorCNN(input_channels=1, sequence_length=100, dropout=0.1)
    assert age_model is not None
    print("✓ Age predictor model can be created")

def test_beta_annealing():
    """Test beta annealing function."""
    from Experiment_Utils.utils import get_beta
    
    # Test beta values at different epochs
    beta_early = get_beta(current_epoch=50, total_epochs=1000, start_epoch=100)
    beta_mid = get_beta(current_epoch=500, total_epochs=1000, start_epoch=100)
    beta_late = get_beta(current_epoch=900, total_epochs=1000, start_epoch=100)
    
    assert beta_early == 0.0  # Should be 0 before start_epoch
    assert 0.0 < beta_mid < 1.0  # Should be between 0 and 1
    assert beta_late > beta_mid  # Should increase over time
    print("✓ Beta annealing works correctly")

def test_loss_functions():
    """Test loss function calculations."""
    from Experiment_Utils.utils import kl_divergence_loss, vae_loss
    
    # Create dummy tensors
    batch_size = 32
    latent_dim = 64
    
    mean = torch.randn(batch_size, latent_dim)
    logvar = torch.randn(batch_size, latent_dim)
    x = torch.randn(batch_size, 1, 100)
    x_hat = torch.randn(batch_size, 1, 100)
    
    # Test KL divergence
    kl_loss = kl_divergence_loss(mean, logvar)
    assert kl_loss is not None
    assert kl_loss.item() >= 0  # KL divergence should be non-negative
    
    # Test VAE loss (returns tuple: total_loss, recon_loss, kl_loss)
    total_loss, recon_loss, kl_loss = vae_loss(x, x_hat, mean, logvar, kl_weight=0.1)
    assert total_loss is not None
    assert total_loss.item() >= 0
    assert recon_loss.item() >= 0
    assert kl_loss.item() >= 0
    print("✓ Loss functions work correctly")

def test_r2_calculation():
    """Test R² score calculation."""
    from Experiment_Utils.utils import calculate_r2_score
    
    # Create dummy predictions (convert to torch tensors)
    y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = torch.tensor([1.1, 2.1, 2.9, 3.8, 5.2])
    
    r2 = calculate_r2_score(y_true, y_pred)
    assert r2 is not None
    assert isinstance(r2, float)
    assert r2 <= 1.0  # R² should be <= 1
    print(f"✓ R² calculation works: {r2:.3f}")

def test_pytorch_version():
    """Test that PyTorch version is adequate."""
    import torch
    version = torch.__version__
    major, minor = version.split('.')[:2]
    major, minor = int(major), int(minor)
    
    assert major >= 2 or (major == 1 and minor >= 10), f"PyTorch version {version} is too old"
    print(f"✓ PyTorch version is adequate: {version}")

if __name__ == "__main__":
    print("🧪 Running AFQ-Insight-Autoencoder-Experiments Installation Tests")
    print("="*60)
    
    test_models_import()
    test_utils_import()
    test_device_selection()
    test_model_creation()
    test_beta_annealing()
    test_loss_functions()
    test_r2_calculation()
    test_pytorch_version()
    
    print("="*60)
    print("🎉 All tests passed! Installation is working correctly.") 