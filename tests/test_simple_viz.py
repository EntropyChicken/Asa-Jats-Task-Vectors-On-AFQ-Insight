#!/usr/bin/env python3
#Simple test to run visualization functions

import torch
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

def simple_visualize_training_progress(results_dict, save_dir, stage_name):
    """
    Create comprehensive training progress visualizations.
    """
    import os
    import matplotlib.pyplot as plt
    
    viz_dir = os.path.join(save_dir, 'training_progress')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Plot different metrics based on what's available
    if stage_name == 'vae_individual':
        # VAE-specific plots
        epochs = range(1, len(results_dict.get('train_loss_epoch', [])) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        if 'train_loss_epoch' in results_dict and 'val_loss_epoch' in results_dict:
            axes[0, 0].plot(epochs, results_dict['train_loss_epoch'], 'b-', label='Train')
            axes[0, 0].plot(epochs, results_dict['val_loss_epoch'], 'r-', label='Validation')
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Reconstruction loss
        if 'train_recon_loss_epoch' in results_dict and 'val_recon_loss_epoch' in results_dict:
            axes[0, 1].plot(epochs, results_dict['train_recon_loss_epoch'], 'b-', label='Train')
            axes[0, 1].plot(epochs, results_dict['val_recon_loss_epoch'], 'r-', label='Validation')
            axes[0, 1].set_title('Reconstruction Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # KL loss
        if 'train_kl_loss_epoch' in results_dict and 'val_kl_loss_epoch' in results_dict:
            axes[1, 0].plot(epochs, results_dict['train_kl_loss_epoch'], 'b-', label='Train')
            axes[1, 0].plot(epochs, results_dict['val_kl_loss_epoch'], 'r-', label='Validation')
            axes[1, 0].set_title('KL Divergence Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('KL Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Beta annealing
        if 'current_beta_epoch' in results_dict:
            axes[1, 1].plot(epochs, results_dict['current_beta_epoch'], 'g-')
            axes[1, 1].set_title('KL Weight (Beta) Annealing')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Beta Value')
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'{stage_name}_training_progress.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved training progress visualization for {stage_name}")

def test_simple_visualization():
    """Test simple visualization function."""
    
    print("Testing simple visualization function...")
    
    # Create test directory
    test_dir = "test_simple_viz"
    os.makedirs(test_dir, exist_ok=True)
    print(f"✓ Created test directory: {test_dir}")
    
    # Test visualize_training_progress with dummy data
    print("\nTesting simple_visualize_training_progress...")
    try:
        dummy_results = {
            'train_loss_epoch': [1.0, 0.8, 0.6, 0.4, 0.2],
            'val_loss_epoch': [1.1, 0.9, 0.7, 0.5, 0.3],
            'train_recon_loss_epoch': [0.8, 0.6, 0.4, 0.2, 0.1],
            'val_recon_loss_epoch': [0.9, 0.7, 0.5, 0.3, 0.2],
            'train_kl_loss_epoch': [0.2, 0.2, 0.2, 0.2, 0.1],
            'val_kl_loss_epoch': [0.2, 0.2, 0.2, 0.2, 0.1],
            'current_beta_epoch': [0.0, 0.1, 0.5, 0.8, 1.0]
        }
        
        simple_visualize_training_progress(
            results_dict=dummy_results,
            save_dir=test_dir,
            stage_name='vae_individual'
        )
        print("✓ simple_visualize_training_progress completed successfully")
        
        # Check if file was created
        expected_file = os.path.join(test_dir, 'training_progress', 'vae_individual_training_progress.png')
        if os.path.exists(expected_file):
            print(f"✓ Training progress plot saved: {expected_file}")
            file_size = os.path.getsize(expected_file)
            print(f"✓ File size: {file_size} bytes")
        else:
            print(f"✗ Training progress plot not found: {expected_file}")
            return False
            
    except Exception as e:
        print(f"✗ simple_visualize_training_progress failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*50)
    print("Simple visualization test passed!")
    print("="*50)
    return True

if __name__ == "__main__":
    success = test_simple_visualization()
    if success:
        print("\nSimple visualization function is working correctly.")
    else:
        print("\nSimple visualization function has issues.") 