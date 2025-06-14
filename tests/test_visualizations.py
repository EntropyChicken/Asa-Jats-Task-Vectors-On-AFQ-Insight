#!/usr/bin/env python3
"""
Test script to verify visualization functions work correctly.
"""

import torch
import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add the utils path
sys.path.insert(0, 'Experiment_Utils')

def test_visualization_functions():
    """Test that visualization functions can be imported and run."""
    
    print("Testing visualization function imports...")
    
    try:
        from utils import visualize_vae_samples, visualize_training_progress
        print("✓ Successfully imported visualization functions from utils")
    except Exception as e:
        print(f"✗ Failed to import visualization functions: {e}")
        return False
    
    # Create test directory
    test_dir = "test_visualizations"
    os.makedirs(test_dir, exist_ok=True)
    print(f"✓ Created test directory: {test_dir}")
    
    # Test visualize_training_progress with dummy data
    print("\nTesting visualize_training_progress...")
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
        
        visualize_training_progress(
            results_dict=dummy_results,
            save_dir=test_dir,
            stage_name='test_vae'
        )
        print("✓ visualize_training_progress completed successfully")
        
        # Check if file was created
        expected_file = os.path.join(test_dir, 'training_progress', 'test_vae_training_progress.png')
        if os.path.exists(expected_file):
            print(f"✓ Training progress plot saved: {expected_file}")
        else:
            print(f"✗ Training progress plot not found: {expected_file}")
            
    except Exception as e:
        print(f"✗ visualize_training_progress failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\nTesting basic matplotlib functionality...")
    try:
        # Simple test plot
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        ax.set_title('Test Plot')
        test_plot_path = os.path.join(test_dir, 'test_plot.png')
        plt.savefig(test_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if os.path.exists(test_plot_path):
            print(f"✓ Basic matplotlib test successful: {test_plot_path}")
        else:
            print(f"✗ Basic matplotlib test failed - file not created")
            
    except Exception as e:
        print(f"✗ Basic matplotlib test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*50)
    print("All visualization tests passed!")
    print("="*50)
    return True

if __name__ == "__main__":
    success = test_visualization_functions()
    if success:
        print("\nVisualization functions are working correctly.")
        sys.exit(0)
    else:
        print("\nVisualization functions have issues.")
        sys.exit(1) 