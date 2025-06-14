#!/usr/bin/env python3
"""
Test script to verify updated visualization functions work correctly.
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

def test_updated_visualization_functions():
    """Test that updated visualization functions work correctly."""
    
    print("Testing updated visualization function imports...")
    
    try:
        from utils import visualize_training_progress, save_debug_visualization
        print("✓ Successfully imported visualization functions from utils")
    except Exception as e:
        print(f"✗ Failed to import visualization functions: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Create test directory
    test_dir = "test_updated_viz"
    os.makedirs(test_dir, exist_ok=True)
    print(f"✓ Created test directory: {test_dir}")
    
    # Test save_debug_visualization
    print("\nTesting save_debug_visualization...")
    try:
        debug_success = save_debug_visualization(test_dir, 1, 'test_stage')
        if debug_success:
            print("✓ save_debug_visualization completed successfully")
        else:
            print("✗ save_debug_visualization failed")
            return False
    except Exception as e:
        print(f"✗ save_debug_visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
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
            stage_name='vae_individual'
        )
        print("✓ visualize_training_progress completed successfully")
        
        # Check if file was created
        expected_file = os.path.join(test_dir, 'training_progress', 'vae_individual_training_progress.png')
        if os.path.exists(expected_file):
            file_size = os.path.getsize(expected_file)
            print(f"✓ Training progress plot saved: {expected_file} ({file_size} bytes)")
        else:
            print(f"✗ Training progress plot not found: {expected_file}")
            return False
            
    except Exception as e:
        print(f"✗ visualize_training_progress failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test combined stage visualization
    print("\nTesting combined stage visualization...")
    try:
        combined_results = {
            'train_loss_epoch': [2.0, 1.8, 1.6, 1.4, 1.2],
            'val_loss_epoch': [2.1, 1.9, 1.7, 1.5, 1.3],
            'train_age_mae_epoch': [5.0, 4.5, 4.0, 3.5, 3.0],
            'val_age_mae_epoch': [5.2, 4.7, 4.2, 3.7, 3.2],
            'train_age_r2_epoch': [0.1, 0.2, 0.3, 0.4, 0.5],
            'val_age_r2_epoch': [0.08, 0.18, 0.28, 0.38, 0.48],
            'train_site_acc_epoch': [25, 30, 35, 40, 45],
            'val_site_acc_epoch': [23, 28, 33, 38, 43],
            'train_recon_loss_epoch': [1.5, 1.3, 1.1, 0.9, 0.7],
            'val_recon_loss_epoch': [1.6, 1.4, 1.2, 1.0, 0.8],
            'current_beta_epoch': [0.0, 0.2, 0.5, 0.8, 1.0],
            'current_grl_alpha_epoch': [0.0, 0.5, 1.0, 1.5, 2.0],
            'current_lr_epoch': [0.001, 0.0008, 0.0006, 0.0004, 0.0002]
        }
        
        visualize_training_progress(
            results_dict=combined_results,
            save_dir=test_dir,
            stage_name='combined_adversarial'
        )
        print("✓ Combined stage visualization completed successfully")
        
        # Check if file was created
        expected_file = os.path.join(test_dir, 'training_progress', 'combined_adversarial_training_progress.png')
        if os.path.exists(expected_file):
            file_size = os.path.getsize(expected_file)
            print(f"✓ Combined training progress plot saved: {expected_file} ({file_size} bytes)")
        else:
            print(f"✗ Combined training progress plot not found: {expected_file}")
            return False
            
    except Exception as e:
        print(f"✗ Combined stage visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # List all created files
    print(f"\nListing all files created in {test_dir}:")
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            rel_path = os.path.relpath(file_path, test_dir)
            print(f"  {rel_path} ({file_size} bytes)")
    
    print("\n" + "="*50)
    print("All updated visualization tests passed!")
    print("="*50)
    return True

if __name__ == "__main__":
    success = test_updated_visualization_functions()
    if success:
        print("\nUpdated visualization functions are working correctly.")
        sys.exit(0)
    else:
        print("\nUpdated visualization functions have issues.")
        sys.exit(1) 