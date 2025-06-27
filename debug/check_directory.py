import os
import sys

print("Script starting...")
print(f"Current working directory: {os.getcwd()}")
print(f"Script path: {__file__}")
print(f"Python version: {sys.version}")
print(f"System path: {sys.path}")

# Check if Experiment_Utils is accessible
sys.path.insert(1, '/mmfs1/gscratch/nrdg/samchou/AFQ-Insight-Autoencoder-Experiments/Experiment_Utils')
try:
    print("\nTrying to import from Experiment_Utils...")
    from utils import select_device
    print("Successfully imported select_device from utils")
except Exception as e:
    print(f"Error importing from Experiment_Utils: {str(e)}")
    import traceback
    print(traceback.format_exc())

# Check directory structure
print("\nChecking directory structure:")
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
experiment_utils_dir = os.path.join(os.path.dirname(os.path.dirname(parent_dir)), "Experiment_Utils")

print(f"Current script directory: {current_dir}")
print(f"Parent directory: {parent_dir}")
print(f"Expected Experiment_Utils path: {experiment_utils_dir}")
print(f"Experiment_Utils exists: {os.path.exists(experiment_utils_dir)}")

# List files in current directory
print("\nFiles in current directory:")
for f in os.listdir(current_dir):
    print(f"  - {f}")

# Try to import torch 
try:
    print("\nTrying to import PyTorch...")
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"Error importing PyTorch: {str(e)}")

print("\nScript completed successfully!") 