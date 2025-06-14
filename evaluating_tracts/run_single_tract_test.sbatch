#!/usr/bin/env bash

#SBATCH --job-name=single-tract-test
#SBATCH --account=escience
#SBATCH --partition=gpu-a40
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=2:00:00

# -- Email settings --
# Mail events (ALL, BEGIN, END, FAIL, REQUEUE, etc.).
#SBATCH --mail-type=END,FAIL

# Replace with your actual email:
#SBATCH --mail-user=samobear@uw.edu

# Output logs
#SBATCH --output=single_tract_test.log
#SBATCH --error=single_tract_test.err

#For testing GPU and CONDA setup quickly, not for running the actual script

# Load CUDA
module load cuda/11.8

# Force TensorFlow to use CPU only, while PyTorch can use GPU
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
# These environment variables prevent TensorFlow from using GPU
export TF_VISIBLE_DEVICES=-1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2

# Activate conda environment - update this path to your conda installation
source /mmfs1/gscratch/nrdg/samchou/conda/etc/profile.d/conda.sh
conda activate afq_new

# Print environment info
which python
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

echo "Hostname: $(hostname)"
echo "Starting time: $(date)"
echo "CUDA devices visible: $CUDA_VISIBLE_DEVICES"

# Verify PyTorch can see and use the GPU
python -c "import torch; print('PyTorch CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Update these paths to your actual installation paths
SCRIPT_PATH="/mmfs1/gscratch/nrdg/samchou/AFQ-Insight-Autoencoder-Experiments/evaluating_tracts/tract_importance_evaluation.py"
CONDA_PYTHON="/gscratch/nrdg/samchou/conda/envs/afq_new/bin/python"

# Define parameters for testing a single tract (0)
OUTPUT_DIR="tract_importance_test"
EPOCHS_STAGE1=20  # Reduced epochs for testing
EPOCHS_STAGE2=20  # Reduced epochs for testing
BATCH_SIZE=128
LEARNING_RATE=0.001
USE_BOTH_FA_MD="--use-both-fa-md"  # Remove this flag to use FA only
START_TRACT=0    # Just test the first tract
END_TRACT=0

echo "Starting Single Tract Test"

# Run the script with test parameters
$CONDA_PYTHON $SCRIPT_PATH \
  --output-dir $OUTPUT_DIR \
  --epochs-stage1 $EPOCHS_STAGE1 \
  --epochs-stage2 $EPOCHS_STAGE2 \
  --batch-size $BATCH_SIZE \
  --learning-rate $LEARNING_RATE \
  --start-tract $START_TRACT \
  --end-tract $END_TRACT \
  $USE_BOTH_FA_MD

echo "Finished time: $(date)"