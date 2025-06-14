#!/usr/bin/env bash

#SBATCH --job-name=vae-first-tract
#SBATCH --account=escience
#SBATCH --partition=gpu-a40
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=8:00:00

# -- Email settings --
# Mail events (ALL, BEGIN, END, FAIL, REQUEUE, etc.).
# Common picks:
#   END = email when job finishes successfully
#   FAIL = email if job fails
#   BEGIN = email when job starts
#   ALL = all of the above and more
#SBATCH --mail-type=END,FAIL

# Replace with your actual UW email:
#SBATCH --mail-user=samobear@uw.edu

# Output logs
#SBATCH --output=all_tracts_output.log
#SBATCH --error=all_tracts_error.log

module load cuda/11.8

# Force TensorFlow to use CPU only, while PyTorch can use GPU
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
# These environment variables prevent TensorFlow from using GPU
export TF_VISIBLE_DEVICES=-1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2

source /mmfs1/gscratch/nrdg/samchou/conda/etc/profile.d/conda.sh
conda activate afq_new

which python
nvidia-smi

python -c "import torch; print(torch.cuda.is_available())"

echo "Hostname: $(hostname)"
echo "Starting time: $(date)"
echo "CUDA devices visible: $CUDA_VISIBLE_DEVICES"

# Verify PyTorch can see and use the GPU
python -c "import torch; print('PyTorch CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo "Starting FA UNFLATTENED Tract Staged Experiment with Confusion Matrix Analysis"

/gscratch/nrdg/samchou/conda/envs/afq_new/bin/python /mmfs1/gscratch/nrdg/samchou/AFQ-Insight-Autoencoder-Experiments/ConvAE_Experiments/Variational/all_tracts/vae_age_site_stages_fa.py

echo "Finished time: $(date)"
echo "Results include site prediction confusion matrices for adversarial training analysis"