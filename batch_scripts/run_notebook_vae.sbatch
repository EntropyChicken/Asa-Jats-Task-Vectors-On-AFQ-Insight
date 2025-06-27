#!/usr/bin/env bash

#SBATCH --job-name=vsc-proxy-jump
#SBATCH --account=escience
#SBATCH --partition=gpu-a40
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=24:00:00

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

# (Optional) Output logs
# #SBATCH --output=my_output.log
# #SBATCH --error=my_error.log

#Testing Variational autoencoder on FA data, flattened to one channel
#Tests permutations of latent dimensions and dropout rates

module load cuda/11.8

source /mmfs1/gscratch/nrdg/samchou/conda/etc/profile.d/conda.sh
conda activate afq_new

which python
nvidia-smi

python -c "import torch; print(torch.cuda.is_available())"

echo "Hostname: $(hostname)"
echo "Starting time: $(date)"
echo "CUDA devices visible: $CUDA_VISIBLE_DEVICES"

echo "Starting program"

/gscratch/nrdg/samchou/conda/envs/afq_new/bin/python /mmfs1/gscratch/nrdg/samchou/AFQ-Insight-Autoencoder-Experiments/ConvAE_Experiments/Variational/fa_tracts_data/vaeconv_combined_fa_flattened.py vaeconv_fa_output.py \
    --log-output \
    --log-level INFO

echo "Finished time: $(date)"