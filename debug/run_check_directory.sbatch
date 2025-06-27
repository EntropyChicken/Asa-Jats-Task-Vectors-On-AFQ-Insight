#!/usr/bin/env bash

#SBATCH --job-name=check-dir
#SBATCH --account=escience
#SBATCH --partition=gpu-a40
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=check_dir_output.log
#SBATCH --error=check_dir_error.log

module load cuda/11.8

source /mmfs1/gscratch/nrdg/samchou/conda/etc/profile.d/conda.sh
conda activate afq_new

echo "=== Environment Check ==="
echo "Hostname: $(hostname)"
echo "Current directory: $(pwd)"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

echo -e "\n=== Directory Structure Check ==="
# Check if the directory exists
ls -la /mmfs1/gscratch/nrdg/samchou/AFQ-Insight-Autoencoder-Experiments/debug

echo -e "\n=== Running Check Script ==="
/gscratch/nrdg/samchou/conda/envs/afq_new/bin/python /mmfs1/gscratch/nrdg/samchou/AFQ-Insight-Autoencoder-Experiments/debug/check_directory.py 