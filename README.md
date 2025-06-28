# AFQ-Insight-Autoencoder-Experiments

Experiments for Autoencoders developed on the AFQ-Insight repository. This repo contains experiment-specific utilities, models, and notebooks for analyzing diffusion MRI data using autoencoders.

## Project Structure

- `Experiment_Utils/`: Utility functions and model definitions
- `FC_AE_Experiments/`: Fully connected autoencoder experiments
  - `Non Variational/`: Standard autoencoders
  - `Variational/`: Variational autoencoders
- `ConvAE_Experiments/`: Convolutional autoencoder experiments
  - `Non Variational/`: Standard convolutional autoencoders
  - `Variational/`: Variational convolutional autoencoders

## Setup Instructions

### Environment Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/AFQ-Insight-Autoencoder-Experiments.git
   cd AFQ-Insight-Autoencoder-Experiments
   ```

2. Create and activate a virtual environment (conda recommended):
   ```
   conda create -n afq_experiments python=3.11
   conda activate afq_experiments
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

### Data Requirements

The experiments require AFQ tractometry data in a format compatible with the `afqinsight` package. The data should be organized according to the AFQ-Insight requirements.

You can learn more about the data format from the [AFQ-Insight documentation](https://yeatmanlab.github.io/AFQ-Insight/).

## Running Experiments

1. Start Jupyter Notebook:
   ```
   jupyter notebook
   ```

2. Navigate to the experiment directories and open the desired notebook:
   - `FC_AE_Experiments/` for fully connected autoencoders
   - `ConvAE_Experiments/` for convolutional autoencoders

3. Execute the notebook cells sequentially to run the experiments

## Customizing Experiments

To customize experiments:

1. Modify hyperparameters in the notebook cells
2. Change model architectures in the `Experiment_Utils/models.py` file
3. Add new data processing methods in `Experiment_Utils/utils.py`

## Requirements
s
See `requirements.txt` for a full list of dependencies.

## 

This repo is meant to work in combination with https://github.com/SamChou05/AFQ-Insight-Autoencoder-Plotting. Experiment resulst should be copied into the plotting repository to convert csv files into plots/graphs. Saved model weights can also be used in the plotting repository to visualize reconstructions and gauge performance visually. 