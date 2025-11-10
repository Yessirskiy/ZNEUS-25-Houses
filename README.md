# ZNEUS-25-Houses

<b>dataset</b>: https://api.openml.org/d/537


This project provides a modular PyTorch pipeline for regression tasks, demonstrated on a housing price prediction dataset. It features a command-line interface (CLI), multiple model architectures, configurable preprocessing, and full integration with Weights & Biases (W&B) for experiment tracking and hyperparameter sweeps.

---

## Documentation
The wandb experiments are present as Experiments-1 and Experiments-2. There are doc-strings in the .ipynb and python are also present.

## Project Structure

The project follows a modular structure to separate concerns:
```
project_root/ 
  ├── main.py # Main CLI entry point for training 
  ├── sweep.py # Entry point for W&B hyperparameter sweeps 
  ├── tools.py # Helper functions (e.g., dynamic model loading) 
  ├── houses.csv # The dataset file 
  └── model/ 
    ├── dataset.py # Contains the PyTorch HousingDataset class 
    ├── model.py # Defines model architectures 
    └── utils.py # Core logic: train() function, helpers, custom losses
```

---

## Installation

1.  **Clone the repository** (if applicable).
2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv_test
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    
    ```
    pip install -r requirements.txt
    ```

---

## Usage

The primary way to use this project is via the `main.py` CLI.

### Training a Model

All training is initiated from `main.py` using the `train` command. You can configure all aspects of the run, from data splitting and model architecture to hyperparameters and logging.

```
usage: main.py train [-h] --dataset DATASET [--model MODEL] [--split-ratio TRAIN VAL TEST] [--batch-size BATCH_SIZE] [--hidden-units HIDDEN_UNITS]
                     [--learning-rate LEARNING_RATE] [--epochs EPOCHS] [--early-stopping] [--loss-function {MAE,MSE,RMSE,R2}] [--optimizer {SGD,Adam,RMSprop}] [--wandb]
                     [--wandb-name WANDB_NAME]

options:
  -h, --help            show this help message and exit
  --dataset DATASET     Relative path to the dataset file or folder
  --model MODEL         Model architecture to use (default: RegressionMLP; check model/model.py for options)
  --split-ratio TRAIN VAL TEST
                        Split ratios for train/validation/test sets (default: 0.6 0.2 0.2)
  --batch-size BATCH_SIZE
                        Batch size for training (default: 64)
  --hidden-units HIDDEN_UNITS
                        Number of hidden units in the model (default: 64)
  --learning-rate LEARNING_RATE
                        Learning rate (default: 0.01)
  --epochs EPOCHS       Number of training epochs (default: 300)
  --early-stopping      Enable early stopping (default: False)
  --loss-function {MAE,MSE,RMSE,R2}
                        Loss function to use (default: MAE). Options: MAE (Mean Absolute Error), MSE (Mean Squared Error), RMSE (Root Mean Squared Error), R2 (R-squared)
  --optimizer {SGD,Adam,RMSprop}
                        Optimizer to use (default: Adam)
  --wandb               Enable Weights & Biases logging (default: True)
  --wandb-name WANDB_NAME
                        Optional custom W&B run name
```

## Examples: Metrics from wandb
### Optimizer comparison

<img width="1103" height="299" alt="Screenshot 2025-11-09 at 19 00 11" src="https://github.com/user-attachments/assets/b5c68251-a94d-4d46-b51c-d6e766c71083" />
In the hyperparameter sweep, where all models used the MSE loss function, comparing RMSprop, Adam, and SGD optimizers. The RMSprop optimizer was the clear winner, as all three of its configurations (using lr=0.001) achieved the lowest test_loss and val_loss values, clustering around 0.005. The Adam model (lr=0.0001) was the next best performer, while the SGD models performed significantly worse; notably, the lr_0.0001_bs_32_hu_128_SGD_MSE run (red bar) yielded the highest (worst) loss by a substantial margin, indicating a poor parameter choice for that optimizer.
