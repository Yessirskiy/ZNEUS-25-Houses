# File: tools/train.py

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import sys
import os
from tqdm import tqdm

# Path to import from project files
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from project.dataset import HousingDataset
from project.model import RegressionMLP
from project.utils import get_preprocessing_stats, plot_loss_curves

# 1. Configuration 
DATA_PATH = os.path.join(project_root, "houses.csv")
TARGET_COL = "median_house_value"

LOG_COLS_EXCLUDE = ['latitude', 'longitude']
MIN_MAX_COLS = ['median_house_value', 'housing_median_age']

# Hyperparameters
TEST_SIZE = 0.2
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 50
# RANDOM_STATE = 420

def main():
    # 2. Load Data 
    raw_df = pd.read_csv(DATA_PATH)

    
    # 3. Train/Test Split
    train_df, test_df = train_test_split(
        raw_df, 
        test_size=TEST_SIZE, 
        # random_state=RANDOM_STATE
    )
    
    # 4. Get Preprocessing Stats 
    print("Calculating statistics ...")
    preprocessing_stats = get_preprocessing_stats(
        train_df,
        log_cols_exclude=LOG_COLS_EXCLUDE,
        min_max_cols=MIN_MAX_COLS,
        target_col=TARGET_COL
    )
    
    # 5. Create Datasets and DataLoaders 
    print("Creating datasets and dataloaders...")
    train_dataset = HousingDataset(
        dataframe=train_df,
        stats=preprocessing_stats,
        target_col=TARGET_COL,
        log_cols_exclude=LOG_COLS_EXCLUDE,
        min_max_cols=MIN_MAX_COLS
    )
    test_dataset = HousingDataset(
        dataframe=test_df,
        stats=preprocessing_stats,
        target_col=TARGET_COL,
        log_cols_exclude=LOG_COLS_EXCLUDE,
        min_max_cols=MIN_MAX_COLS
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )

    # 6. Model, Loss, Optimizer 
    input_features = train_dataset.get_feature_count()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using: {device} device")
    
    model = RegressionMLP(input_features=input_features).to(device)
    loss_f = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 7. Training Loop 
    print("Starting training...")

    history = {
        'train_loss': [],
        'test_loss': []
    }

    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Train]", leave=False)
        
        for features, target in train_pbar:
            features, target = features.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = loss_f(outputs, target)
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            train_pbar.set_postfix(batch_loss=f"{loss.item():.6f}")
        
        # 8. Val Loop 
        model.eval()
        test_loss_sum = 0.0
        
        test_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Test]", leave=False)

        with torch.no_grad():
            for features, target in test_pbar:
                features, target = features.to(device), target.to(device)
                outputs = model(features)
                loss = loss_f(outputs, target)
                test_loss_sum += loss.item()
                test_pbar.set_postfix(batch_loss=f"{loss.item():.6f}")

        avg_train_loss = train_loss_sum / len(train_loader)
        avg_test_loss = test_loss_sum / len(test_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)

        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] - "
              f"Train Loss (MSE): {avg_train_loss:.6f}, "
              f"Test Loss (MSE): {avg_test_loss:.6f}")
    
    print("Training finished.")

    plot_loss_curves(history)

if __name__ == "__main__":
    main()