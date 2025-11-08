import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Dict, Optional, Type
from pathlib import Path

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb as wandbb
from tqdm import tqdm

from model.dataset import HousingDataset

LOSS_PICKER: Dict[str, nn.MSELoss | nn.L1Loss] = {
    "MSE": nn.MSELoss,
    "MAE": nn.L1Loss,
}
OPTIMIZER_PICKER: Dict[str, torch.optim.Optimizer] = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
    "RMSprop": torch.optim.RMSprop,
}


def train(
    data_path: Path,
    model: Type[nn.Module],
    train_size: float,
    val_size: float,
    test_size: float,
    epochs: int,
    batch_size: int,
    hidden_units: int,
    learning_rate: float,
    early_stopping: bool,
    loss_function: str,
    optimizer: str,
    wandb_name: Optional[str] = None,
    wandb: bool = False,
    random_state: int = 42,
    shuffle: bool = True,
    debug: bool = False,
):
    TARGET_COL: str = "median_house_value"
    LOG_COLS_EXCLUDE: List[str] = ["latitude", "longitude"]
    MIN_MAX_COLS: List[str] = ["median_house_value", "housing_median_age"]

    if debug:
        print("[0]\tStarting training with the following parameters:")
        print(f"\tSplit ratios (train/val/test): {train_size}/{val_size}/{test_size}")
        print(f"\tEpochs: {epochs}")
        print(f"\tBatch size: {batch_size}")
        print(f"\tLearning rate: {learning_rate}")
        print(f"\tEarly stopping: {early_stopping}")
        print(f"\tLoss function: {loss_function}")

    raw_df = pd.read_csv(data_path)
    df_train, df_temp = train_test_split(
        raw_df, test_size=(1 - train_size), random_state=random_state, shuffle=shuffle
    )
    relative_test_ratio = test_size / (val_size + test_size)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=relative_test_ratio,
        random_state=random_state,
        shuffle=shuffle,
    )
    if debug:
        print(
            f"[1]\tDF read & split (train/val/test): {len(df_train)}/{len(df_val)}/{len(df_test)}"
        )

    train_stats = get_preprocessing_stats(
        df_train,
        log_cols_exclude=LOG_COLS_EXCLUDE,
        min_max_cols=MIN_MAX_COLS,
        target_col=TARGET_COL,
    )
    val_stats = get_preprocessing_stats(
        df_val,
        log_cols_exclude=LOG_COLS_EXCLUDE,
        min_max_cols=MIN_MAX_COLS,
        target_col=TARGET_COL,
    )
    test_stats = get_preprocessing_stats(
        df_test,
        log_cols_exclude=LOG_COLS_EXCLUDE,
        min_max_cols=MIN_MAX_COLS,
        target_col=TARGET_COL,
    )
    if debug:
        print("[2]\tPreprocessing statistics calculated.")

    train_dataset = HousingDataset(
        dataframe=df_train,
        stats=train_stats,
        target_col=TARGET_COL,
        log_cols_exclude=LOG_COLS_EXCLUDE,
        min_max_cols=MIN_MAX_COLS,
    )
    val_dataset = HousingDataset(
        dataframe=df_val,
        stats=val_stats,
        target_col=TARGET_COL,
        log_cols_exclude=LOG_COLS_EXCLUDE,
        min_max_cols=MIN_MAX_COLS,
    )
    test_dataset = HousingDataset(
        dataframe=df_test,
        stats=test_stats,
        target_col=TARGET_COL,
        log_cols_exclude=LOG_COLS_EXCLUDE,
        min_max_cols=MIN_MAX_COLS,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if debug:
        print("[3]\tDatasets and DataLoaders created.")

    input_features = train_dataset.get_feature_count()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model_final = model(input_features=input_features, hidden_units=hidden_units).to(
        device
    )
    loss_final = LOSS_PICKER[loss_function]()
    optimizer_final = OPTIMIZER_PICKER[optimizer](
        model_final.parameters(), lr=learning_rate
    )

    if debug:
        print("[4]\tModel, loss and optimizer created.")

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0
    PATIENCE = 10

    if wandb:
        run = wandbb.init(
            project="zneus_part_2",
            name=wandb_name,
            config={
                "learning_rate": learning_rate,
                "epochs": epochs,
                "batch_size": batch_size,
                "model": model.NAME,
                "loss_function": loss_function,
                "optimizer": optimizer,
            },
        )

    print("Starting training loop...")
    for epoch in range(epochs):
        model_final.train()
        train_loss_sum = 0.0
        for features, target in train_loader:
            features, target = features.to(device), target.to(device)

            optimizer_final.zero_grad()
            outputs = model_final(features)
            loss = loss_final(outputs, target)
            loss.backward()
            optimizer_final.step()

            train_loss_sum += loss.item()
        avg_train_loss = train_loss_sum / len(train_loader)

        model_final.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for features, target in val_loader:
                features, target = features.to(device), target.to(device)
                outputs = model_final(features)
                loss = loss_final(outputs, target)
                val_loss_sum += loss.item()
        avg_val_loss = val_loss_sum / len(val_loader)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        if debug:
            print(
                f"Epoch [{epoch+1:02d}/{epochs}] - "
                f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
            )

        if wandb:
            wandbb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                }
            )

        if early_stopping:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model_final.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    if debug:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                    model_final.load_state_dict(best_model_state)
                    break

    print("Training complete. Evaluating on test set...")
    model_final.eval()
    test_loss_sum = 0.0
    with torch.no_grad():
        for features, target in test_loader:
            features, target = features.to(device), target.to(device)
            outputs = model_final(features)
            loss = loss_final(outputs, target)
            test_loss_sum += loss.item()
    avg_test_loss = test_loss_sum / len(test_loader)
    if wandb:
        wandbb.log({"test_loss": avg_test_loss})
        wandbb.finish()
    if debug:
        print(f"Final Test Loss: {avg_test_loss:.6f}")

    return model_final, history, avg_test_loss


def get_preprocessing_stats(
    dataframe: pd.DataFrame,
    log_cols_exclude: List[str],
    min_max_cols: List[str],
    target_col: str,
) -> Dict[str, Dict[str, float]]:

    stats: Dict[str, Dict[str, float]] = {}
    df_log = dataframe.copy()

    log_cols = [col for col in df_log.columns if col not in log_cols_exclude]
    for col in log_cols:
        df_log[col] = np.log1p(df_log[col])

    feature_cols = [col for col in dataframe.columns if col != target_col]
    z_score_cols = [col for col in feature_cols if col not in min_max_cols]

    for col in min_max_cols:
        data_source = df_log if col in log_cols else dataframe
        stats[col] = {"min": data_source[col].min(), "max": data_source[col].max()}

    for col in z_score_cols:
        data_source = df_log if col not in log_cols_exclude else dataframe

        stats[col] = {"mean": data_source[col].mean(), "std": data_source[col].std()}

    return stats


def plot_loss_curves(history: Dict[str, List[float]]) -> None:
    sns.set_style("whitegrid")

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(epochs, train_loss, "b-o", label="Train Loss", markersize=4)
    plt.plot(epochs, val_loss, "r-o", label="Validation Loss", markersize=4)

    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()
    plt.grid(True)
    plt.show()
