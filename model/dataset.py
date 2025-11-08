import torch
import numpy as np
from torch.utils.data import Dataset

import pandas as pd
from typing import List, Dict, Tuple

# Applying scaling and transformations will create Data Leakege
# if done on the whole dataset before splitting


class HousingDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        stats: Dict[str, Dict[str, float]],
        target_col: str,
        log_cols_exclude: List[str],
        min_max_cols: List[str],
    ) -> None:
        """
        Initializes the HousingDataset with transformations and scaling.

        Args:
        -----
            dataframe (pd.DataFrame): The input dataframe containing features and target.
            stats (dict): A dictionary containing precomputed statistics for scaling.
            target_col (str): The name of the target column.
            log_cols_exclude (list): List of columns to exclude from log transformation.
            min_max_cols (list): List of columns to apply Min-Max scaling.
        """
        self.stats = stats
        self.target_col = target_col

        df = dataframe.copy()

        log_cols = [col for col in df.columns if col not in log_cols_exclude]

        for col in log_cols:
            df[col] = np.log1p(df[col])

        for col in min_max_cols:
            if col not in self.stats:
                continue

            min_val = self.stats[col]["min"]
            max_val = self.stats[col]["max"]
            df[col] = (df[col] - min_val) / (max_val - min_val)

        feature_cols = [col for col in df.columns if col != self.target_col]
        z_score_cols = [col for col in feature_cols if col not in min_max_cols]

        for col in z_score_cols:
            mean_val = self.stats[col]["mean"]
            std_val = self.stats[col]["std"]
            df[col] = (df[col] - mean_val) / (std_val + 1e-7)

        self.y = torch.tensor(df[self.target_col].values, dtype=torch.float32)
        self.X = torch.tensor(df[feature_cols].values, dtype=torch.float32)

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx].unsqueeze(0)

    def get_feature_count(self) -> int:
        return self.X.shape[1]
