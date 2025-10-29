import torch
import numpy as np
from torch.utils.data import Dataset


class HousingDataset(Dataset):
    def __init__(self, dataframe, stats, target_col='median_house_value'):
        self.stats = stats
        self.target_col = target_col
        
        # Deep copy to avoid modifying the original dataframe in place
        df = dataframe.copy()
        
        # 1. Apply Log Transform - All except lat/lon
        log_cols = [col for col in df.columns if col not in ['latitude', 'longitude']]
        
        for col in log_cols:
            df[col] = np.log1p(df[col])
            
        # 2. Apply Normalization - Min-Max cols
        min_max_cols = ['median_house_value', 'housing_median_age']
        for col in min_max_cols:
            min_val = self.stats[col]['min']
            max_val = self.stats[col]['max']
            # Epsilon for math. stability  (division by zero if min == max)
            df[col] = (df[col] - min_val) / (max_val - min_val + 1e-7)
            
        # Z-Score cols
        feature_cols = [col for col in df.columns if col != self.target_col]
        z_score_cols = [
            col for col in feature_cols if col not in ['housing_median_age']
        ]

        for col in z_score_cols:
            mean_val = self.stats[col]['mean']
            std_val = self.stats[col]['std']
            # Add epsilon to prevent division by zero
            df[col] = (df[col] - mean_val) / (std_val + 1e-7)
        
        # 3. Store as Tensors 
        self.y = torch.tensor(df[self.target_col].values, dtype=torch.float32)
        
        # Get all feature columns after processing
        self.X = torch.tensor(df[feature_cols].values, dtype=torch.float32)

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.y)

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        Returns features and target as a tuple.
        """
        return self.X[idx], self.y[idx].unsqueeze(0) # Unsqueeze to make target shape (1,)