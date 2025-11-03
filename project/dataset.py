import torch
import numpy as np
from torch.utils.data import Dataset

#TODO instead of applying transfomations in ipynb file pickle them and reuse them instead

class HousingDataset(Dataset):
    def __init__(self, dataframe, stats, target_col, log_cols_exclude, min_max_cols):
        self.stats = stats
        self.target_col = target_col
        
        df = dataframe.copy()
        
        # 1. Apply Log Transform 
        log_cols = [col for col in df.columns if col not in log_cols_exclude]
        
        for col in log_cols:
            df[col] = np.log1p(df[col])
            
        # 2. Apply Normalization - Min-Max cols 
        for col in min_max_cols:
            if col not in self.stats: continue # Skip if stats not present (e.g. target in test set)
            min_val = self.stats[col]['min']
            max_val = self.stats[col]['max']
            df[col] = (df[col] - min_val) / (max_val - min_val)
            
        # Z-Score cols
        feature_cols = [col for col in df.columns if col != self.target_col]
        z_score_cols = [col for col in feature_cols if col not in min_max_cols]

        for col in z_score_cols:
            mean_val = self.stats[col]['mean']
            std_val = self.stats[col]['std']
            # Add epsilon to prevent division by zero
            df[col] = (df[col] - mean_val) / (std_val + 1e-7)
        
        # --- 3. Store as Tensors ---
        self.y = torch.tensor(df[self.target_col].values, dtype=torch.float32)
        
        # Get all feature columns after processing
        self.X = torch.tensor(df[feature_cols].values, dtype=torch.float32)

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.y)

    def __getitem__(self, idx):
        # Unsqueeze the target to make its shape [1] instead of scalar
        return self.X[idx], self.y[idx].unsqueeze(0)

    def get_feature_count(self):
        return self.X.shape[1]