import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_preprocessing_stats(dataframe, log_cols_exclude, min_max_cols, target_col):
    stats = {}
    
    df_log = dataframe.copy()
    
    log_cols = [col for col in df_log.columns if col not in log_cols_exclude]
    for col in log_cols:
        df_log[col] = np.log1p(df_log[col])
        
    feature_cols = [col for col in dataframe.columns if col != target_col]
    z_score_cols = [col for col in feature_cols if col not in min_max_cols]

    for col in min_max_cols:
        # Check if col is log
        data_source = df_log if col in log_cols else dataframe
        stats[col] = {
            'min': data_source[col].min(),
            'max': data_source[col].max()
        }
            
    # 4. Calculate stats for Z-Score columns 
    for col in z_score_cols:
        data_source = df_log if col not in log_cols_exclude else dataframe
        
        stats[col] = {
            'mean': data_source[col].mean(),
            'std': data_source[col].std()
        }
                
    return stats

def plot_loss_curves(history):
    sns.set_style("whitegrid")

    train_loss = history.get('train_loss', [])
    test_loss = history.get('test_loss', [])
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 6))
    
    plt.plot(epochs, train_loss, 'b-o', label='Train Loss', markersize=4)
    plt.plot(epochs, test_loss, 'r-o', label='Validation Loss', markersize=4)
    
    plt.title('Training and Validation Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    plt.grid(True)
    plt.show()