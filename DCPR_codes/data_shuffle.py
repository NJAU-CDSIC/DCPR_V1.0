# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 14:56:11 2023

@author: hx
"""



import numpy as np
import pandas as pd
from pathlib import Path

def shuffle_time_series(data, times, out_dir):
    """
    Shuffle time-series gene expression data while keeping time-gene pairs intact
    
    Args:
        data: 2D array (genes x timepoints)  
        times: 1D array of timepoints
        out_dir: Path to save results
        
    Returns:
        (shuffled_genes, shuffled_times)
    """
    np.random.seed(42)  
    
    norm_times = times
    if times is not None:
        norm_times = np.array(times) % 24.0 

    if data.columns[0] == 'symbol' or not np.issubdtype(data.dtypes[0], np.number):
        data0 = data.iloc[:,1:]
    else:
        data0 = data
        
    n = data0.shape[1]
    
    shuffle_order = np.random.permutation(n)

    shuffled_genes = data0.values[:, shuffle_order]
    symbol_col = data['symbol'].values if 'symbol' in data.columns else None
    shuffled_genes = pd.DataFrame(np.column_stack((symbol_col, shuffled_genes)))
    shuffled_genes.columns = ['symbol'] + [f'col_{i}' for i in range(1, shuffled_genes.shape[1])]
    shuffled_times = norm_times[shuffle_order]
    orig_times_shuffled = times.values[shuffle_order]
    
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    
    return pd.DataFrame(shuffled_genes), shuffled_times



