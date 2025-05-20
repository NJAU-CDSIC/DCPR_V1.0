# -*- coding: utf-8 -*-
"""
Time-series Gene Expression Preprocessing Pipeline

This script performs comprehensive preprocessing of time-series gene expression data,
including dimensionality reduction, period detection, and temporal alignment. 
The pipeline is designed for circadian rhythm analysis and other periodic biological processes.

Key Features:
1. Data standardization 
2. PCA-based temporal ordering
3. Constructing a reference time series
4. Cosinor regression for periodic pattern detection
5. Matrix reconstruction and error analysis

@author: hx
"""
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler
from CosinorPy import cosinor
import matplotlib.pyplot as plt
from DCPR_codes.data_shuffle import shuffle_time_series


def get_seed_gene_data(matrix, seed_gene_table):
    """
    Extracts the expression matrix corresponding to seed genes for downstream training.

    Parameters:
    ----------
    matrix : pandas.DataFrame
        The full gene expression matrix (rows: genes, columns: samples or time points).
    
    seed_gene_table : pandas.DataFrame
        A data frame containing a list of seed genes, with gene symbols in the column 'human Symbol'.
    
    Returns:
    -------
    numpy.ndarray
        A matrix of expression values corresponding only to seed genes.
    """
    if seed_gene_table is not None and not seed_gene_table.empty:
    
        
        candidate_cols = ['human Symbol', 'Rat Symbol']

        existing_cols = [col for col in candidate_cols if col in seed_gene_table.columns]
        
        if existing_cols:
            seed_gene_table = seed_gene_table.dropna(subset=[existing_cols[0]])
        else:
            raise ValueError("No known gene symbol column found.")

        seed_genes = seed_gene_table['human Symbol']
        gene_annotation = matrix[['symbol']]

        matched_genes = gene_annotation[gene_annotation['symbol'].isin(seed_genes)]
        matched_indices = list(matched_genes.index)

        filtered_expression = matrix.iloc[matched_indices, :].values[:,1:]

    else:

        filtered_expression = matrix.values[:,1:]
    filtered_expression = np.array(filtered_expression, dtype=np.float32)

    return filtered_expression


def detect_noise_structure(matrix):
    """
    Detect if the matrix has significant noise structure by comparing pairwise MSE.
    
    Args:
        matrix (ndarray): Input gene expression matrix (genes x timepoints)
        
    Returns:
        tuple: (indicator, max_value, max_value_indices)
            indicator: 0 for non-noise, 1 for noise
            max_value: maximum zero count in modified MSE matrix
            max_value_indices: indices of max_value
    """
    mse_all = np.zeros((matrix.shape[1], matrix.shape[1]))
    mse_all_zero = np.zeros((matrix.shape[1], matrix.shape[1]))
    
    for i in range(matrix.shape[1]):
        reference = matrix[:, i]
        for j in range(matrix.shape[1]):
            reference_1 = matrix[:, j]
            mse_mean = np.mean(np.square(reference - reference_1))
            mse_all[i, j] = mse_mean
            mse = np.square(reference - reference_1)
            count_of_zeros = len(mse) - np.count_nonzero(mse)
            mse_all_zero[i, j] = count_of_zeros
    
    modified_mse_all_zero = mse_all_zero.copy()
    np.fill_diagonal(modified_mse_all_zero, 0)
    
    max_value = np.max(modified_mse_all_zero)
    max_value_index = np.unravel_index(
        np.argmax(modified_mse_all_zero, axis=None), 
        modified_mse_all_zero.shape
    )
    
    ind_0, ind_1 = max_value_index
    indicator = 0 if (ind_0 != ind_1 and max_value > 10) else 1
    
    noise_status = "nonoise" if indicator == 0 else "noise"
    
    return indicator, max_value, max_value_index


def standardize_expression_data(matrix, large_sample_threshold=2000):
    """
    Standardize gene expression data using sin transform and variance normalization.
    
    Args:
        matrix (ndarray): Input gene expression matrix
        large_sample_threshold (int): Threshold for considering as large sample
        
    Returns:
        ndarray: Standardized matrix
    """
    def standardize(X):
        mean_vals = np.mean(X, axis=0)
        std_dev = np.std(X, axis=0)
        std_dev[std_dev == 0] = 1.0
        return (X - mean_vals) / std_dev, mean_vals, std_dev
    
    if matrix.shape[0] > large_sample_threshold:
        return matrix  
    else:
        standardized_data, _, _ = standardize(np.sin(matrix).T)
        return standardized_data.T


def perform_pca_reduction(matrix, n_components=2):
    """
    Perform PCA dimensionality reduction on gene expression data.
    
    Args:
        matrix (ndarray): Input gene expression matrix
        n_components (int): Number of PCA components
        
    Returns:
        ndarray: PCA-transformed data (n_components x samples)
    """
    
    pca = PCA(n_components=n_components)
    pca.fit(matrix.T)
    return pca.transform(matrix.T).T


def temporal_ordering(pca_data, original_matrix, time_points):
    """
    Perform temporal ordering of samples based on PCA angles.
    
    Args:
        pca_data (ndarray): PCA-transformed data
        original_matrix (ndarray): Original gene expression matrix
        time_points (ndarray): Original time points
        
    Returns:
        tuple: (sorted_matrix, sorted_indices, sorted_theta)
    """
    theta = np.arctan2(pca_data[1, :], pca_data[0, :])
    sorted_indices = np.argsort(theta)
    sorted_matrix = original_matrix[:, sorted_indices]
    radial_dist = np.sqrt(np.square(pca_data[1, :]) + np.square(pca_data[0, :]))
    radial_var = np.max(np.abs(radial_dist - np.mean(radial_dist))) - \
                 np.min(np.abs(radial_dist - np.mean(radial_dist)))
    
    threshold = 2 * (24 / (pca_data.shape[1] - 1))
    
    if radial_var < threshold:
        return sorted_matrix, sorted_indices, theta[sorted_indices]
    else:
        return handle_complex_temporal_case(theta, sorted_indices, original_matrix, time_points)


def handle_complex_temporal_case(theta, sorted_indices, matrix, time_points):
    """
    Handle complex temporal ordering when simple angular ordering isn't sufficient.
    
    Args:
        theta (ndarray): Angles from PCA
        sorted_indices (ndarray): Initially sorted indices
        matrix (ndarray): Gene expression matrix
        time_points (ndarray): Original time points
        
    Returns:
        tuple: (adjusted_matrix, adjusted_indices, adjusted_theta)
    """
    theta_sorted = (theta[sorted_indices] + 2 * np.pi) % (2 * np.pi)
    theta_diff = np.diff(theta_sorted)
    theta_diff = np.hstack((theta_diff, (theta_sorted[0] - theta_sorted[-1])))
    theta_diff = (theta_diff + 2 * np.pi) % (2 * np.pi)
    diff_sum = np.zeros(len(theta_diff))
    diff_lin = np.zeros(len(theta_diff))
    
    for i in range(len(theta_diff)):
        i_begin = len(theta_diff) - 1 if i == 0 else i - 1
        i_after = 0 if i == (len(theta_diff) - 1) else i + 1
        diff_sum[i] = np.abs(theta_diff[i_after] - theta_diff[i_begin])
        diff_lin[i] = np.abs(theta_diff[i_after] - theta_diff[i])
    
    i_max = np.argmax(diff_sum)
    threshold = (2 * np.pi) / len(theta_sorted)
    
    i_end = i_max
    for i in range(i_max, len(theta_diff)):
        if theta_diff[i] < threshold:
            i_end = i + 1
        else:
            break
    
    t_adjust_ind = np.arange(i_max, min(i_end + 2, len(theta_diff)))
    if i_end == (len(sorted_indices) - 1):
        t_adjust_ind = t_adjust_ind[:-1]
    
    t_adjust_ind_theta = sorted_indices[t_adjust_ind % len(sorted_indices)]
    
    Y_sorted = matrix[:, sorted_indices]
    adjusted_indices = sorted_indices.copy()
    
    for k in range(len(t_adjust_ind)):
        if len(t_adjust_ind) >= 2:
            error_ref = np.mean(np.square(
                Y_sorted[:, (i_end + 2) % Y_sorted.shape[1]] - 
                Y_sorted[:, (i_end + 1) % Y_sorted.shape[1]]
            ))
            
            errors = np.array([
                np.mean(np.square(
                    Y_sorted[:, t_adjust_ind[i] % Y_sorted.shape[1]] - 
                    Y_sorted[:, (i_end + 1) % Y_sorted.shape[1]]
                )) for i in range(len(t_adjust_ind))
            ])
            
            min_error_idx = np.argmin(np.abs(errors - error_ref))
            
            if matrix.shape[1] < 10:
                adjusted_indices = handle_small_matrix_case(
                    matrix, adjusted_indices, t_adjust_ind, i_end, i_max
                )
                break
            
            adjusted_indices = update_temporal_ordering(
                adjusted_indices, t_adjust_ind, min_error_idx, i_end
            )
    
    adjusted_matrix = matrix[:, adjusted_indices]
    adjusted_theta = theta[adjusted_indices]
    
    return adjusted_matrix, adjusted_indices, adjusted_theta


def handle_small_matrix_case(matrix, indices, adjust_ind, end_idx, max_idx):
    """
    Special handling for small matrices (<10 timepoints).
    """
    middle = indices[(end_idx + 2) % len(indices):]
    if len(middle) > 1:
        before_end = indices[end_idx]
        after_begin = middle[0]
        after_end = middle[-1]
        
        error_begin = np.mean(np.square(matrix[:, before_end] - matrix[:, after_begin]))
        error_end = np.mean(np.square(matrix[:, before_end] - matrix[:, after_end]))
        
        if error_begin > error_end:
            middle = middle[::-1]  # Reverse the order
        
        return np.hstack((indices[:max_idx + 1], middle))
    return indices


def update_temporal_ordering(indices, adjust_ind, min_idx, end_idx):
    """
    Update temporal ordering based on error minimization.
    """
    new_indices = indices.copy()
    new_indices[end_idx] = indices[adjust_ind[min_idx]]
    new_indices[adjust_ind[min_idx]] = indices[end_idx]
    return new_indices


def compute_slope_matrix(matrix, theta):
    """
    Compute slope matrix between consecutive time points.
    
    Args:
        matrix (ndarray): Gene expression matrix
        theta (ndarray): Time angles
        
    Returns:
        ndarray: Slope matrix (genes x time_intervals)
    """
    k_matrix = np.zeros((matrix.shape[0], len(theta) - 1))
    for i in range(matrix.shape[0]):
        for j in range(len(theta) - 1):
            k_matrix[i, j] = (matrix[i, j + 1] - matrix[i, j]) / \
                             (theta[j + 1] - theta[j])
    return k_matrix


def compute_slope_error_matrices(matrix):
    """
    Compute various error matrices between original and reconstructed data.
    
    Args:
        original (ndarray): Original expression matrix
        reconstructed (ndarray): Reconstructed expression matrix
        
    Returns:
        tuple: (error_matrix, std_error_matrix, slope_error_matrix)
    """
    n_genes, n_timepoints = matrix.shape
    slope_errors = np.zeros((n_genes, n_timepoints - 1))
    slope_errors = np.diff(matrix, axis=1)
    boundary_diffs = matrix[:, 0] - matrix[:, -1]
    error_matrix = np.column_stack((slope_errors, boundary_diffs))
    error_means = np.mean(error_matrix, axis=0)
    return error_matrix, error_means


def get_theta_order_update_nonoise(indicator, sorted_indices, matrix, max_value_index):
    ind_0 = max_value_index[0]
    ind_1 = max_value_index[1]
    theta_sorted_ind_new = sorted_indices
    if indicator == 0:
        indices_0 = np.where(sorted_indices == ind_0)[0][0]
        indices_1 = np.where(sorted_indices == ind_1)[0][0]
        min_indice = min(indices_0, indices_1)
        max_indice = max(indices_0, indices_1)
    
        if min_indice == np.min(sorted_indices) and max_indice == np.max(sorted_indices):  
            theta_sorted_ind_new = sorted_indices
        else:    
            if np.abs(indices_0-indices_1)<=1:
                theta_sorted_ind_new = np.hstack((sorted_indices[max_indice:],sorted_indices[:(min_indice+1)]))
    else:
        theta_sorted_ind_new = sorted_indices
    matrix_sort = matrix[:,theta_sorted_ind_new]
    
    return matrix_sort, theta_sorted_ind_new
    
def perform_temporal_sampling(matrix, method='uniform', k_sum=None, theta_time_sorted=None):
    """
    Perform temporal sampling using specified method.
    
    Args:
        matrix (ndarray): Gene expression matrix (genes x timepoints)
        method (str): Temporal sampling method:
            - 'uniform': Uniform 24-hour sampling
            - 'slope': Slope-proportional sampling (requires k_sum and theta_time_sorted)
            - 'umap': UMAP-based non-uniform sampling (default)
        k_sum (ndarray): Slope sums for 'slope' method
        theta_time_sorted (ndarray): Theta values for 'slope' method
        
    Returns:
        tuple: (x_axis, theta_umap, theta_umap_diff)
            x_axis: Constructed time points
            theta_umap: UMAP angles (None for non-UMAP methods)
            theta_umap_diff: UMAP angle differences (None for non-UMAP methods)
    """
    if method == 'uniform':
        x_axis = np.linspace(0, 24, matrix.shape[1])
        return x_axis, None, None
        
    elif method == 'slope':
        if k_sum is None or theta_time_sorted is None:
            raise ValueError("For slope method, both k_sum and theta_time_sorted must be provided")
        k_sum_abs = np.abs(k_sum)
        k_sum_normalized = k_sum_abs / np.sum(k_sum_abs) * 24
        
        x_axis = np.zeros_like(theta_time_sorted)
        for i in range(len(k_sum_normalized)-1):
            x_axis[i+1] = x_axis[i] + k_sum_normalized[i]

        x_axis = (x_axis + 24) % 24
        return x_axis, None, None
        
    elif method == 'umap':
        reducer = umap.UMAP(n_neighbors=7, min_dist=0.3, n_components=2, random_state=123)
        X_umap = reducer.fit_transform(matrix.T)
        x_mean = np.mean(X_umap[:, 0])
        y_mean = np.mean(X_umap[:, 1])
        x_centered = X_umap[:, 0] - x_mean
        y_centered = X_umap[:, 1] - y_mean
        norm = np.sqrt(x_centered**2 + y_centered**2)
        x_norm = x_centered / norm
        y_norm = y_centered / norm
        theta_umap = np.arctan2(y_norm, x_norm)
        theta_diff = np.diff(theta_umap)
        theta_diff = np.hstack((theta_diff, (theta_umap[0] - theta_umap[-1])))
        theta_diff = (theta_diff + 2*np.pi) % (2*np.pi)
        theta_diff_abs = np.abs(theta_diff)
        time_proportions = (theta_diff_abs / np.sum(theta_diff_abs)) * 24
        x_axis = np.zeros(len(theta_umap))
        for i in range(len(time_proportions)-1):
            x_axis[i+1] = x_axis[i] + time_proportions[i]

        x_axis[0] = (24 + time_proportions[-1]) % 24
        
        return x_axis, theta_umap, theta_diff
        
    else:
        raise ValueError(f"Invalid method '{method}'. Choose from 'uniform', 'slope', or 'umap'")
    

def cosinor_regression(matrix, x_axis):
    """
    Perform cosinor regression on gene expression data.
    
    Args:
        matrix (ndarray): Gene expression matrix
        x_axis (ndarray): Time points for regression
        theta (ndarray): Angular time points
        
    Returns:
        DataFrame: Cosinor regression results
    """
    data_all = pd.DataFrame({'x': [0]*2, 'y': [0]*2})
    data_all.insert(0, 'test', 'test_00')
    
    for i in range(matrix.shape[0]):
        df = pd.DataFrame({'x': x_axis, 'y': matrix[i, :]})
        df.insert(0, 'test', ['test' + str(i)] * len(x_axis))
        data_all = pd.concat([data_all, df], ignore_index=True)
    
    data_all = data_all.drop([0, 1])  # Remove initial dummy rows
    results = cosinor.fit_group(data_all, n_components=1, plot=False)
    results["test_num"] = results["test"].str.extract("(\d+)").astype(int)
    results = results.sort_values("test_num").drop("test_num", axis=1)
    
    return results


def reconstruct_expression(results, x_axis, n_genes=1000):
    """
    Reconstruct expression matrix using cosinor regression results.
    
    Args:
        results (DataFrame): Cosinor regression results
        x_axis (ndarray): Time points for reconstruction
        n_genes (int): Number of top genes to return
        
    Returns:
        tuple: (reconstructed_matrix, normalized_matrix, q_sorted_indices)
    """
    
    reconstructed = np.zeros((len(results), len(x_axis)))
    for i in range(len(results)):
        amp = results.iloc[i]['amplitude']
        phase = results.iloc[i]['acrophase']
        reconstructed[i, :] = amp * np.cos((x_axis / 24 * (2 * np.pi)) + phase)
    
    normalized = reconstructed / np.max(reconstructed, axis=1, keepdims=True)
    q_values = results['q'] if not results['q'].isna().all() else results['p']
    q_sorted_ind = np.argsort(q_values.values.T)
    
    return reconstructed, normalized, q_sorted_ind[:n_genes]


def compute_error_matrices(original, reconstructed):
    """
    Compute various error matrices between original and reconstructed data.
    
    Args:
        original (ndarray): Original expression matrix
        reconstructed (ndarray): Reconstructed expression matrix
        
    Returns:
        tuple: (error_matrix, std_error_matrix, slope_error_matrix)
    """
    
    error_matrix = original - reconstructed
    scaler = StandardScaler()
    std_error_matrix = scaler.fit_transform(error_matrix)
    slope_error = np.zeros((original.shape[0], original.shape[1] - 1))
    for i in range(original.shape[0]):
        for j in range(original.shape[1] - 1):
            slope_error[i, j] = original[i, j + 1] - original[i, j]

    circular_diff = original[:, 0] - original[:, -1]
    slope_error = np.hstack((slope_error, circular_diff[:, np.newaxis]))

    return error_matrix, std_error_matrix, slope_error


def save_results(fold_name, **kwargs):
    """
    Save all results to CSV files in specified folder.
    
    Args:
        fold_name (str): Directory to save results
        kwargs: Dictionary of {filename: data} pairs to save
    """
    for name, data in kwargs.items():
        df = pd.DataFrame(data)
        os.makedirs(f"{fold_name}/草稿",exist_ok = True)
        filepath = f"{fold_name}/草稿/{name}.csv"
        df.to_csv(filepath, index=False)


def read_and_preprocess(matrix, time_all, seed_gene_table, fold_name):
    """
    Main preprocessing pipeline for time-series gene expression data.
    
    Args:
        matrix (ndarray): Gene expression matrix (genes x timepoints)
        time_all (ndarray): Original time points
        fold_name (str): Output directory for results
        
    Returns:
        tuple: (normalized_matrix, new_timepoints, original_sorted, 
                normalized_error, theta_umap_diff, slope_error_matrix,
                mean_slope_error, standardized_error)
    """

    matrix = get_seed_gene_data(matrix, seed_gene_table)
    noise_indicator, _, max_value_index = detect_noise_structure(matrix)
    standardized_data = standardize_expression_data(matrix)
    pca_data = perform_pca_reduction(standardized_data)
    Y_sorted, sorted_indices, theta_sorted = temporal_ordering(
        pca_data, standardized_data, time_all
    )
    k_matrix = compute_slope_matrix(Y_sorted, theta_sorted)
    Y_sorted, sorted_indices = get_theta_order_update_nonoise(noise_indicator, sorted_indices, standardized_data, max_value_index)
    x_axis, theta_umap, theta_umap_diff = perform_temporal_sampling(Y_sorted)
    cosinor_results = cosinor_regression(Y_sorted, x_axis)
    Y_reconstructed, Y_normalized, top_indices = reconstruct_expression(
        cosinor_results, x_axis
    )
    error_matrix, std_error, slope_error = compute_error_matrices(
        Y_sorted, Y_normalized
    )
    mean_slope_error = np.mean(slope_error, axis=0)

    if pd.DataFrame(time_all) is not None:
        Y_all_time_sorted=time_all[sorted_indices]
    else:
        Y_all_time_sorted=time_all
    Amp_all = cosinor_results['amplitude'] 
    Phi_all = cosinor_results['acrophase']
    q_all = cosinor_results['q']

    
    return (
        Y_normalized[top_indices, :],
        Y_all_time_sorted,
        Y_sorted[top_indices, :],
        error_matrix[top_indices, :],
        theta_umap_diff,
        slope_error[top_indices, :],
        mean_slope_error,
        std_error[top_indices, :],
        Y_sorted,
        Amp_all,
        Phi_all,
        q_all
    )



def load_and_preprocess_data(data_path, time_path, seedgene_path, output_dir):
    """ Load_and_preprocess_data """    
    data = pd.read_csv(data_path)
    if time_path is not None:
        times = pd.read_csv(time_path)
    elif time_path == 'time':
        times = np.array(data.columns[1:])
        times = pd.DataFrame(times.astype(float).T)
    else:
        times = None
    else:
        times = None
    if seedgene_path is not None:
        seedgene = pd.read_excel(seedgene_path)
    else:
        seedgene = None
    if times is not None:
        shuffled_data, shuffled_times = shuffle_time_series(data, times, output_dir)
    else:
        shuffled_data, shuffled_times = data, times
    processed_data = read_and_preprocess(shuffled_data, shuffled_times, seedgene, output_dir)

    return processed_data
