import numpy as np
import pandas as pd
import os
from typing import Union, Tuple, Optional, Dict
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, silhouette_score


def create_periodic_grid(n_points: int, period: float = 24.0) -> np.ndarray:
    """Generate uniform grid over periodic interval [0, period)."""
    return np.linspace(0, period, num=n_points, endpoint=False)


def adjust_circadian_phase(
    expression_data: np.ndarray,
    predicted_phase: np.ndarray,
    time_points: np.ndarray,
    amplitude: np.ndarray,
    phase: np.ndarray,
    q_value: np.ndarray,
    q_threshold: float = 0.05
) -> np.ndarray:
    """
    Adjusts predicted circadian phases based on reference cosinor fits and quality metrics.
    
    Args:
        expression_data: Gene expression matrix (genes x timepoints)
        predicted_phase: Initial phase predictions in hours (0-24)
        time_points: reference time points in hours
        amplitude : Array of oscillation amplitudes for each gene (n_genes,)
        phase : Array of initial phase estimates in radians (n_genes,)
        q_value : Array of statistical significance q-values (n_genes,)
        q_threshold: FDR cutoff for significant rhythmic genes (default: 0.05)
        
    Returns:
        Adjusted phase predictions in hours (0-24)
    
    Raises:
        ValueError: If input dimensions are incompatible
    """
    
    try:
        if time_points is not None:
            if expression_data.shape[1] != len(time_points):
                raise ValueError("Dimension mismatch: expression_data columns != time_points length")
         
    except ValueError as e:
        time_points = create_periodic_grid(expression_data.shape[1])  
    
    amp, phi, q = amplitude, phase, q_value
    q_ind_005 = np.where(q < 0.05)[0]
    Y_all = np.expand_dims(expression_data, axis=0)
    adjusted_phase = predicted_phase
    
    if len(q_ind_005) != 0:
        mse_all, mse_all_zero = calculate_pairwise_mse(Y_all[0][q_ind_005,:])
        max_zero_ind = find_max_non_diagonal(mse_all_zero)
        ind_0, ind_1 = max_zero_ind
        if ind_0 == ind_1:
            sig_genes = q < q_threshold
            if not np.any(sig_genes):
                return (predicted_phase - predicted_phase[0]) % 24
            
            ref_curve, pred_curve = _generate_reference_curves(
                time_points, predicted_phase, amp, phi, sig_genes
            )
            
            phase_shift = _calculate_optimal_shift(
                ref_curve, 
                pred_curve,
                expression_data[sig_genes, :]
            )
            
            adjusted_phase = _apply_phase_correction(
                predicted_phase,
                phase_shift,
                time_points
            )
        else:
            
            c = ((predicted_phase - predicted_phase[0]) + 24) % 24
            adjusted_phase = c
       
    return adjusted_phase


def _generate_reference_curves(
    time_grid: np.ndarray,
    predicted_phase: np.ndarray,
    amplitudes: np.ndarray,
    phases: np.ndarray,
    sig_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate reference and predicted expression curves
    """
    ref_curve = amplitudes[sig_mask, None] * np.cos(
        2 * np.pi * (time_grid/24)[None, :] + phases[sig_mask, None]
    )
    
    pred_curve = amplitudes[sig_mask, None] * np.cos(
        2 * np.pi * (predicted_phase/24)[None, :] + phases[sig_mask, None]
    )
    
    return ref_curve, pred_curve

def _calculate_optimal_shift(
    ref_curve: np.ndarray,
    pred_curve: np.ndarray,
    observed_data: np.ndarray
) -> dict:
    """
    Calculate optimal phase shift through circular cross-correlation
    """
    
    metrics = {
        'forward': {'mse': [], 'shift': []},
        'reverse': {'mse': [], 'shift': []}
    }
    
    n_timepoints = ref_curve.shape[1]
    for shift in range(n_timepoints):
        
        shifted = np.roll(pred_curve, shift, axis=1)
        metrics['forward']['mse'].append(np.mean((shifted - ref_curve)**2))
        metrics['forward']['shift'].append(shift)
        
        rev_shifted = np.roll(pred_curve[:, ::-1], shift, axis=1)
        metrics['reverse']['mse'].append(np.mean((rev_shifted - ref_curve)**2))
        metrics['reverse']['shift'].append(shift)
    
    best_config = {}
    for direction in ['forward', 'reverse']:
        idx = np.argmin(metrics[direction]['mse'])
        best_config[direction] = {
            'shift': metrics[direction]['shift'][idx],
            'mse': metrics[direction]['mse'][idx]
        }
    
    return best_config

def _apply_phase_correction(
    phases: np.ndarray,
    shift_info: dict,
    time_grid: np.ndarray
) -> np.ndarray:
    """
    Apply optimal phase correction based on shift analysis
    """
    
    if shift_info['forward']['mse'] < shift_info['reverse']['mse']:
        direction = 1
        best_shift = shift_info['forward']['shift']
    else:
        direction = -1
        best_shift = shift_info['reverse']['shift']
    
    dt = np.mean(np.diff(time_grid))
    
    adjusted = np.roll(phases[::direction], best_shift)
    return (adjusted + best_shift * dt) % 24



def perform_clustering(data, n_clusters=4, random_state=42):
    """
    Perform K-means clustering on the input data.
    
    Parameters:
    -----------
    data : array-like
        Input data to be clustered (n_samples, n_features)
    n_clusters : int, optional
        Number of clusters to form (default: 4)
    random_state : int, optional
        Random seed for reproducibility (default: 42)
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'cluster_indices': list of arrays, indices of samples in each cluster
        - 'cluster_centers': array, coordinates of cluster centers
        - 'average_distances': array, mean distance to center for each cluster
        - 'silhouette_score': float, overall silhouette score
    """
    
    if data.size == 0:
        raise ValueError("Input data cannot be empty!")
    if len(data.shape) != 2:
        raise ValueError("Data must be a 2D array")

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(data)
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    cluster_samples = {i: [] for i in range(n_clusters)}
    for i, label in enumerate(cluster_labels):
        cluster_samples[label].append(data[i])
    
    distances = {}
    for i in range(n_clusters):
        dist = pairwise_distances_argmin_min(np.array(cluster_samples[i]), 
                                           np.array([cluster_centers[i]]))[1][0]
        distances[i] = np.mean(dist)
    
    sil_score = silhouette_score(data, cluster_labels)
    
    cluster_indices = []
    for i in range(n_clusters):
        indices = np.where(cluster_labels == i)[0]
        cluster_indices.append(indices)
    
    return {
        'cluster_indices': cluster_indices,
        'cluster_centers': cluster_centers,
        'average_distances': distances,
        'silhouette_score': sil_score
    }


def calculate_cluster_overlap(cluster_all, cluster_all_X1, cluster_all_X2):
    """
    Calculate overlap between clusters from different feature sets.
    
    Parameters:
    -----------
    cluster_all : list of arrays
        Cluster indices from full feature set
    cluster_all_X1 : list of arrays
        Cluster indices from first feature subset
    cluster_all_X2 : list of arrays
        Cluster indices from second feature subset
        
    Returns:
    --------
    tuple
        Two matrices (A, B) representing overlap counts between:
        - A: full features vs first subset
        - B: full features vs second subset
    """
    
    A = np.zeros((len(cluster_all), len(cluster_all_X1)))
    B = np.zeros((len(cluster_all), len(cluster_all_X2)))
    
    for i in range(len(cluster_all['cluster_indices'])):
        for j in range(len(cluster_all_X1['cluster_indices'])):
            intersection = set(cluster_all['cluster_indices'][i]).intersection(cluster_all_X1['cluster_indices'][j])
            A[i,j] = len(intersection)
            
        for j in range(len(cluster_all_X2['cluster_indices'])):
            intersection = set(cluster_all['cluster_indices'][i]).intersection(cluster_all_X2['cluster_indices'][j])
            B[i,j] = len(intersection)
    
    return A, B


def calculate_cluster_entropy(cluster_all, cluster_all_Xi, overlap_matrix):
    """
    Calculate entropy measure for cluster stability across feature subsets.
    
    Parameters:
    -----------
    cluster_all : list of arrays
        Cluster indices from full feature set
    cluster_all_Xi : list of arrays
        Cluster indices from feature subset
    overlap_matrix : array
        Overlap counts between full and subset clusters
        
    Returns:
    --------
    float
        Total entropy measure for cluster stability
    """
    entropy_matrix = np.zeros((len(cluster_all['cluster_indices']), len(cluster_all_Xi['cluster_indices'])))
    
    for i in range(len(cluster_all['cluster_indices'])):
        for j in range(len(cluster_all_Xi['cluster_indices'])):
            if overlap_matrix[i,j] > 0:
                p = overlap_matrix[i,j] / len(cluster_all_Xi['cluster_indices'][j])
                entropy_matrix[i,j] = -p * np.log(p)
    
    total_entropy = np.sum(entropy_matrix)
    return total_entropy


def calculate_cluster_overlap_percentage(cluster_all, cluster_all_Xi, overlap_matrix):
    """
    Calculate percentage overlap measure for cluster stability.
    
    Parameters:
    -----------
    cluster_all : list of arrays
        Cluster indices from full feature set
    cluster_all_Xi : list of arrays
        Cluster indices from feature subset
    overlap_matrix : array
        Overlap counts between full and subset clusters
        
    Returns:
    --------
    float
        Total overlap percentage measure
    """
    
    total_samples = sum(len(c) for c in cluster_all['cluster_indices'])
    
    overlap_measures = np.zeros(len(cluster_all['cluster_indices']))
    for i in range(len(cluster_all['cluster_indices'])):
        cluster_entropy = 1.0
        for j in range(len(cluster_all_Xi['cluster_indices'])):
            if overlap_matrix[i,j] > 0:
                p = overlap_matrix[i,j] / len(cluster_all_Xi['cluster_indices'][j])
                cluster_entropy *= p
        
        cluster_weight = len(cluster_all['cluster_indices'][i]) / total_samples
        overlap_measures[i] = cluster_entropy / cluster_weight
    
    total_overlap = np.sum(overlap_measures)
    return total_overlap



def determine_circadian_phase(
    data: np.ndarray,
    observed_times: np.ndarray,
    perform_clustering: callable,
    calculate_cluster_overlap: callable,
    calculate_cluster_entropy: callable,
    calculate_cluster_overlap_percentage: callable
) -> np.ndarray:
    """
    Determine optimal circadian phase using cluster stability analysis.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix (features x timepoints)
    observed_times : np.ndarray
        Observed time points (0-24h scale)
    perform_clustering : callable
        Function that performs clustering on data
    calculate_cluster_overlap : callable
        Function that calculates cluster overlaps
    calculate_cluster_entropy : callable
        Function that computes cluster entropy
    calculate_cluster_overlap_percentage : callable
        Function that calculates cluster overlap percentages

    Returns
    -------
    np.ndarray
        Optimal phase prediction (0-24h or 24-0h)
    """
    
    
    if data.shape[1] != len(observed_times):
        raise ValueError("Data and time points dimension mismatch")
    time_idx = np.argsort(observed_times)
    data_sorted = data[:, time_idx]
    times_sorted = observed_times[time_idx]
    cluster_all = perform_clustering(data)
    threshold = (len(times_sorted) - 1) / 2
    below_thresh = np.where(times_sorted < threshold)[0]
    
    if 0 < len(below_thresh) < len(times_sorted):
        split_idx = below_thresh[-1] + 1
    else:
        split_idx = int(threshold) + 1

    early_segment = data_sorted[:, :split_idx]
    late_segment = data_sorted[:, split_idx:]

    cluster_early = perform_clustering(early_segment)
    cluster_late = perform_clustering(late_segment)
    
    A, B = calculate_cluster_overlap(cluster_all, cluster_early, cluster_late)

    entropy_early = calculate_cluster_entropy(cluster_all, cluster_early, A)
    entropy_late = calculate_cluster_entropy(cluster_all, cluster_late, B)

    overlap_early = calculate_cluster_overlap_percentage(cluster_all, cluster_early, A)
    overlap_late = calculate_cluster_overlap_percentage(cluster_all, cluster_late, B)
    
    if entropy_early > entropy_late or overlap_early > overlap_late:
        predtime_sorted = 24 - observed_times
    else:
        predtime_sorted = observed_times

    return predtime_sorted
    
    


# Helper functions for modular organization
def calculate_pairwise_mse(Y):
    """Calculate pairwise MSE between all columns of Y"""
    mse_all = np.zeros((Y.shape[1], Y.shape[1]))
    mse_all_zero = np.zeros((Y.shape[1], Y.shape[1]))
    
    for i in range(Y.shape[1]):
        reference = Y[:,i]
        for j in range(Y.shape[1]):
            diff = reference - Y[:,j]
            mse_all[i,j] = np.mean(np.square(diff))
            mse_all_zero[i,j] = len(diff) - np.count_nonzero(diff)
            
    return mse_all, mse_all_zero

def find_max_non_diagonal(matrix):
    """Find maximum value in matrix excluding diagonal"""
    modified = matrix.copy()
    np.fill_diagonal(modified, 0)
    max_index = np.unravel_index(np.argmax(modified), modified.shape)
    return max_index




def cyclic_mse(t_pred, t_true, period=24):
    """
    Compute the mean squared error (MSE) between predicted and true time points
    on a circular scale (e.g., 24-hour clock).
    
    Parameters
    ----------
    t_pred : array-like
        Predicted time points.
    t_true : array-like
        True time points.
    period : float
        The period of the cycle (default is 24 for circadian).

    Returns
    -------
    float
        Mean squared error considering circular wrap-around.
    """
    t_pred = np.array(t_pred)[:len(t_true)]
    t_true = np.array(t_true)
    differences = np.abs(t_pred - t_true)
    cyclic_differences = period - differences
    min_differences = np.minimum(differences, cyclic_differences)
    return np.mean(min_differences**2)

def generate_circadian_transforms(c):
    """Generate 16 circadian time transformations.
    
    Args:
        c: Array of predicted times in hours (shape: (N+1,))
        
    Returns:
        List of transformed time series with descriptions
    """
    transforms, desc = [], []
    N = len(c) - 1
    
    transforms.extend([c, 24-c])
    desc.extend(["Original", "Global inversion"])
    
    t = np.concatenate([[c[0]], c[-1:0:-1]])
    transforms.extend([t, 24-t])
    desc.extend(["First-point fixed reversal", "Inverted first-point fixed"])
    
    t = c[::-1]
    transforms.extend([t, 24-t])
    desc.extend(["Full reversal", "Inverted full reversal"])
    
    t = np.roll(c, -1)
    transforms.extend([t, 24-t])
    desc.extend(["Left-shifted", "Inverted left-shifted"])
    
    t = np.roll(c, 1)
    transforms.extend([t, 24-t])
    desc.extend(["Right-shifted", "Inverted right-shifted"])
    
    t = np.concatenate([c[-2::-1], [c[-1]]])
    transforms.extend([t, 24-t])
    desc.extend(["Last-point fixed reversal", "Inverted last-point fixed"])
    
    t = np.concatenate([[c[1]], [c[0]], c[2:]])
    transforms.extend([t, 24-t])
    desc.extend(["Second-point leading", "Inverted second-point leading"])
    
    t = np.concatenate([c[2:], c[:2]])
    transforms.extend([t, 24-t])
    desc.extend(["Second-point trailing", "Inverted second-point trailing"])
    
    return transforms, desc


def evaluate_circadian_transformations(pred_times, time_all, folder_name001):
    """
    Evaluates 16 distinct circadian time transformations and selects the optimal 
    temporal representation based on circular error minimization.
    
    Parameters:
        c: ndarray (n_cells,)
            Initial circadian time predictions (hours)
        time_all: ndarray (n_cells,)
            Reference circadian phases (hours)
        folder_name001: str
            Output directory path
            
    Returns:
        ndarray: Optimized circadian time predictions
        
    Methodology:
        1. Generates 16 temporal transformations through:
           - Phase inversion (12h/24h)
           - Temporal reversal
           - Combined transformations
        2. Computes circular error for each variant:
           min(|Δt|, 12-|Δt|) for each cell
        3. Selects transformation with minimal mean circular error
        4. Ensures positive temporal progression via slope analysis
    """
    
    n_cells = len(pred_times)
    forward_idx = np.arange(n_cells)
    reverse_idx = (n_cells-1) - forward_idx
    all_transforms, _ = generate_circadian_transforms(pred_times)
    time_all = np.nan_to_num(time_all, nan=np.nanmean(time_all))
    def circular_error(pred, true):
        return np.mean(np.minimum((pred-true)%12, 12-(pred-true)%12))
    
    errors = [circular_error(t, time_all) for t in all_transforms]
    optimal_idx = np.argmin(errors)
    optimal_transform = all_transforms[optimal_idx]
    optimal_time = optimal_transform
    slope = np.polyfit(time_all, optimal_time, 1)[0]
    if slope < 0:
        reversed_versions = [
            optimal_time[reverse_idx],
            24 - optimal_time
        ]
        optimal_time = min(reversed_versions, 
                          key=lambda x: np.mean(np.minimum(
                              np.abs(x-time_all)%12, 
                              12-np.abs(x-time_all)%12)))
    
    return optimal_time, time_all



import numpy as np

def delta_phi(phi_0, phi, period=2*np.pi, mode="forgotten", N=200, median_scale=1):
    """
    Calculate the optimal phase adjustment between two sets of phases.
    
    Parameters:
    phi_0 (array-like): Reference phases
    phi (array-like): Phases to adjust
    period (float): Period of the circular space (default 2*pi)
    mode (str): Output mode ("forgotten", "say", "return", "no_median")
    N (int): Number of test offsets
    median_scale (float): Scale factor for median
    
    Returns:
    Depending on mode:
    - "say": prints median and returns bestphi
    - "return": returns dict with 'phi' and 'median'
    - "no_median": returns bestphi
    - default: prints median and returns dict
    """
    phi_0 = np.mod(phi_0, period)
    phi = np.mod(phi, period)
    isP = abs(period - 2*np.pi) > 0.001
    
    if not isP:
        obj = calc_delta(phi_0, phi, N)
        bestphi = obj['phi']
        mad = obj['median'] * median_scale
    else:
        obj = calc_delta(phi_0/period*2*np.pi, phi/period*2*np.pi, N)
        bestphi = obj['phi'] * period / 2 / np.pi
        mad = obj['median'] * period / 2 / np.pi * median_scale
    
    if mode == "say":
        print(f"median: {mad}")
        return bestphi
    elif mode == "return":
        return {'phi': bestphi, 'median': mad}
    elif mode == "no_median":
        return bestphi
    else:
        print(f"median: {mad}")
        return {'phi': bestphi, 'median': mad}

def calc_delta(phi_0, phi, N=200):
    """
    Helper function to calculate optimal phase adjustment.
    
    Parameters:
    phi_0 (array-like): Reference phases
    phi (array-like): Phases to adjust
    N (int): Number of test offsets
    
    Returns:
    dict: Contains 'phi' (adjusted phases) and 'median' of deltas
    """
    mad = 12
    bestphi = None
    sdel = None
    
    for j in range(1, N+1):
        offset = j/N * 2 * np.pi
        theta = np.mod(phi - offset, 2*np.pi)
        del_ = np.mod(np.abs(theta - phi_0), 2*np.pi)
        delta = np.minimum(del_, 2*np.pi - del_)
        
        current_median = np.median(delta)
        if current_median < mad:
            mad = current_median
            bestphi = theta
            j_temp = j
            sdel = delta
        
        phi_flipped = np.mod(-phi, 2*np.pi)
        theta = np.mod(phi_flipped - offset, 2*np.pi)
        del_ = np.mod(np.abs(theta - phi_0), 2*np.pi)
        delta = np.minimum(del_, 2*np.pi - del_)
        
        current_median = np.median(delta)
        if current_median < mad:
            mad = current_median
            bestphi = theta
            j_temp = -j
            sdel = delta
    
    return {'phi': bestphi, 'median': mad}

def adjust_phases(realphi, infphi, period=2*np.pi):
    """
    Adjust inferred phases to match true phases on a circular space.
    
    Parameters:
    realphi (array-like): True phases
    infphi (array-like): Inferred phases to adjust
    period (float): Period of the circular space (default 2*pi)
    
    Returns:
    array: Adjusted inferred phases
    """
    if abs(period - 2*np.pi) > 0.001:
        return adjust_phases(realphi*2*np.pi/period, infphi*2*np.pi/period) * period/(2*np.pi)
    
    infphi = np.array(infphi)
    realphi = np.array(realphi)
    
    mask1 = (realphi - infphi) > np.pi
    mask2 = (realphi - infphi) < -np.pi
    
    infphi[mask1] += 2*np.pi
    infphi[mask2] -= 2*np.pi
    
    return infphi


def get_phase_correct0(pred_times, true_times, folder_name):
    """
    Find best alignment of predicted time points to true time points, save the result,
    and return adjusted inferred phases.

    Parameters
    ----------
    pred_times : array-like
        Predicted time points.
    true_times : array-like
        True time points.
    folder_name : str
        Base folder path to save output.

    Returns
    -------
    np.ndarray
        Phase-corrected predicted times.
    """
    true_times = np.array(true_times).flatten()
    pred_times = np.array(pred_times)

    results = []
    for i in range(len(pred_times)):
        forward = np.concatenate([pred_times[i:], pred_times[:i]])
        backward = np.concatenate([pred_times[i::-1], pred_times[:i:-1]])

        mse_f = cyclic_mse(forward, true_times)
        mse_b = cyclic_mse(backward, true_times)

        if mse_f < mse_b:
            results.append([mse_f, 'forward', forward])
        else:
            results.append([mse_b, 'backward', backward])

    df = pd.DataFrame(results, columns=['MSE', 'Direction', 'Sorted Vector'])

    if df['MSE'].notnull().any():
        best_prediction = df.loc[df['MSE'].idxmin(), 'Sorted Vector']
    else:
        raise ValueError("No valid MSE values were found.")

    result = delta_phi(true_times, best_prediction, period=24, mode="say")
    adjusted = adjust_phases(true_times, result, period=24)
    return adjusted




def get_phase_correct(pred_times, true_times, label='both', folder_name='.'):
    """
    Find best alignment of predicted time points to true time points, save the result,
    and return adjusted inferred phases.

    Parameters
    ----------
    pred_times : array-like
        Predicted time points.
    true_times : array-like
        True time points.
    label : str
        Phase correction method: 'median', 'MSE', or 'both'.
    folder_name : str
        Base folder path to save output.

    Returns
    -------
    np.ndarray
        Phase-corrected predicted times.
    """
    true_times = np.array(true_times).flatten()
    pred_times = np.array(pred_times)
    
    if true_times is not None:
        
        pred_times, true_times = evaluate_circadian_transformations(pred_times, true_times, folder_name)

    if label == 'median':
        best_prediction = pred_times
        result = delta_phi(true_times, best_prediction, period=24, mode="say")
        adjusted = adjust_phases(true_times, result, period=24)
    elif label == 'MSE':
        results = []
        for i in range(len(pred_times)):
            forward = np.concatenate([pred_times[i:], pred_times[:i]])
            backward = np.concatenate([pred_times[i::-1], pred_times[:i:-1]])

            mse_f = cyclic_mse(forward, true_times)
            mse_b = cyclic_mse(backward, true_times)

            if mse_f < mse_b:
                results.append([mse_f, 'forward', forward])
            else:
                results.append([mse_b, 'backward', backward])

        df = pd.DataFrame(results, columns=['MSE', 'Direction', 'Sorted Vector'])

        if df['MSE'].notnull().any():
            best_prediction = df.loc[df['MSE'].idxmin(), 'Sorted Vector']
        else:
            raise ValueError("No valid MSE values were found.")
        
        adjusted = best_prediction

    else:  
        results = []
        for i in range(len(pred_times)):
            forward = np.concatenate([pred_times[i:], pred_times[:i]])
            backward = np.concatenate([pred_times[i::-1], pred_times[:i:-1]])

            mse_f = cyclic_mse(forward, true_times)
            mse_b = cyclic_mse(backward, true_times)

            if mse_f < mse_b:
                results.append([mse_f, 'forward', forward])
            else:
                results.append([mse_b, 'backward', backward])

        df = pd.DataFrame(results, columns=['MSE', 'Direction', 'Sorted Vector'])

        if df['MSE'].notnull().any():
            best_prediction = df.loc[df['MSE'].idxmin(), 'Sorted Vector']
        else:
            raise ValueError("No valid MSE values were found.")

        result = delta_phi(true_times, best_prediction, period=24, mode="say")
        adjusted = adjust_phases(true_times, result, period=24)

    return adjusted, true_times


def get_phase_correct1(pred_times, true_times, label='both', folder_name='.'):
    """
    Find best alignment of predicted time points to true time points, save the result,
    and return adjusted inferred phases.

    Parameters
    ----------
    pred_times : array-like
        Predicted time points.
    true_times : array-like
        True time points.
    label : str
        Phase correction method: 'median', 'MSE', or 'both'.
    folder_name : str
        Base folder path to save output.

    Returns
    -------
    np.ndarray
        Phase-corrected predicted times.
    """
    true_times = np.array(true_times).flatten()
    pred_times = np.array(pred_times)

    if label == 'median':
        best_prediction = pred_times
        result = delta_phi(true_times, best_prediction, period=24, mode="say")
        adjusted = adjust_phases(true_times, result, period=24)

    elif label == 'MSE':
        results = []
        for i in range(len(pred_times)):
            forward = np.concatenate([pred_times[i:], pred_times[:i]])
            backward = np.concatenate([pred_times[i::-1], pred_times[:i:-1]])

            mse_f = cyclic_mse(forward, true_times)
            mse_b = cyclic_mse(backward, true_times)

            if mse_f < mse_b:
                results.append([mse_f, 'forward', forward])
            else:
                results.append([mse_b, 'backward', backward])

        df = pd.DataFrame(results, columns=['MSE', 'Direction', 'Sorted Vector'])

        if df['MSE'].notnull().any():
            best_prediction = df.loc[df['MSE'].idxmin(), 'Sorted Vector']
        else:
            raise ValueError("No valid MSE values were found.")
        
        adjusted = best_prediction

    else:  
        results = []
        for i in range(len(pred_times)):
            forward = np.concatenate([pred_times[i:], pred_times[:i]])
            backward = np.concatenate([pred_times[i::-1], pred_times[:i:-1]])

            mse_f = cyclic_mse(forward, true_times)
            mse_b = cyclic_mse(backward, true_times)

            if mse_f < mse_b:
                results.append([mse_f, 'forward', forward])
            else:
                results.append([mse_b, 'backward', backward])

        df = pd.DataFrame(results, columns=['MSE', 'Direction', 'Sorted Vector'])

        if df['MSE'].notnull().any():
            best_prediction = df.loc[df['MSE'].idxmin(), 'Sorted Vector']
        else:
            raise ValueError("No valid MSE values were found.")

        result = delta_phi(true_times, best_prediction, period=24, mode="say")
        adjusted = adjust_phases(true_times, result, period=24)
    
    return adjusted, true_times



def get_corrected_phase(
    pred_times: Union[np.ndarray, list],
    true_times: Union[np.ndarray, list, str],
    processed_data: Tuple,
    label: str = 'both',
    correct_way: str = 'Phase',
    folder_name: Union[str, Path] = '.'
) -> Tuple[np.ndarray, Union[np.ndarray, str]]:
    """
    Corrects the predicted circadian phase based on different strategies.
    
    Parameters:
        pred_times (np.ndarray or list): Predicted circadian times.
        true_times (np.ndarray, list, or str): True circadian times. If 'None', indicates unsupervised mode.
        expression_data (np.ndarray): Gene expression matrix.
        Amp_all (np.ndarray): Estimated amplitude of circadian components.
        Phi_all (np.ndarray): Estimated phase of circadian components.
        q_all (np.ndarray): q-values associated with each gene.
        label (str): Correction strategy label. Default is 'both'.
        correct_way (str): Either 'Phase', 'Direction+Start', or 'None'.
        folder_name (str or Path): Directory for saving intermediate results. Default is '.'.

    Returns:
        Tuple[np.ndarray, Union[np.ndarray, str]]:
            - adjusted_times: Time vector after correction.
            - true_times: Provided or unchanged true times.
    """
    
    (matrix_new, time_all_new, Y_values, error_values, theta_umap_diff,
     Y_diff, Y_diff_mean, Y_std_error_values,
     Y_values_all, Amp_all, Phi_all, q_all) = processed_data
    
    expression_data = matrix_new
    expression_data_all = Y_values_all

    if true_times is not None:
        
        if correct_way == 'Phase':
            
            adjusted_times, true_times = get_phase_correct1(
                pred_times, true_times, label=label, folder_name=folder_name
            )
        
        else:
            print("333")
            optimal_phase = determine_circadian_phase(
                data=expression_data,
                observed_times=pred_times,
                perform_clustering=perform_clustering,
                calculate_cluster_overlap=calculate_cluster_overlap,
                calculate_cluster_entropy=calculate_cluster_entropy,
                calculate_cluster_overlap_percentage=calculate_cluster_overlap_percentage
            )

            print("optimal_phase:",optimal_phase)
            
            adjusted = adjust_circadian_phase(
                expression_data=expression_data_all,
                predicted_phase=optimal_phase,
                time_points='None',
                amplitude=Amp_all,
                phase=Phi_all,
                q_value=q_all
            )

            
            adjusted_times, true_times = get_phase_correct(
                adjusted, true_times, label=label, folder_name=folder_name
            )


    else:
        
        if correct_way == 'None':
            
            adjusted_times = pred_times
        else:
            
            optimal_phase = determine_circadian_phase(
                data=expression_data,
                observed_times=pred_times,
                perform_clustering=perform_clustering,
                calculate_cluster_overlap=calculate_cluster_overlap,
                calculate_cluster_entropy=calculate_cluster_entropy,
                calculate_cluster_overlap_percentage=calculate_cluster_overlap_percentage
            )

            adjusted = adjust_circadian_phase(
                expression_data=expression_data_all,
                predicted_phase=optimal_phase,
                time_points='None',
                amplitude=Amp_all,
                phase=Phi_all,
                q_value=q_all)
            
            adjusted_times = adjusted

    output_dir = os.path.join(folder_name)
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame({'adjusted_times': adjusted_times,'true_times': true_times})
    df.to_csv(f"{output_dir}/predresults.csv", index=False)

    return adjusted_times, true_times





