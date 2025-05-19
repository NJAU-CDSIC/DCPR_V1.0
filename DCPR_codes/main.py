# -*- coding: utf-8 -*-
"""
Created on Tue May 13 13:19:57 2025

@author: hx
"""



import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.sparse import random
from keras.models import Sequential
import keras
import keras.backend as K
import tensorflow 
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Dot, Activation, Add
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras import constraints, initializers


from keras.layers import Input, Dense, Reshape, Dot, Activation, Add, Flatten
import keras.layers as layers
from keras.models import Model
from functools import lru_cache
from typing import Callable
import os


@tf.function
def absolute_difference_loss(
    targets: tf.Tensor,    # Ground truth values [batch_size, ...]
    predictions: tf.Tensor  # Model predictions [batch_size, ...]
) -> tf.Tensor:
    """
    Computes the sum of absolute differences between target and predicted values.
    Particularly effective for robust regression tasks where outlier sensitivity
    should be minimized compared to squared error losses.
    
    Args:
        targets: Ground truth values of any shape
        predictions: Model outputs of same shape as targets
    
    Returns:
        Scalar loss value (sum of all absolute differences)
    """
    return tf.reduce_sum(tf.abs(targets - predictions))



@tf.function
def cross_entropy_loss(X: tf.Tensor, Y: tf.Tensor) -> tf.Tensor:
    """
    Computes the cross-entropy loss between high-dimensional (X) and 
    low-dimensional (Y) neighborhood distributions to preserve local topology.
    
    Args:
        X: High-dimensional data (d x n tensor)
        Y: Low-dimensional embedding (k x n tensor)
    
    Returns:
        Scalar loss value
    """
    spread = tf.constant(1.0, dtype=tf.float32)
    n_neighbors = tf.constant(5, dtype=tf.int32)  
    eta = tf.constant(0.5, dtype=tf.float32)
    n_samples = X.shape[1]
    lambdas = tf.ones(n_samples, dtype=tf.float32)  
    sigmas = spread * lambdas  
    D = tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(X, 2) - tf.expand_dims(X, 1)), axis=-1))
    neighbors = tf.cast(tf.math.top_k(-D, k=n_neighbors).indices, dtype=tf.int64)
    F_x = tf.exp(-tf.square(D / sigmas))  
    P = F_x / tf.reduce_sum(F_x, axis=2, keepdims=True)
    dist_Y = tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(Y, 2) - tf.expand_dims(Y, 1)), axis=-1))
    F = 1.0 / (1.0 + tf.square(dist_Y))
    Q = F / tf.reduce_sum(F, axis=2, keepdims=True)
    loss = 1 / 1000 * tf.reduce_sum(tf.where(P > 0, tf.ones_like(P), eta) * 1 / (1 + tf.exp(P * Q)), axis=1)
    
    return loss


@tf.function
def variance_preservation_loss(X, Y):
    """
    Computes the standard deviation alignment loss between high-dimensional space (X) and embedded space (Y).
    
    The loss encourages the preservation of variance structure by minimizing the squared distance between 
    each sample and the mean in both spaces, then computing the mean squared error between the two.

    Parameters:
        X (tf.Tensor): Input tensor of shape [batch_size, n_samples, feature_dim], representing high-dimensional data.
        Y (tf.Tensor): Embedded tensor of shape [batch_size, n_samples, embed_dim], representing low-dimensional embedding.

    Returns:
        tf.Tensor: A scalar loss value representing the alignment between the variance of X and Y.
    """
    X = X[0] 
    Y = Y[0]  
    X_mean = tf.reduce_mean(X, axis=0)  
    Y_mean = tf.reduce_mean(Y, axis=0) 
    dist_X = tf.reduce_sum(tf.square(tf.expand_dims(X, axis=0) - tf.expand_dims(X_mean, axis=0)), axis=-1)
    dist_Y = tf.reduce_sum(tf.square(tf.expand_dims(Y, axis=0) - tf.expand_dims(Y_mean, axis=0)), axis=-1)
    loss = tf.sigmoid(tf.reduce_mean(tf.square(dist_Y - dist_X)))

    return loss


@tf.function
def sinusoidal_consistency_loss(
    X: tf.Tensor,  
    Y: tf.Tensor   
) -> tf.Tensor:
    """Enforces sinusoidal pattern in sorted angular embeddings."""
    X = tf.squeeze(X, axis=-1)
    Y_sin = Y[..., 0]
    Y_cos = Y[..., 1]
    angles = tf.math.atan2(Y_sin, Y_cos)
    angles = tf.math.mod(angles + 2*np.pi, 2*np.pi)
    sorted_sin = tf.gather(Y_sin, tf.argsort(angles), batch_dims=1)
    return tf.reduce_mean(tf.square(sorted_sin - X))


@tf.function
#cos_loss
def cosinusoidal_consistency_loss(
    target: tf.Tensor,      # [n_samples,] Target cosine values
    Y: tf.Tensor    # [n_samples, 2] (sinθ, cosθ) coordinates
) -> tf.Tensor:
    """
    Enforces consistency between sorted angular coordinates and target cosine values.
    
    Args:
        target: Target cosine values for sorted angles
        embedding: Embedded coordinates (sinθ, cosθ)
    
    Returns:
        Scaled MSE loss
    """
    target = tf.squeeze(target, axis=-1)
    Y_sin = Y[..., 0]
    Y_cos = Y[..., 1]
    angles = tf.math.atan2(Y_sin, Y_cos)
    angles = tf.math.mod(angles + 2*np.pi, 2*np.pi)
    sorted_cos = tf.gather(Y_cos, tf.argsort(angles), batch_dims=1)
    loss = tf.reduce_mean(tf.square(sorted_cos - target)) / 1000
    return loss


@tf.function 
def orthogonality_loss(
    X: tf.Tensor,
    Y: tf.Tensor
) -> tf.Tensor:
    """Penalizes deviation from sine-cosine orthogonality."""
    
    X = tf.squeeze(X, axis=-1)
    Y_sin = Y[..., 0]
    Y_cos = Y[..., 1]
    sorted_idx = tf.argsort(tf.math.atan2(Y_sin, Y_cos))
    sin_cos = tf.gather(Y_sin, sorted_idx, batch_dims=1) * tf.gather(Y_cos, sorted_idx, batch_dims=1)
    return tf.reduce_mean(tf.square(sin_cos - X))



@tf.function
def slope_preservation_loss(X, Y):
    """
    Computes a slope-preserving loss between the high-dimensional space (X) and the embedded space (Y).
    
    This loss is designed to encourage preservation of pairwise angular relationships (slopes) between samples,
    particularly when Y represents a circular embedding (e.g., via sine and cosine encoding).

    Parameters:
        X (tf.Tensor): Input tensor of shape [n_samples, features], representing high-dimensional features.
        Y (tf.Tensor): Embedded tensor of shape [n_samples, 2], where [:, 0] is sin(θ), [:, 1] is cos(θ).

    Returns:
        tf.Tensor: A scalar tensor representing the slope difference loss.
    """
    import numpy as np  

    X = tf.transpose(X)[0]  
    X_k_diff = tf.math.subtract(X[1:], X[:-1])  
    Y_sin = tf.transpose(Y)[0]  
    Y_cos = tf.transpose(Y)[1]  
    Y_radians = tf.atan2(Y_sin, Y_cos)
    Y_radians = tf.math.mod(Y_radians + 2 * np.pi, 2 * np.pi)  
    sorted_indices = tf.argsort(Y_radians)
    Y_sin_order = tf.gather(Y_sin, sorted_indices)
    Y_cos_order = tf.gather(Y_cos, sorted_indices)
    k_sin = tf.math.subtract(Y_sin_order[1:], Y_sin_order[:-1])
    k_cos = tf.math.subtract(Y_cos_order[1:], Y_cos_order[:-1])
    k_sin_0 = tf.expand_dims(tf.math.subtract(Y_sin_order[0], Y_sin_order[-1]), 0)
    k_cos_0 = tf.expand_dims(tf.math.subtract(Y_cos_order[0], Y_cos_order[-1]), 0)
    k_sin_all = tf.concat([k_sin, k_sin_0], axis=0)
    k_cos_all = tf.concat([k_cos, k_cos_0], axis=0)
    k_all = (k_sin_all + 1.0) / (k_cos_all + 1.0)
    k_all_clean = tf.where(tf.math.is_nan(k_all), tf.zeros_like(k_all), k_all)
    loss = tf.reduce_sum(k_all_clean)

    return loss



@tf.function
def denoising_loss(
    X: tf.Tensor, 
    Y: tf.Tensor, 
    noise: tf.Tensor
) -> tf.Tensor:
    """
    Computes MSE between noise-corrupted inputs and denoised outputs.
    
    Args:
        X: Raw input (shape: [batch, n_samples, d_features])
        Y: Denoised output (shape: [batch, d_features, n_samples])
        noise: Additive noise (shape: [batch, n_samples, d_features])
    
    Returns:
        Scalar MSE value
    """
    
    noise = tf.cast(noise, tf.float32)
    noisy_input = X - noise
    return tf.reduce_mean(tf.square(noisy_input - tf.transpose(Y, [0, 2, 1])))


def make_denoising_wrapper(noise: tf.Tensor) -> callable:
    """Factory for denoising loss with fixed noise tensor"""
    def loss_fn(X, Y):
        return denoising_loss(X, Y, noise)
    return loss_fn



@tf.function
def cosine_similarity_loss(X, Y):
    """
    Parameters:
        X (tf.Tensor): Target tensor of shape [batch_size, time_steps, features], 
                       
        Y (tf.Tensor): Output tensor of shape [batch_size, time_steps, features], 
          
    Returns:
        tf.Tensor: A scalar loss value representing the mean squared error between
                   the sample-wise cosine similarity matrices of X and Y.
    """
    X_T = tf.transpose(X, perm=[0, 2, 1])  # [batch_size, features, time_steps]
    Y = tf.cast(Y, tf.float32)
    dot_product_X = tf.matmul(X_T, X_T, transpose_b=True)  # shape: [batch_size, features, features]
    norm_X = tf.norm(X_T, axis=-1, keepdims=True)          # shape: [batch_size, features, 1]
    similarity_matrix_X = dot_product_X / (norm_X * tf.transpose(norm_X, perm=[0, 2, 1]))
    dot_product_Y = tf.matmul(Y, Y, transpose_b=True)      # shape: [batch_size, time_steps, time_steps]
    norm_Y = tf.norm(Y, axis=-1, keepdims=True)            # shape: [batch_size, time_steps, 1]
    similarity_matrix_Y = dot_product_Y / (norm_Y * tf.transpose(norm_Y, perm=[0, 2, 1]))
    mse_error = tf.reduce_mean(tf.square(similarity_matrix_Y - similarity_matrix_X))

    return mse_error


@tf.function
def expression_differential_loss(X, Y):
    """
    Computes the mean squared error between the sample-wise temporal expression 
    difference of Y and X.

    This loss is designed to capture dynamic changes in circadian gene expression 
    across time by comparing the first-order temporal differences of the denoised 
    output to the transposed original signal. An additional cyclic difference (between 
    start and end timepoints) is appended to account for periodicity.

    Parameters:
        X (tf.Tensor): Target tensor of shape [batch_size, samples, features]. 
                       
        Y (tf.Tensor): Output tensor of shape [batch_size, samples, features]. 
                       

    Returns:
        tf.Tensor: A scalar loss value representing the mean squared error between 
                   the temporal difference of Y and X.
    """
    X_T = tf.transpose(X, perm=[0, 2, 1])
    Y_diff = tf.math.subtract(Y[:, 1:], Y[:, :-1])  # shape: [batch_size, time_steps - 1, features]
    diff_start_end = tf.math.subtract(Y[:, 0], Y[:, -1])  # shape: [batch_size, features]
    Y_diff_all = tf.concat([Y_diff, tf.expand_dims(diff_start_end, axis=1)], axis=1)  # shape: [batch_size, time_steps, features]
    mse_error = tf.reduce_mean(tf.square(Y_diff_all - X_T))

    return mse_error




@tf.function
def mean_expression_differential_loss(X, Y, noise):
    """
    
    Parameters:
        X (tf.Tensor): Tensor of shape [batch_size, time_steps, features], representing 
                       original circadian gene expression values before denoising.
        Y (tf.Tensor): Tensor of the same shape as X, representing the denoised output.
        noise (tf.Tensor): Tensor of predicted noise components corresponding to the original input.

    Returns:
        tf.Tensor: A scalar loss value measuring the mean squared error between predicted noise and
                   the original signal mean, encouraging minimal distortion from denoising.
    """
    mean_values = tf.reduce_mean(X, axis=1, keepdims=True)  # Shape: [batch_size, 1, features]
    Y = tf.cast(Y, tf.float32)
    noise = tf.cast(noise, tf.float32)
    mse_error = tf.reduce_mean(tf.square(noise - mean_values))  # Scalar loss

    return mse_error


def make_mean_reference_wrapper(noise):
    """
    Parameters:
        noise (tf.Tensor): Predicted noise tensor.

    Returns:
        function: A loss function compatible with TensorFlow/Keras models.
    """
    def loss_func(X, Y):
        return mean_expression_differential_loss(X, Y, noise)
    return loss_func



@tf.function
def standardized_denoising_loss(X, Y, noise):
    """
    Computes MSE between normalized-noise inputs and outputs.
    
    Args:
        X: Raw input (shape: [batch, n_samples, d_features])
        Y: Denoised output (shape: [batch, d_features, n_samples])
        noise: Normalized noise (shape: [batch, n_samples, d_features])
    
    Returns:
        Scalar MSE value
    """
    X_T = tf.transpose(X, perm=[0, 2, 1])
    Y = tf.cast(Y, tf.float32)
    Y_T = tf.transpose(Y, perm=[0, 2, 1])
    Y_std_error = tf.cast(noise, tf.float32)
    X_error = X + Y_std_error
    mse_error = tf.reduce_mean(tf.square(X_error - Y_T))
    
    return mse_error


def make_std_denoising_wrapper(noise):
    """
    Factory for standardized denoising loss
    """
    
    def loss_func(X, Y):
        return standardized_denoising_loss(X, Y, noise)
    
    return loss_func



class ExpressionFitter:
    _shared_coeffs = [tf.Variable(1.0) for _ in range(3)]  # [a, b, c]
    _shared_optimizer = tf.keras.optimizers.Adam()

    def __init__(self, learning_rate=0.01):
        self.coeffs = self._shared_coeffs
        self.optimizer = self._shared_optimizer
        if learning_rate != 0.01:
            self.optimizer.learning_rate.assign(learning_rate)
    
    @tf.function
    def quadratic(self, x):
        return self.coeffs[0]*x**2 + self.coeffs[1]*x + self.coeffs[2]
    
    @tf.function
    def compute_loss(
        self,
        angles: tf.Tensor,    
        expression: tf.Tensor 
    ) -> tf.Tensor:
        """Computes MSE between expression profile and quadratic fit"""
        with tf.GradientTape() as tape:
            pred = self.quadratic(tf.expand_dims(angles, -1))
            loss = tf.reduce_mean(tf.square(pred - expression))
        
        grads = tape.gradient(loss, self.coeffs)
        self.optimizer.apply_gradients(zip(grads, self.coeffs))
        return loss

_global_fitter = ExpressionFitter()


@tf.function
def expression_profile_fit_loss(
    target: tf.Tensor,      # [batch, features, samples]
    Y: tf.Tensor    # [batch, samples, 2]
) -> tf.Tensor:
    """
    Fits quadratic profiles to expression data in angular coordinates.
    
    Args:
        target: Gene expression values
        embedding: Circular coordinates (sinθ, cosθ)
    
    Returns:
        MSE between expression profiles and quadratic fits
    """
    fitter = _global_fitter
    Y_sin = Y[..., 0]
    Y_cos = Y[..., 1]
    angles = tf.math.atan2(Y_sin, Y_cos)
    angles = tf.math.mod(angles + 2*np.pi, 2*np.pi)
    sorted_idx = tf.argsort(angles)
    losses = []
    for feature in tf.unstack(target, axis=1):
        sorted_expr = tf.gather(feature, sorted_idx, batch_dims=1)
        loss = fitter.compute_loss(angles, sorted_expr)
        losses.append(loss)
    return tf.reduce_mean(losses)



@tf.function
def quadratic_function(x, a, b, c):
    return a * x**2 + b * x + c

def mean_squared_error(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
variables = [tf.Variable(1.0), tf.Variable(1.0), tf.Variable(1.0)]  # 初始化拟合参数



@tf.function
def value_fit_loss(X, Y):
    """
    Custom loss function that fits a quadratic function to temporally ordered circadian gene expression
    and computes the mean squared error between predicted and fitted values.

    Parameters:
        X : tf.Tensor
            The original gene expression tensor of shape (batch_size, time_points, features).
        Y : tf.Tensor
            The output tensor of shape (batch_size, time_points, features).

    Returns:
        loss : tf.Tensor
            The mean squared error between the true gene expression and the fitted quadratic curve.
    """
    
    # Use global optimization variables (a, b, c)
    global variables, optimizer
    neuron1 = Y[0][:, 0]  # x-coordinates
    neuron2 = Y[0][:, 1]  # y-coordinates
    Y_radians = tf.atan2(neuron1, neuron2)
    Y_radians = tf.math.mod(Y_radians + 2 * np.pi, 2 * np.pi)
    sorted_indices = tf.argsort(Y_radians)
    t_sort = tf.gather(Y_radians, sorted_indices)
    X_transposed = tf.transpose(X, perm=[0, 2, 1])
    X_sorted_transposed = tf.gather(X_transposed, sorted_indices, axis=1)
    X_sort = tf.transpose(X_sorted_transposed, perm=[0, 2, 1])
    t_sort_expanded = tf.expand_dims(t_sort, axis=0)
    x_data_all = tf.tile(t_sort_expanded, [X_sort.shape[1], 1])
    y_data_all = X_sort[0]
    x_batch = tf.expand_dims(x_data_all, axis=-1)
    y_batch = tf.expand_dims(y_data_all, axis=-1)
    with tf.GradientTape() as tape:
        y_fit = quadratic_function(x_batch, *variables)
        loss = mean_squared_error(y_batch, y_fit)
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    a_fit, b_fit, c_fit = [v.value() for v in variables]
    y_fit = quadratic_function(x_batch, a_fit, b_fit, c_fit)
    mse = mean_squared_error(y_batch, y_fit)
    return mse


@tf.function
def error_loss_noise(X, Y, noise):
    """
    Compute mean squared error loss between input X and noisy embedding Y plus error Y_error.
    """
    X_T = tf.transpose(X, perm=[0, 2, 1])
    Y = tf.cast(Y, tf.float32)
    Y_T = tf.transpose(Y, perm=[0, 2, 1])
    Y_error = tf.cast(noise, tf.float32)
    X_error = Y_T + Y_error
    mse_error = tf.reduce_mean(tf.square(X_error - X))
    return mse_error

def make_denoise_loss(noise):
    """
    Return a loss function that computes error_loss_noise.
    """
    def loss_func(X, Y):
        return error_loss_noise(X, Y, noise)
    return loss_func

@tf.function
def cross_entropy_loss_noise(X, Y, noise):
    """
    Compute cross-entropy style loss encouraging consistency between input X and embedding Y,
    considering noise level Y_error.
    """
    # Define constants for neighborhood structure
    spread = tf.constant(1.0, dtype=tf.float32)
    n_neighbors = tf.constant(5, dtype=tf.int32)
    eta = tf.constant(0.5, dtype=tf.float32)
    n_samples = X.shape[1]
    lambdas = tf.ones(n_samples, dtype=tf.float32)
    sigmas = spread * lambdas
    D = tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(X, axis=2) - tf.expand_dims(X, axis=1)), axis=-1))
    neighbors = tf.cast(tf.math.top_k(-D, k=n_neighbors).indices, dtype=tf.int64)
    F_x = tf.exp(-tf.square(D / sigmas))
    P = F_x / tf.reduce_sum(F_x, axis=2, keepdims=True)
    distances = tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(Y, axis=2) - tf.expand_dims(Y, axis=1)), axis=-1))
    F = 1.0 / (1.0 + tf.square(distances))
    Q = F / tf.reduce_sum(F, axis=2, keepdims=True)
    loss = 1 / 1000 * tf.reduce_sum(tf.where(P > 0, tf.ones_like(P), eta) * 1 / (1 + tf.exp(P * Q)), axis=1)
    return loss

def create_noise_aware_loss(noise):
    """
    Return a noise-aware loss function that combines input noise with embedding similarity loss.
    """
    def loss_func(X, Y):
        X_T = tf.transpose(X, perm=[0, 2, 1])
        Y = tf.cast(Y, tf.float32)
        Y_T = tf.transpose(Y, perm=[0, 2, 1])
        Y_error_new = tf.cast(noise, tf.float32)
        X_error = Y_T + noise
        Y = tf.transpose(X_error, perm=[0, 2, 1])
        X = X_T
        
        return cross_entropy_loss_noise(X, Y, Y_error_new)
    return loss_func




@tf.function
def embedding_criterion(reference_signal, predicted_output):
    """
    Embedding Correction Criterion based on Mean Squared Error (MSE) Minimization.

    Args:
        reference_signal (tf.Tensor): 1D tensor representing the true sinusoidal signal.
        predicted_output (tf.Tensor): 2D tensor of shape (batch_size, 2), 
                                      where each row contains (sin_component, cos_component).

    Returns:
        tf.Tensor: A corrected embedding tensor of shape (batch_size, 2),
                   after resolving potential phase ambiguity.
    """
    
    sin_component = tf.transpose(predicted_output)[0]
    cos_component = tf.transpose(predicted_output)[1]
    phase_angles = tf.atan2(sin_component, cos_component)
    sorted_indices = tf.argsort(phase_angles)
    sin_sorted = tf.gather(sin_component, sorted_indices)
    mse_original = tf.reduce_mean(tf.square(sin_sorted - reference_signal))
    sin_sorted_negated = -sin_sorted
    mse_negated = tf.reduce_mean(tf.square(sin_sorted_negated - reference_signal))
    should_negate = tf.greater(mse_original, mse_negated)
    sin_corrected = tf.where(should_negate, -sin_component, sin_component)
    corrected_embedding = keras.layers.Concatenate(name='embedding_corrected', axis=1)(
        [sin_corrected, cos_component]
    )

    return corrected_embedding



class DiscriminationLayer(tf.keras.layers.Layer):
    """Neural embedding discrimination layer with phase alignment correction."""
    
    def __init__(self, sin_values, input_matrix, k, **kwargs):
        super(DiscriminationLayer, self).__init__(**kwargs)
        self.sin_values = sin_values
        self.input_matrix = tf.constant(input_matrix)
        self.k = tf.constant(k, dtype=tf.float32)

    def get_slope(self, embeddings):
        """Calculate phase progression slope using vectorized operations."""
        neuron_x, neuron_y = tf.unstack(tf.transpose(embeddings[0]), num=2)
        phases = tf.math.mod(tf.atan2(neuron_x, neuron_y) + 2*np.pi, 2*np.pi)
        
        x_axis = tf.linspace(0.0, 2*np.pi, tf.shape(phases)[0])
        x_mean, y_mean = tf.reduce_mean(x_axis), tf.reduce_mean(phases)
        
        cov = tf.reduce_sum((x_axis - x_mean) * (phases - y_mean))
        var = tf.reduce_sum(tf.square(x_axis - x_mean))
        slope = cov / var
        
        return tf.cond(
            tf.less(tf.abs(slope - 1.0), tf.abs(self.k - 1.0)),
            lambda: slope,
            lambda: self.k
        )

    def get_optimal_transformation(self, embeddings):
        """Vectorized implementation of phase transformation selection."""
        X = tf.convert_to_tensor(self.sin_values)
        neuron_x, neuron_y = tf.unstack(tf.transpose(embeddings[0]), num=2)
        
        phases = tf.math.mod(tf.atan2(neuron_x, neuron_y) + 2*np.pi, 2*np.pi)
        sorted_idx = tf.argsort(phases)
        ordered_phases = tf.gather(phases, sorted_idx)
        
        time_intervals = ordered_phases / (2*np.pi) * 24
        interval_diffs = time_intervals[1:] - time_intervals[:-1]
        ref_intervals = tf.fill(tf.shape(interval_diffs), 24/tf.cast(tf.shape(interval_diffs)[0], tf.float32))
        spacing_error = tf.reduce_mean(tf.abs(interval_diffs - ref_intervals))
        
        def true_fn():
            return tf.constant(-3.0, dtype=tf.float32)
        
        def false_fn():
            slope = self.get_slope(embeddings)
            cond = tf.abs(slope - 1.0) < tf.abs(self.k - 1.0)
            
            def phase_analysis():
                num_phases = tf.shape(ordered_phases)[0]
                indices = tf.range(num_phases)
                shifted_phases = tf.map_fn(
                    lambda i: tf.roll(ordered_phases, shift=-i, axis=0),
                    indices,
                    fn_output_signature=tf.float32
                )
                mse_values = tf.reduce_mean(tf.square(tf.sin(shifted_phases) - X), axis=1)
                min_mse_idx = tf.argmin(mse_values)
                
                reversed_phases = tf.math.mod(-ordered_phases + 2*np.pi, 2*np.pi)
                reversed_mse = tf.reduce_mean(tf.square(tf.sin(reversed_phases) - X))
                
                return tf.cond(
                    tf.less(tf.reduce_min(mse_values), reversed_mse),
                    lambda: tf.cond(
                        tf.greater_equal(tf.reduce_min(tf.roll(ordered_phases, shift=-min_mse_idx, axis=0)),
                                        tf.reduce_min(reversed_phases)),
                        lambda: tf.constant(-3.0, dtype=tf.float32),
                        lambda: tf.constant(1.0, dtype=tf.float32)
                    ),
                    lambda: tf.constant(-3.0, dtype=tf.float32)
                )
            
            return tf.cond(cond, phase_analysis, lambda: tf.constant(1.0, dtype=tf.float32))
        
        return tf.cond(tf.less(spacing_error, 0.1), true_fn, false_fn)

    def call(self, inputs):
        """Graph-compatible phase correction."""
        trans_factor = self.get_optimal_transformation(inputs)
        neuron_x, neuron_y = tf.unstack(tf.transpose(inputs[0]), num=2)
        
        corrected_x = trans_factor * neuron_x
        return (
            tf.expand_dims(tf.transpose([corrected_x]), 0),
            tf.expand_dims(tf.transpose([neuron_y]), 0)
        )


tf.keras.backend.clear_session()
tf.random.set_seed(42)
class Predict(tf.keras.Model):
    def __init__(self, inputs, batch_size, hidden_size, num_heads, linear_width, sin_values):
        super().__init__()  # 修改这里
        self.inputs = inputs
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.linear_width = linear_width
        self.depth = hidden_size // num_heads
        self.emd_width = inputs.shape[0]
        self.out_width = inputs.shape[1]
        self.sin_values = sin_values
        self.k1 = tf.constant(0.0, dtype=tf.float32)
        self.DiscriminationLayer = DiscriminationLayer(self.sin_values, self.inputs, self.k1)
        self.model = self.build_model()

    def split_heads(self, Input_x1, layer_name):
        reshaped_x1 = tf.keras.layers.Lambda(
            lambda x: tf.reshape(x, [self.batch_size, -1, self.num_heads, self.depth]), 
            name=f"{layer_name}_reshape")(Input_x1)
        return tf.keras.layers.Lambda(
            lambda x: tf.transpose(x, perm=[0, 2, 1, 3]), 
            name=f"{layer_name}_transpose")(reshaped_x1)

    def build_model(self):
        
        Inputs = tf.keras.Input(shape=(self.inputs.shape[0], self.inputs.shape[1]), name='input')
        Inputs_transpose = tf.keras.layers.Lambda(
            lambda x: tf.transpose(x, perm=[0, 2, 1]), 
            name='input_transpose')(Inputs)

        u1 = Dense(1, activation='relu', name='gene_cos')(Inputs)
        u2 = Dense(1, activation='relu', name='gene_sin')(Inputs)
        m1 = Dense(1, activation='relu', name='cell_cos')(Inputs_transpose)
        m2 = Dense(1, activation='relu', name='cell_sin')(Inputs_transpose)

        u01 = tf.keras.layers.Lambda(keras.backend.cos, name='u00_sqr')(u1)
        u02 = tf.keras.layers.Lambda(keras.backend.sin, name='u01_sqr')(u2)
        u001 = tf.keras.layers.Lambda(keras.backend.square, name='u001_sqr')(u01)
        u002 = tf.keras.layers.Lambda(keras.backend.square, name='u002_sqr')(u02)
        u0012 = tf.keras.layers.Add(name='u_sqr_len')([u001, u002])
        u0012 = tf.keras.layers.Lambda(keras.backend.sqrt, name='u_len')(u0012)
        u11 = tf.keras.layers.Lambda(lambda x: x[0]/x[1], name='u_encoder_circular_out_0')([u01, u0012])
        u12 = tf.keras.layers.Lambda(lambda x: x[0]/x[1], name='u_encoder_circular_out_1')([u02, u0012])
        u112 = tf.keras.layers.Lambda(lambda x: x[0]/x[1], name='u_encoder_circular_tan_fai')([u12, u11])
        fai_g = tf.keras.layers.Lambda(tensorflow.math.atan, name='u_ten')(u112)
        u21 = tf.keras.layers.Lambda(keras.backend.cos, name='u200_sqr')(fai_g)
        u22 = tf.keras.layers.Lambda(keras.backend.sin, name='u201_sqr')(fai_g)
        u = tf.keras.layers.Lambda(lambda x: layers.Concatenate(axis=2)(x), name='concatenated_u')([u21, u22])

        m01 = tf.keras.layers.Lambda(keras.backend.cos, name='m00_sqr')(m1)
        m02 = tf.keras.layers.Lambda(keras.backend.sin, name='m01_sqr')(m2)
        m001 = tf.keras.layers.Lambda(keras.backend.square, name='m001_sqr')(m01)
        m002 = tf.keras.layers.Lambda(keras.backend.square, name='m002_sqr')(m02)
        m0012 = tf.keras.layers.Add(name='m_sqr_len')([m001, m002])
        m0012 = tf.keras.layers.Lambda(keras.backend.sqrt, name='m_len')(m0012)
        m11 = tf.keras.layers.Lambda(lambda x: x[0]/x[1], name='m_encoder_circular_out_0')([m01, m0012])
        m12 = tf.keras.layers.Lambda(lambda x: x[0]/x[1], name='m_encoder_circular_out_1')([m02, m0012])
        m112 = tf.keras.layers.Lambda(lambda x: x[0]/x[1], name='m_encoder_circular_tan_theta')([m12, m11])
        theta = tf.keras.layers.Lambda(tensorflow.math.atan, name='m_ten')(m112)
        m21 = tf.keras.layers.Lambda(keras.backend.cos, name='m200_sqr')(theta)
        m22 = tf.keras.layers.Lambda(keras.backend.sin, name='m201_sqr')(theta)
        m = tf.keras.layers.Lambda(lambda x: layers.Concatenate(axis=2)(x), name='concatenated_m')([m21, m22])

        x_1 = tf.keras.layers.Lambda(lambda x: layers.Dot(axes=2)(x), name='dot_layer')([u, m])
        
        Q = Dense(self.hidden_size, name='Q')(Inputs)
        K = Dense(self.hidden_size, name='K')(Inputs)
        V = Dense(self.hidden_size, name='V')(Inputs)
        
        Q = self.split_heads(Q, 'Q')
        K = self.split_heads(K, 'K')
        V = self.split_heads(V, 'V')
        
        attention_weights = tf.keras.layers.Lambda(
            lambda x: tf.nn.softmax(x), name='attention_weights')(
                tf.keras.layers.Lambda(
                    lambda x: tf.matmul(x[0], x[1], transpose_b=True), 
                    name='Q_K_matmul_transpose')([Q, K]))
        
        out1 = tf.keras.layers.Lambda(
            lambda x: tf.matmul(x[0], x[1]), name='out1')([attention_weights, V])
        out2 = tf.keras.layers.Lambda(
            lambda x: tf.transpose(x, perm=[0, 2, 1, 3]), name='out2')(out1)
        out3 = tf.keras.layers.Lambda(
            lambda x: tf.reshape(x, [tf.shape(x)[0], -1, self.hidden_size]), name='out3')(out2)
        
        out = Dense(self.hidden_size, name='final')(out3)
        out_final = Dense(self.inputs.shape[0], name='final1')(out)
        
        x_2 = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=0), name='attention')(out_final)
        x_2 = tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x, axis=1), name='softmax_output')(x_2)
        x_2 = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=0), name='expanded_output')(x_2)
        x_2 = tf.keras.layers.Lambda(
            lambda x: tf.reshape(x, [-1, Inputs.shape[1], x.shape[2]]), name='reshaped_output')(x_2)

        import keras.backend as K
        x_3 = layers.Lambda(lambda x: tf.concat(x, axis=2), name='concat_layer')([x_1, x_2])
        
        x00 = keras.layers.Dense(name='cos0',
                                  units=1,
                                  kernel_initializer=keras.initializers.glorot_normal(seed=None),
                                  bias_initializer=keras.initializers.Zeros()
                                  )(tf.keras.layers.Lambda(lambda x: K.cos(x), name='cos_0_layer')(x_3))
        x01 = keras.layers.Dense(name='sin0',
                                  units=1,
                                  kernel_initializer=keras.initializers.glorot_normal(seed=None),
                                  bias_initializer=keras.initializers.Zeros()
                                  )(tf.keras.layers.Lambda(lambda x: K.sin(x), name='sin_0_layer')(x_3))
        
        x_total1 = keras.layers.Concatenate(name='embedding1')([x00, x01])
        
        y_1 = keras.layers.Dense(name='output1',
                                    units=self.emd_width,
                                    kernel_initializer=keras.initializers.glorot_normal(seed=None),
                                    bias_initializer=keras.initializers.Zeros()
                                    )(x_total1)
        
        x10 = keras.layers.Dense(name='cos1',
                                  units=1,
                                  kernel_initializer=keras.initializers.glorot_normal(seed=None),
                                  bias_initializer=keras.initializers.Zeros()
                                  )(tf.keras.layers.Lambda(lambda x: K.cos(x), name='cos_1_layer')(y_1))
        x11 = keras.layers.Dense(name='sin1',
                                  units=1,
                                  kernel_initializer=keras.initializers.glorot_normal(seed=None),
                                  bias_initializer=keras.initializers.Zeros()
                                  )(tf.keras.layers.Lambda(lambda x: K.sin(x), name='sin_1_layer')(y_1))
        
        x_total2 = keras.layers.Concatenate(name='embedding2')([x10, x11])
        
        y_2 = keras.layers.Dense(name='output2',
                                    units=self.out_width,
                                    kernel_initializer=keras.initializers.glorot_normal(seed=None),
                                    bias_initializer=keras.initializers.Zeros()
                                    )(x_total2)
       
        x4_T = tf.keras.layers.Lambda(
            lambda x: tf.transpose(x, perm=[0, 2, 1]), name='predict_transposed_output')(y_2)

        
        x00 = Dense(1, name='encoder_circular_in_0',
                   kernel_initializer=keras.initializers.glorot_normal(),
                   bias_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05))(x4_T)
        x01 = Dense(1, name='encoder_circular_in_1',
                   kernel_initializer=keras.initializers.glorot_normal(),
                   bias_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05))(x4_T)
        x002 = tf.keras.layers.Lambda(keras.backend.square, name='x00_sqr')(x00)
        x012 = tf.keras.layers.Lambda(keras.backend.square, name='x01_sqr')(x01)
        xx0 = tf.keras.layers.Add(name='sqr_len')([x002, x012])
        xx0 = tf.keras.layers.Lambda(keras.backend.sqrt, name='len')(xx0)
        x00 = tf.keras.layers.Lambda(lambda x: x[0]/x[1], name='encoder_circular_out_0')([x00, xx0])
        x01 = tf.keras.layers.Lambda(lambda x: x[0]/x[1], name='encoder_circular_out_1')([x01, xx0])

        if self.linear_width > 0:
            x1 = Dense(self.linear_width, name='encoder_linear_out',
                      kernel_initializer=keras.initializers.glorot_normal(),
                      bias_initializer=keras.initializers.Zeros())(x4_T)
            x = tf.keras.layers.Concatenate(name='embedding')([x00, x01, x1])
        else:
            x = tf.keras.layers.Concatenate(name='embedding')([x00, x01])
        
        
        x10, x11 = self.DiscriminationLayer(x)
        x_again = tf.keras.layers.Concatenate(name='embedding_true', axis=2)([x10, x11])
        
        y_hat = keras.layers.Dense(name='output',
                                    units=self.inputs.shape[0],
                                    kernel_initializer=keras.initializers.glorot_normal(seed=None),
                                    bias_initializer=keras.initializers.Zeros()
                                    )(x_again)
        
        self.model = keras.Model(outputs=[y_hat, y_hat, x, x, x, x, x, x, y_hat, y_hat, y_hat, y_hat, y_hat, x, y_hat, y_hat], inputs=Inputs)
        
        return self.model
    
    
    class MyCallback(keras.callbacks.Callback):
        """
        A comprehensive training monitoring callback that tracks and visualizes:
        - Model performance metrics (AUC, median error)
        - Training loss progression
        - Embedding space evolution
        - Weight parameter changes
        
        Periodically saves model states and generates analytical visualizations.
        """
        
        def __init__(self, data, time_all, main_instance, batch_size, verbose=0):
            """
            Initializes the monitoring callback.
            
            Args:
                data: Validation data tuple (inputs, targets) used for evaluation
                time_all: Dictionary containing experimental timeline metadata  
                folder_name: Base output directory path for saving results
                main_instance: Reference to the main training class instance
                batch_size: Training batch size
                verbose: Control output verbosity (0 = silent, 1 = verbose)
            """
            super().__init__()
            self.main_instance = main_instance
            self.data = data
            self.time_all = time_all
            self.batch_size = batch_size
            self.verbose = verbose
            self.cum_all_auc = []      
            self.cum_all_merr = []     
            self.slope_update_all = [0] 
            self.trans_update_all = [0] 
            self.get_k_all = []         
            self.train_loss = []        
            self.last_pred_times = None  
            self.last_true_times = None 
        
        def on_train_begin(self, logs=None):
            """Executes at training start: saves initial state and baselines."""
            predTimes, trueTimes = self.main_instance.plot_all_compare1(self.data, 0, self.time_all)
            
        def on_epoch_end(self, epoch, logs=None):
            """Executes at each epoch end: updates tracking and periodic saves."""
            if logs is not None:
                self.train_loss.append(logs.get('loss'))
            emb_x_up = self.main_instance.emb_x_update(self.data)
            if (epoch+1) % 1000 == 0:
                self._perform_periodic_saving(epoch)
            
        def _perform_periodic_saving(self, epoch):
            """Handles periodic model state saving and visualization."""
            self.last_pred_times, self.last_true_times = self.main_instance.plot_all_compare1(self.data, epoch, self.time_all)
            
        def get_last_predictions(self):
            return self.last_pred_times, self.last_true_times
    
        
    def train(self, data, sin_values2, cos_values, dot_values, k_values, 
              Y_values, Y_error_values, Y_diff, matrix_error_mean, 
              Y_std_error_values, Y_values_all, Amp_all, Phi_all, q_all, loss_p, time_all, 
              batch_size: int = None, epochs: int = 100, 
              verbose: int = 10, rate: float = 1e-4):
        """
        Optimizes model parameters through multi-objective loss minimization.
        
        Parameters:
            data (array): Training dataset containing input features
            sin_values2, cos_values, dot_values, k_values: Trigonometric and geometric constraints
            Y_values, Y_error_values: Target outputs and associated error terms
            Y_diff: Differential constraints
            matrix_error_mean: Matrix-based error metrics
            Y_std_error_values: Standard deviation error terms
            loss_p: Loss weighting parameters
            time_all: Temporal reference values
            folder_name001: Output directory for training artifacts
            batch_size: Mini-batch size for stochastic optimization
            epochs: Maximum training iterations
            verbose: Logging frequency
            rate: Learning rate
        
        Returns:
            array: Training loss history
        """
        self.model.compile(
            optimizer='adam',
            loss=[
                'mean_squared_error',  
                absolute_difference_loss,              
                cross_entropy_loss,    
                variance_preservation_loss,             
                sinusoidal_consistency_loss, 
                cosinusoidal_consistency_loss,    
                orthogonality_loss, 
                slope_preservation_loss,      
                make_denoising_wrapper(Y_error_values),         
                cosine_similarity_loss,         
                expression_differential_loss,     
                make_mean_reference_wrapper(matrix_error_mean),      
                make_std_denoising_wrapper(Y_std_error_values),  
                expression_profile_fit_loss,       
                make_denoise_loss(Y_error_values),    
                create_noise_aware_loss(Y_error_values)  
            ],
            loss_weights=loss_p
        )
        
        callback = self.MyCallback(data, time_all, self, batch_size, verbose)
        
        print("---------------------training--------------------")
        training_history = self.model.fit(
            data,
            [np.expand_dims(data[0].T,0) for _ in range(4)] +   
            [sin_values2, cos_values, dot_values, k_values, 
             Y_values, Y_values, Y_diff, Y_values, Y_values,
             Y_values, Y_values, Y_values],
            batch_size=batch_size,
            epochs=20000,
            verbose=0,
            callbacks=[callback]
        )

        last_pred, last_true = callback.get_last_predictions()
        print("-----------------------end--------------------")

        return {
            'last_predictions': last_pred,
            'last_true_values': last_true
        }
    
        
    
    def plot_all_compare1(self, data, epoch, time_all):
        """
        Compares two circadian time prediction methods using:
        1. Angular-to-linear time conversion
        2. Linear regression analysis
        3. Geometric distance metrics
        
        Parameters:
            data: Input gene expression matrix (cells × genes)
            epoch: Training epoch identifier 
            time_all: Reference times in radians [0,2π]
            folder_name001: Output directory path
        
        Returns:
            tuple: (predTimes, trueTimes) - Selected predictions and ground truth
        
        Methodology:
            1. Generates predictions from two methods (predict_pseudotime0/predict_pseudotime)
            2. Converts [-π,π] radians to [0,24] hours
            3. Evaluates using:
               - Slope deviation from ideal (|β-1|)
               - Mean distance to y=x line
            4. Selects better prediction based on combined metrics
        """
        
        pseudotime1 = self.predict_pseudotime0(data)
        real_hour1 = (pseudotime1 + 2*np.pi)%(2*np.pi)*24/(2*np.pi)
        real_hour1_order = np.sort(real_hour1)
        pseudotime2 = self.predict_pseudotime(data)
        real_hour2 = (pseudotime2 + 2*np.pi)%(2*np.pi)*24/(2*np.pi)
        real_hour2_order = np.sort(real_hour2)
        x_axis = np.linspace(0, 24.0, len(pseudotime1))
        
       
        def get_metrics(x, y):
            slope = np.polyfit(x, y, 1)[0]
            distances = np.abs(x - y)/np.sqrt(2)
            return abs(slope-1), np.mean(distances)
        
        slope_cha1, dis_abs1 = get_metrics(x_axis, real_hour1_order)
        slope_cha2, dis_abs2 = get_metrics(x_axis, real_hour2_order)
    
        predTimes = real_hour2 if (slope_cha1 > slope_cha2 and dis_abs1 > dis_abs2) else real_hour1
        trueTimes = np.squeeze(time_all)
        return predTimes, trueTimes
    
    
    def predict_gene_exp(self, data: np.ndarray):
        res1 = keras.backend.function(inputs=[self.model.input],
                                      outputs=[self.model.get_layer('output').output]
                                      )([data])
        return res1[0]
    
    
    def predict_pseudotime0(self, data: np.ndarray):
        res = keras.backend.function(inputs=[self.model.input],
                                      outputs=[self.model.get_layer('embedding').output]
                                      )([data])
        return np.arctan2(res[0][0][:, 0], res[0][0][:, 1])
    
    
    def predict_pseudotime(self, data: np.ndarray):
        res = keras.backend.function(inputs=[self.model.input],
                                      outputs=[self.model.get_layer('embedding_true').output]
                                      )([data])
        return np.arctan2(res[0][0][:, 0], res[0][0][:, 1])
    
    def slope_update(self, data: np.ndarray):
        res1 = keras.backend.function(inputs=[self.model.input],
                                      outputs=[self.model.get_layer('discrimination_layer').output]
                                      )([data])
        return res1[0][0][0]
    
    
    def emb_x_update(self, data: np.ndarray):
        res1 = keras.backend.function(inputs=[self.model.input],
                                      outputs=[self.model.get_layer('embedding').output]
                                  )([data])
        return res1[0]
def call(self,input_matrix):
    return self.predict_gene_exp(input_matrix)


    
    
