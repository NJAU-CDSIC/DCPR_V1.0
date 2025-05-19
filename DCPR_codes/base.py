import os
import numpy as np
import pandas as pd
import tensorflow as tf
from DCPR_codes.main import Predict
from math import pi


def generate_time_features(sample_count):
    t_inter = 24.0 / (sample_count - 1)
    time_points = np.arange(0, 24, t_inter)
    if len(time_points) != sample_count:
        time_points = np.hstack((time_points, 24))
    radians = time_points / 24 * 2 * np.pi
    sin_values = np.expand_dims(np.sin(radians), axis=0).T
    cos_values = np.expand_dims(np.cos(radians), axis=0).T
    dot_values = sin_values * cos_values
    slope_values = sin_values / cos_values
    return (
        tf.constant(sin_values, dtype=tf.float32),
        tf.constant(cos_values, dtype=tf.float32),
        tf.constant(dot_values, dtype=tf.float32),
        tf.constant(slope_values, dtype=tf.float32)
    )

def compute_pred_hour(pseudotime):
    real_theta = (pseudotime + 2 * pi) % (2 * pi)
    return real_theta / (2 * pi) * 24


def model_train_single(processed_data, loss_p):
    
    (matrix_new, time_all_new, Y_values, error_values,theta_umap_diff, Y_diff, Y_diff_mean, Y_std_error_values,Y_values_all, Amp_all, Phi_all, q_all) = processed_data
    
    sin_values, cos_values, dot_values, slope_values = generate_time_features(matrix_new.shape[1])
    
    Omega = np.array(matrix_new)
    model = Predict(Omega, 1, 6, 2, 0, sin_values)
   
    result_pre = model.train(
        np.expand_dims(Omega, 0),
        np.expand_dims(sin_values, 0),
        np.expand_dims(cos_values, 0),
        np.expand_dims(dot_values, 0),
        np.expand_dims(slope_values, 0),
        np.expand_dims(Y_values, 0),
        np.expand_dims(error_values, 0),
        np.expand_dims(Y_diff, 0),
        np.expand_dims(Y_diff_mean, 0),
        np.expand_dims(Y_std_error_values, 0),
        np.expand_dims(Y_values_all, 0),
        np.expand_dims(Amp_all, 0),
        np.expand_dims(Phi_all, 0),
        np.expand_dims(q_all, 0),
        loss_p, time_all_new, 
        epochs=20000, verbose=100, rate=2e-4
    )
    
    last_pred = result_pre['last_predictions']
    last_true = result_pre['last_true_values']
    

    if np.all(last_pred < 10):
        
        last_pred = compute_pred_hour(last_pred)

    return last_pred, last_true
