# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:22:03 2024

@author: hx
"""




import numpy as np
import pandas as pd
import tensorflow as tf
import os


#设置全局随机种子
np.random.seed(42)
tf.random.set_seed(42)


def make_dir_all(fold_name):
    os.makedirs(f"{fold_name}", exist_ok=True)
