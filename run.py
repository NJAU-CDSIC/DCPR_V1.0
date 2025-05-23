

# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import pickle
import numpy as np
import pandas as pd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from DCPR_codes.data_shuffle import shuffle_time_series
from DCPR_codes.preprocess import load_and_preprocess_data
from DCPR_codes.base import model_train_single
from DCPR_codes.result import get_corrected_phase
from Visualization.Scripts import batch_process
from DCPR_codes.make_dir import make_dir_all


def run_pipeline(args):

    make_dir_all(args.O)
    processed_data = load_and_preprocess_data(args.X, args.V, args.S, args.O)
    predicted_hours, last_observed = model_train_single(processed_data,[args.loss_parameters])
    adjusted_times, ground_truth_times = get_corrected_phase(predicted_hours, last_observed, processed_data,label='median',correct_way='start_phase',folder_name=args.O)
    analysis_groups = [{'tissue': 'GSE161566','condition': 'per1','replicate': 'per1','true_times': {'DCPR': ground_truth_times},'pred_times': {'DCPR': adjusted_times}}]
    AUC_values, median_absolute_errors = batch_process(analysis_groups, save_base_path=args.O)
    

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--X', default='Datasets/Real datasets/GEO datasets/GSE161566.csv')
    parser.add_argument('--V', default='time')
    parser.add_argument('--S',default='Supplementary files/Seed genes/GSE161566_seed_genes.xlsx')
    parser.add_argument('--O', default='results_final')
    parser.add_argument('--loss_parameters', nargs='+', type=int, default=[0,0,0,0,0,1,0,0,0,1,0,0,0,1,1,1],
                        help='Loss parameters as a list of 16 integers')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)






