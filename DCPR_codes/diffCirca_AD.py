# -*- coding: utf-8 -*-
"""
Created on Sun May 25 00:20:03 2025

@author: hx
"""



import os  
import pandas as pd
import numpy as np

import CosinorPy
from CosinorPy import cosinor


def convert_to_float_list(x):
    """
    Convert various types of input (string, list, Series) into a numpy array of floats.

    Parameters
    ----------
    x : str, list, Series, or numpy array
        The input which may represent a list of numeric values, possibly as a string.

    Returns
    -------
    numpy.ndarray
        A numpy array of floats parsed from the input.
    """
    if isinstance(x, str):
        x = x.strip('[]').strip()
        values = x.split(', ') if ', ' in x else x.split()
        float_list = [float(i) for i in values if i]
    elif isinstance(x, (list, pd.Series)):
        float_list = [float(i) for i in x]
    else:
        float_list = x

    return np.array(float_list)




def get_cosinorPY_results(matrix_input, time_input, fold_name_save):
    """
    Perform cosinor analysis on rhythmic gene expression data under control and AD conditions.

    Parameters
    ----------
    matrix_input : DataFrame
        Input gene expression matrix where rows are genes and columns are expression values across time points.

    time_input : DataFrame
        Matrix containing time points for both control and AD conditions.

    fold_name_save : str
        Directory path to save the resulting CSV file containing fitted rhythmicity metrics.

    Returns
    -------
    DataFrame
        A merged DataFrame of fitted rhythmicity metrics for control and AD groups.
    """

    import pandas as pd
    import numpy as np
    import os
    import CosinorPy
    from CosinorPy import cosinor

    
    rename_mapping = {'Gene.1': 'Gene'}
    matrix_input.rename(columns=rename_mapping, inplace=True)

    
    matrix_sort_AD1 = matrix_input.copy()
    gene_cols = [i for i, col in enumerate(matrix_input.columns) if col == 'Gene']
    AD_gene_index = gene_cols[1]  

   
    CON_t = time_input.iloc[1:, 1].dropna().to_numpy(dtype=float)
    CON_len = len(CON_t)
    CON_gene = matrix_sort_AD1.iloc[:, 1]
    CON_matrix = matrix_input.iloc[:, 2:(CON_len+2)]
    CON_time_sort_index = np.argsort(CON_t)
    CON_time_sort = CON_t[CON_time_sort_index]
    CON_matrix_sort = CON_matrix.iloc[:, CON_time_sort_index]

    
    AD_t = time_input.iloc[1:, 2].dropna().to_numpy(dtype=float)
    AD_gene = matrix_sort_AD1.iloc[:, 2 + CON_len]
    AD_matrix = matrix_input.iloc[:, (3 + CON_len):]
    AD_time_sort_index = np.argsort(AD_t)
    AD_time_sort = AD_t[AD_time_sort_index]
    AD_matrix_sort = AD_matrix.iloc[:, AD_time_sort_index]

    df_best_fit_all = []

    for i in range(len(CON_gene)):
        gene_i = CON_gene[i]

        con_y = CON_matrix_sort.values[i, :]
        AD_y = AD_matrix_sort.values[i, :]

        
        con_df = pd.DataFrame({'x': CON_time_sort, 'y': con_y})
        con_df.insert(0, 'test', [gene_i] * len(CON_time_sort))
        cosinor.periodogram_df(con_df)
        con_results = cosinor.fit_group(con_df, n_components=1,
                                        period=list(range(20, 29)), plot=False)
        con_best = cosinor.get_best_fits(con_results, n_components=[1],
                                         criterium='RSS', reverse=False)
        con_best.rename(columns={'test': 'Gene'}, inplace=True)
        con_final = con_best[['Gene', 'p', 'period', 'amplitude', 'mesor',
                              'acrophase', 'peaks', 'troughs']].copy()
        con_final.rename(columns={'p': 'p_val', 'peaks': 'peak', 'troughs': 'trough'}, inplace=True)
        con_final['peak'] = convert_to_float_list(con_final['peak']).tolist()
        con_final['trough'] = convert_to_float_list(con_final['trough']).tolist()
        con_final.loc[con_final['p_val'] >= 0.05, con_final.columns[2:]] = 0

        
        AD_df = pd.DataFrame({'x': AD_time_sort, 'y': AD_y})
        AD_df.insert(0, 'test', [gene_i] * len(AD_time_sort))
        cosinor.periodogram_df(AD_df)
        AD_results = cosinor.fit_group(AD_df, n_components=1,
                                       period=list(range(20, 29)), plot=False)
        AD_best = cosinor.get_best_fits(AD_results, n_components=[1],
                                        criterium='RSS', reverse=False)
        AD_best.rename(columns={'test': 'Gene'}, inplace=True)
        AD_final = AD_best[['Gene', 'p', 'period', 'amplitude', 'mesor',
                            'acrophase', 'peaks', 'troughs']].copy()
        AD_final.rename(columns={'p': 'p_val', 'peaks': 'peak', 'troughs': 'trough'}, inplace=True)
        AD_final['peak'] = convert_to_float_list(AD_final['peak']).tolist()
        AD_final['trough'] = convert_to_float_list(AD_final['trough']).tolist()
        AD_final.loc[AD_final['p_val'] >= 0.05, AD_final.columns[2:]] = 0

        merged_df = pd.merge(con_final, AD_final, on='Gene', suffixes=('_control', '_AD'))
        df_best_fit_all.append(merged_df)

    df_final = pd.concat(df_best_fit_all, ignore_index=True)
    os.makedirs(fold_name_save, exist_ok=True)
    df_final.to_csv(f'{fold_name_save}/df_con_AD.csv', index=False)

    return df_final



'''                            

#R:difficircadian_process.R


options(encoding = "UTF-8")
.libPaths("D:/R-2023/R-4.3.0/library")
library(diffCircadian)
library(nloptr)


get_rhythmicity_diffcircadian <- function(df_brain_region, matrix_input, time_input, fold_name_save) {
  
  matrix_sort_AD <- matrix_input
  time_sort_AD <- time_input
  
  
  matrix_sort_AD1 <- matrix_sort_AD
  
  gene_col <- which(colnames(matrix_sort_AD) == 'Gene')
  AD_gene_index <- gene_col[2]
  matrix_sort_AD <- matrix_sort_AD[, -AD_gene_index]
  
  
  
  CON_t <- time_sort_AD[, 2]
  CON_t <- na.omit(CON_t)
  CON_len <- length(CON_t)
  CON_gene <- matrix_sort_AD1[, 2]
  CON_matrix <- matrix_sort_AD[, 3:(CON_len+2)]
  CON_time_sort_index <- order(CON_t)
  CON_time_sort <- CON_t[CON_time_sort_index]
  CON_matrix_sort <- CON_matrix[, CON_time_sort_index]
  
  
  
  AD_t <- time_sort_AD[, 3]
  AD_t <- na.omit(AD_t)
  AD_gene <- matrix_sort_AD1[, (3+CON_len)]
  AD_matrix <- matrix_sort_AD[, (CON_len + 3):ncol(matrix_sort_AD)]
  AD_time_sort_index <- order(AD_t)
  AD_time_sort <- AD_t[AD_time_sort_index]
  AD_matrix_sort <- AD_matrix[, AD_time_sort_index]
  
  
  p_columns <- which(colnames(df_brain_region) == "p_val")
  second_p_col_index <- p_columns[2]
  
  CON_data <- df_brain_region[, 3:(second_p_col_index - 1)]
  AD_data <- df_brain_region[, second_p_col_index:ncol(df_brain_region)]
  CON_data['period'][1,]
  
  diff_all <- data.frame()
  
  
  for (g_ii in 1:nrow(df_brain_region)) {
    
    gene <- df_brain_region['Gene'][g_ii,]
    
    CON_g_ii <- t(as.data.frame(t(CON_data[g_ii, ])))
    
    
    AD_g_ii <- t(as.data.frame(t(AD_data[g_ii, ])))
    
    diff_all0 <- df_brain_region[g_ii, ]
    
    if (any(CON_g_ii[, 2:ncol(CON_g_ii)] != 0) && all(AD_g_ii[, 2:ncol(AD_g_ii)] == 0)) {
      mode_ij <- "Loss of rhythmicity"
    } else if (all(CON_g_ii[, 2:ncol(CON_g_ii)] == 0) && any(AD_g_ii[, 2:ncol(AD_g_ii)] != 0)) {
      mode_ij <- "Gain of rhythmicity"
    }else if (any(CON_g_ii[, 2:ncol(CON_g_ii)] != 0) && any(AD_g_ii[, 2:ncol(AD_g_ii)] != 0)) {
      
      if (abs(as.data.frame(CON_g_ii)$period - as.data.frame(AD_g_ii)$period) < 2){
        
        diff_period <- mean(as.data.frame(CON_g_ii)$period, as.data.frame(AD_g_ii)$period) 
        
        
        
        yy1 <- as.vector(t(CON_matrix_sort[CON_gene == gene, ]))
        tt1 <- CON_time_sort
        
        yy2 <- as.vector(t(AD_matrix_sort[AD_gene == gene, ]))
        
        tt2 <- AD_time_sort
        
        LR <- LR_rhythmicity(tt1, yy1) 
        diff_fit <- LR_diff(tt1, yy1, tt2, yy2, period = diff_period, type="fit")
        
        dif_amp <- LR_diff(tt1, yy1, tt2, yy2, period = diff_period, type="amplitude")
        
        dif_pha <- LR_diff(tt1, yy1, tt2, yy2, period = diff_period, type="phase")
        
        dif_mesor <- LR_diff(tt1, yy1, tt2, yy2, period = diff_period, type="basal")
        
        mode_ij = c()
        
        
        if (dif_amp$pvalue < 0.05){
          
          mode_ij <- c(mode_ij, "Amplitude change")
          
        }
        
        if(dif_pha$pvalue< 0.05){
          
          mode_ij <- c(mode_ij, "Phase shift")
          
        }
        
        if(dif_mesor$pvalue < 0.05){
          
          mode_ij <- c(mode_ij, "Base shift")
          
        }
        
        if(length(mode_ij) == 0){
          
          mode_ij <- c(mode_ij, "No difference")
        }
        
        mode_ij = paste(mode_ij, collapse = ";")
        
      }else{
        
        
        df_CON_g_ii <- as.data.frame(CON_g_ii)
        df_AD_g_ii <- as.data.frame(AD_g_ii)
        
        CON_amp <- df_CON_g_ii['amplitude'][[1]]
        AD_amp <- df_AD_g_ii['amplitude'][[1]]
        
        CON_T <- df_CON_g_ii['period'][[1]]
        AD_T <- df_AD_g_ii['period'][[1]]
        
        CON_pha <- df_CON_g_ii['peak'][[1]]
        AD_pha <- df_AD_g_ii['peak'][[1]]
        CON_pha_pro <- CON_pha/CON_T
        AD_pha_pro <- AD_pha/AD_T
        
        CON_mes <- df_CON_g_ii['mesor'][[1]]
        AD_mes <- df_AD_g_ii['mesor'][[1]]
        
        
        
        
        mode_ij = c('period change')
        
        
        mode_ij = paste(mode_ij, collapse = ";")
        
      }
    }else{
      
      mode_ij <- "No difference"
    }
    diff_all0$mode <- mode_ij
    diff_all0$paper_brain_region <- 'brain_region'
    diff_all <- rbind(diff_all, diff_all0)
  }
  
  filepath2 <- file.path(fold_name_save, "diff_all2.csv")
  write.csv(diff_all, filepath2, row.names = FALSE)
  return(diff_all)
  
}



args <- commandArgs(trailingOnly = TRUE)

df_brain_region_file <- args[1]
matrix_input_file <- args[2]
time_input_file <- args[3]
fold_name_save <- args[4]

df_brain_region <- read.csv(df_brain_region_file, header = TRUE, check.names = FALSE)
matrix_input <- read.csv(matrix_input_file, header = TRUE, check.names = FALSE)
time_input <- read.csv(time_input_file, header = TRUE, check.names = FALSE)


get_rhythmicity_diffcircadian(df_brain_region, matrix_input, time_input, fold_name_save)

'''


def get_rhythmicity_diffcircadian(matrix_input, time_input, fold_name_save):  
    
    
    import subprocess
    
    df_brain_region = get_cosinorPY_results(matrix_input, time_input, fold_name_save)
    df_brain_region.to_csv("df_brain_region.csv", index=True)
    matrix_input.to_csv("matrix_input.csv", index=False)
    time_input.to_csv("time_input.csv", index=False, header = False)
    result = subprocess.run(['Rscript', 'difficircadian_process.R', 'df_brain_region.csv','matrix_input.csv','time_input.csv', fold_name_save], capture_output=True, text=True)
    
    
    return result
    
