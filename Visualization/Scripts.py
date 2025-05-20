

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import CosinorPy
from CosinorPy import cosinor
import os


def calculate_auc_mederr(true_times, pred_times, folder_name):
    """
    Compute the Area Under Curve (AUC) and median absolute error (MedAE)
    for predicted time points against true time points.

    Parameters:
    ----------
    true_times : array-like
        True time points (in hours).
    pred_times : array-like
        Predicted time points (in hours).

    Returns:
    -------
    auc : float
        Normalized AUC value.
    mederr : float
        Median absolute error between predicted and true times (in hours).
    hrsoff : ndarray
        Array of thresholds (in hours) for cumulative error.
    fracacc : ndarray
        Cumulative percentage of samples within each threshold.
    """
    hrerr = np.abs(pred_times - true_times) % 24
    hrerr = np.minimum(hrerr, 24 - hrerr)
    
    hrerr = np.sort(hrerr)
    hrsoff = np.insert(hrerr, 0, 0)
    hrsoff = np.append(hrsoff, 12)
    
    fracacc = np.array([100 * np.sum(hrerr > hrtol) / len(hrerr) for hrtol in hrsoff])
    
    norm_fracacc = (100 - fracacc) / 100
    norm_hrsoff = hrsoff / 12
    auc = np.sum(norm_fracacc[:-1] * np.diff(norm_hrsoff))
    
    mederr = np.median(hrerr)
    print(f"Pipeline completed. AUC: {auc:.3f}, MedAE: {mederr:.3f}")

    output_dir = os.path.join(folder_name)
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame({'AUC': [auc],'MedAE': [mederr]})
    df.to_csv(f"{output_dir}/evaluate.csv", index=False)
    
    return auc, mederr, hrsoff, fracacc


def plot_cdf(hrsoff_dict, fracacc_dict, auc_dict, mederr_dict, 
             tissue_name, condition, replicate_id, save_dir=None, colors_rgb=None):
    """
    Plot cumulative distribution function (CDF) curves for different prediction methods.

    Parameters:
    ----------
    hrsoff_dict : dict
        Dictionary mapping method names to arrays of time thresholds.
    fracacc_dict : dict
        Dictionary mapping method names to cumulative accuracy percentages.
    auc_dict : dict
        Dictionary mapping method names to AUC values.
    mederr_dict : dict
        Dictionary mapping method names to median errors.
    tissue_name : str
        Name of tissue or dataset.
    condition : str
        Experimental condition label.
    replicate_id : str
        Replicate name or identifier.
    save_dir : str, optional
        Directory to save plots. If None, plot will be displayed.
    colors_rgb : list of tuple, optional
        RGB colors for plotting different methods.
    """
    
    if colors_rgb is None:
        colors_rgb = [
            (100, 146, 205), (39, 173, 62), (234, 113, 105),
            (222, 170, 76), (244, 227, 195), (238, 28, 0)
        ]
    
    colors_normalized = [(r/255, g/255, b/255) for r, g, b in colors_rgb]
    
    fig, ax = plt.subplots(figsize=(3.0, 3.0), dpi=300)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('black')
    
    ax.tick_params(axis='both', width=0.5, length=2)
    rcParams['font.family'] = 'Times New Roman'
    ax.set_xlim([-0.3, 12.5])
    ax.set_ylim([-2, 102])
    ax.plot([-0.3, 12.5], [-2, 102], color='lightgray', linestyle='-', linewidth=0.7, alpha=0.6, zorder=0)
    method_names = {
        'DCPR': 'DP',
        'CYCLOPS': 'CP',
        'CHIRAL': 'CR',
        'Cyclum': 'CM'
    }
    
    color_assign = {
        'DCPR': colors_normalized[5],  
        'Cyclum': 'orange',  
        'CYCLOPS': colors_normalized[0],  
        'CHIRAL': colors_normalized[1]   
    }
    
    lines_dict = {}
    for method in ['CYCLOPS', 'CHIRAL', 'Cyclum', 'DCPR']:
        if method in hrsoff_dict:
            linestyle = '-'  
            label = f"{method_names[method]}: {auc_dict[method]:.3f},   {mederr_dict[method]:.3f}" if method != 'Cyclum' else f"{method_names[method]}: {auc_dict[method]:.3f},   {mederr_dict[method]:.3f}"
            line, = ax.plot(hrsoff_dict[method], 
                            100 - fracacc_dict[method], 
                            linestyle=linestyle, 
                            color=color_assign[method], 
                            lw=1.0,
                            label=label)
            lines_dict[method] = line
    legend_order = ['DCPR', 'CYCLOPS', 'CHIRAL', 'Cyclum']
    handles = [lines_dict[method] for method in legend_order if method in lines_dict]
    blank_line = plt.Line2D([], [], color='none')
    handles.insert(0, blank_line)
    labels = ['        AUC,  MedAE'] + [h.get_label() for h in handles[1:]]
    
    legend = ax.legend(handles=handles, labels=labels, loc='lower right', fontsize='large', 
                       prop={'family': 'Times New Roman', 'size': 7})
    
    legend.get_frame().set_linewidth(0.3)
    legend.get_frame().set_edgecolor('#333333')
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    ax.set_xticks(np.arange(0, 13, 2))
    ax.set_yticks(np.arange(0, 105, 20))
    ax.tick_params(axis='both', labelsize=8)
    ax.set_xlabel("Correct to within (hrs)", fontname='Times New Roman', fontsize=9)
    sample_size = None
    for method in hrsoff_dict:
        if hrsoff_dict[method] is not None:  
            sample_size = len(hrsoff_dict[method]) - 2
            break  

    if sample_size is None:
        raise ValueError("All methods in hrsoff_dict have None values!")  
    ax.set_ylabel(f"% correct (N = {sample_size})", fontname='Times New Roman', fontsize=9)
    
    title = "Absolute error CDF"
    ax.set_title(title, fontname='Times New Roman', fontsize=10, fontweight='bold')
    fig.subplots_adjust(left=0.15, bottom=0.15)
    
    if save_dir is not None:
        save_folder = os.path.join(save_dir, replicate_id)
        os.makedirs(save_folder, exist_ok=True)
        file_name = os.path.join(save_folder, f"{condition}_cdf.tiff")
        fig.savefig(file_name, dpi=300)
        plt.show()
        plt.close(fig)
    else:
        plt.show()


def plot_prediction(true_times, pred_times, tissue_name, condition, replicate_id, save_dir=None, colors_rgb=None):
    """
    Plot predicted versus true circadian times with confidence intervals.
    
    This function generates a scatter plot comparing the predicted circadian times 
    with the true times, including ±2h and ±4h confidence intervals, and saves 
    the figure if a directory is specified.

    Parameters
    ----------
    true_times : array-like
        Ground truth circadian times.
    pred_times : array-like
        Predicted circadian times.
    tissue_name : str
        Name of the tissue or sample (used for plot title).
    condition : str
        Experimental condition (used for labeling and saving).
    replicate_id : str
        Replicate identifier (used in saving figure).
    save_dir : str, optional
        Directory path where the figure will be saved. If None, the figure is displayed but not saved.
    colors_rgb : list of tuples, optional
        List of RGB color codes (0-255) to customize the plot colors.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object.
    ax : matplotlib.axes.Axes
        The generated axes object.

    """

    if colors_rgb is None:
        colors_rgb = [
            (225, 225, 225), (187, 187, 187), (242, 242, 242),
            (205, 205, 205), (186, 186, 186), (222, 170, 76),
            (244, 227, 195), (123, 186, 237), (208, 231, 249)
        ]
    colors_normalized = [(r/255, g/255, b/255) for r, g, b in colors_rgb]
    
    fig, ax = plt.subplots(figsize=(3.0, 3.0), dpi=300)
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('black')
    
    ax.tick_params(axis='both', width=0.5, length=2)
    
    rcParams['font.family'] = 'Times New Roman'
    
    ax.plot(true_times, pred_times, 'o', markersize=5,
            color='black',
            markeredgecolor=colors_normalized[5],
            markerfacecolor=colors_normalized[6])
    
    
    ax.set_xlabel("True Time (h)", fontsize=9)
    ax.set_ylabel("Predicted Time (h)", fontsize=9)
    
    title = tissue_name
    ax.set_title(title, fontsize=10, fontweight='bold')
    
    ax.set_xticks(np.arange(0, 27, 2))
    ax.set_yticks(np.arange(0, 27, 2))
    ax.tick_params(labelsize=8)
    
    x = np.arange(-1, 26, 1)
    y = np.arange(-1, 26, 1)
    ax.plot([-1, 24.5], [-1, 24.5], linestyle='--', linewidth=0.5, color=colors_normalized[4])
    ax.fill_between(x, y-2, y+2, color=colors_normalized[0], alpha=1.0)
    ax.plot(x, y-2, '--', linewidth=0.5, color=colors_normalized[1])
    ax.plot(x, y+2, '--', linewidth=0.5, color=colors_normalized[1])
    ax.fill_between(x, y+2, y+4, color=colors_normalized[2], alpha=1.0)
    ax.fill_between(x, y-4, y-2, color=colors_normalized[2], alpha=1.0)
    ax.plot(x, y+4, ':', linewidth=0.5, color=colors_normalized[3])
    ax.plot(x, y-4, ':', linewidth=0.5, color=colors_normalized[3])
    ax.fill_between(x, y + 24, y + 22, color=colors_normalized[0], alpha=1.0)
    ax.fill_between(x, y + 24, y + 26, color=colors_normalized[0], alpha=1.0)
    ax.fill_between(x, y + 22, y + 20, color=colors_normalized[2], alpha=1.0)
    ax.plot(x, y + 24, '--', linewidth=0.5, color=colors_normalized[1])
    ax.plot(x, y + 22, '--', linewidth=0.5, color=colors_normalized[1])
    ax.plot(x, y + 20, ':', linewidth=0.5, color=colors_normalized[3])

    ax.fill_between(x, y - 24, y - 22, color=colors_normalized[0], alpha=1.0)
    ax.fill_between(x, y - 24, y - 26, color=colors_normalized[0], alpha=1.0)
    ax.fill_between(x, y - 22, y - 20, color=colors_normalized[2], alpha=1.0)
    ax.plot(x, y - 24, '--', linewidth=0.5, color=colors_normalized[1])
    ax.plot(x, y - 22, '--', linewidth=0.5, color=colors_normalized[1])
    ax.plot(x, y - 20, ':', linewidth=0.5, color=colors_normalized[3])

    ax.set_xlim([-1, 24.5])
    ax.set_ylim([0, 24])
    
    fig.subplots_adjust(left=0.15, bottom=0.15)
    
    if save_dir is not None:
        output_dir = os.path.join(save_dir, str(replicate_id))
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, f'{condition}.tiff')
        fig.savefig(fig_path, dpi=300)
        plt.show()
        plt.close(fig)
    else:
        plt.show()
    
    return fig, ax



def batch_process(groups_data, save_base_path):
    """
    Batch process multiple groups of data, generating CDF curves and true-prediction scatter plots.

    Parameters
    ----------
    groups_data : list of dict
        Each dict contains information for one group, for example:
        {
            'tissue': 'SD1',
            'condition': 'SD1',
            'replicate': 'rep1',
            'true_times_dict': {
                'CYCLOPS': true_times_cyclops,
                'CHIRAL': true_times_chiral,
                'Cyclum': true_times_cyclum,
                'DCPR': true_times_dcpr
            },
            'pred_times_dict': {
                'CYCLOPS': pred_times_cyclops,
                'CHIRAL': pred_times_chiral,
                'Cyclum': pred_times_cyclum,
                'DCPR': pred_times_dcpr
            }
        }
    save_base_path : str
        The base directory where plots will be saved.
    """
    
    for group in groups_data:
        tissue = group['tissue']
        condition = group['condition']
        replicate = group['replicate']
        true_times = group['true_times']
        pred_times_dict = group['pred_times']
        
        auc_dict = {}
        mederr_dict = {}
        hrsoff_dict = {}
        fracacc_dict = {}
        
        for method, pred_times in pred_times_dict.items():
            
            if true_times.get(method) is not None and pred_times is not None:
                auc, mederr, hrsoff, fracacc = calculate_auc_mederr(true_times[method], pred_times, save_base_path)
                auc_dict[method] = auc
                mederr_dict[method] = mederr
                hrsoff_dict[method] = hrsoff
                fracacc_dict[method] = fracacc

        if auc_dict:  
            plot_cdf(hrsoff_dict, fracacc_dict, auc_dict, mederr_dict, 
                     tissue_name=tissue, condition=condition, replicate_id=replicate, 
                     save_dir=save_base_path)
        
        for method in pred_times_dict:
            if method in true_times and true_times[method] is not None and pred_times_dict[method] is not None:
                plot_prediction(
                    true_times[method], 
                    pred_times_dict[method],
                    tissue_name=tissue,
                    condition=condition,
                    replicate_id=replicate,
                    save_dir=save_base_path
                )

    return auc, mederr




def plot_circadian_expression(method_name, all_pred_times, all_expression, gene_symbols, 
                             core_clock_genes, output_dir="results"):
    """
    Plot circadian expression patterns with cosine fitting for core clock genes
    
    Parameters:
    - method_name: Name of computational method (e.g., 'CHIRAL')
    - all_pred_times: List of predicted time arrays from all subsets
    - all_expression: List of expression matrices from all subsets
    - gene_symbols: Array of gene symbols matching expression matrix rows
    - core_clock_genes: List of core clock gene symbols to plot
    - output_dir: Directory to save plots
    """
    
    pred_times = np.concatenate(all_pred_times)
    expression = np.hstack(all_expression)
    sort_idx = np.argsort(pred_times.flatten())
    pred_times = pred_times[sort_idx]
    expression = expression[:, sort_idx]
    os.makedirs(output_dir, exist_ok=True)
    for gene in core_clock_genes:
        if gene not in gene_symbols:
            continue
            
        idx = np.where(gene_symbols == gene)[0][0]
        gene_exp = expression[idx]
        
        df = pd.DataFrame({
            'test': [gene]*len(pred_times),  # 必须用 'test'
            'x': pred_times.flatten(),
            'y': gene_exp
        })
        
        results = cosinor.fit_group(df, n_components=1, plot=False)
        results["test_num"] = results["test"].str.extract("(\d+)").astype(int)
        results = results.sort_values("test_num").drop("test_num", axis=1)
        
        
        fig, ax = plt.subplots(figsize=(3, 2), dpi=300)
        
        ax.scatter(pred_times, gene_exp, s=10, color='blue', alpha=0.6, label='Expression')
        
        if results['p'].values[0] < 0.05:
            x_fit = results['X_plot_fit'].values[0]
            y_fit = results['Y_plot_fit'].values[0]
            ax.plot(x_fit, y_fit, 'r-', linewidth=1.5, label='Cosine Fit')
            peak = results['peaks'].values[0][0]
            ax.axvline(x=peak, color='gray', linestyle='--', alpha=0.7)
        
        p_value_sci = "{:.2e}".format(results['p'].values[0])
        ax.set_xlabel('Circadian Time (h)', fontsize=9)
        ax.set_ylabel('Expression', fontsize=9)
        #ax.set_title(f'{gene} - {method_name}', fontsize=10)
        ax.set_title(f'{gene}_{method_name}_(Pvalue: {p_value_sci})', fontname='Times New Roman', fontweight='bold', fontsize=10)
        ax.set_xlim([0, 24])
        ax.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{method_name}_{gene}.png", dpi=300)
        plt.show()

def plot_seamless_bar_chart(df, ylabel, save_path=None):
    """
    Plot a seamless bar chart comparing multiple methods across different datasets.
    The chart is designed with the following features:
    - No gaps between bars.
    - No values displayed on top of the bars.
    - No diagonal fill patterns.
    - Horizontal axis labels are not rotated.

    Parameters:
    ----------
    df : pandas.DataFrame
        A DataFrame containing the data to be plotted. It must include a 'Datasets' column
        and one or more method columns to represent the comparison.
    ylabel : str
        The label for the y-axis of the chart.
    save_path : str, optional
        If provided, the chart will be saved to the specified path. If None, the chart will
        only be displayed but not saved.

    Returns:
    -------
    fig : matplotlib.figure.Figure
        The figure object representing the chart.
    ax : matplotlib.axes._axes.Axes
        The axis object used to create the chart.
    """
    
    title = "Comparison of Methods Across Datasets"
    
    datasets = df['Datasets'].tolist()
    methods = [col for col in df.columns if col != 'Datasets']
    data = df[methods].values  # Extract the values for each method
    
    n_methods = len(methods)
    colors_rgb = [
        (255, 131, 100),  
        (78, 172, 91),    
        (33, 105, 247),  
        (247, 193, 67)    
    ]
    
    colors = [(r/255, g/255, b/255) for r, g, b in colors_rgb]
    width = 1.2
    bar_width = width / n_methods  

    x = np.arange(0, 2*len(datasets), 2)  
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    
    for i, method in enumerate(methods):
        ax.bar(x + i*bar_width, data[:, i], bar_width,
               color=colors[i],
               edgecolor='None',
               linewidth=0,
               alpha=0.9,
               label=method)

    ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
    ax.set_xticks(x + width/2 - bar_width/2)  
    ax.set_xticklabels(datasets, rotation=0, ha='center', fontsize=12, fontweight='bold')
    
    buffer = data.max() * 0.15
    ax.set_ylim(0, data.max() + buffer)
    
    ax.tick_params(axis='y', labelsize=12)  
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')  

    plt.tight_layout()
    
    if save_path:
        save_kwargs = {
            'dpi': 300,
            'bbox_inches': 'tight',
            'format': 'tiff',
            'pil_kwargs': {'compression': 'tiff_lzw'}
        }
        plt.savefig(save_path, **save_kwargs)

    plt.show()
    
    return fig, ax
