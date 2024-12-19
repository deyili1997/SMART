import numpy as np
from scipy.stats import ttest_rel
import string

def save_figure(fig, folder_name: str, figure_name: str) -> None:
    output_dir = "/home/lideyi/AKI_SMART/SMART/High_res_figures/%s"%(folder_name)
    file_name = "%s.tif"%(figure_name)
    file_path = f"{output_dir}/{file_name}"
    print("Figure saved at: ", file_path)
    fig.savefig(file_path, format='tiff', dpi=120)
    

def add_subplot_index(axs, n_rows: int, n_cols: int) -> None:
    """
    Adds alphabetical labels (A, B, C, ...) to each subplot.

    Parameters:
        axs: array-like
            The array of subplot axes, typically returned by plt.subplots.
        n_rows: int
            Number of rows of subplots.
        n_cols: int
            Number of columns of subplots.

    Returns:
        None
    """
    labels = list(string.ascii_lowercase)  # Lowercase letters for labels
    label_idx = 0  # Track the current label index

    if n_rows == 1:  # Handle single row (axs is 1D)
        for col in range(n_cols):
            axs[col].text(
                -0.1, 1.1,  # Position relative to the axes
                labels[label_idx],  # Current label
                transform=axs[col].transAxes,  # Use subplot's coordinate system
                fontsize=14,
                fontweight='bold',
                va='top',  # Vertical alignment
                ha='right'  # Horizontal alignment
            )
            label_idx += 1
    else:  # Handle multiple rows (axs is 2D)
        for row in range(n_rows):
            for col in range(n_cols):
                axs[row, col].text(
                    -0.1, 1.1,  # Position relative to the axes
                    labels[label_idx],  # Current label
                    transform=axs[row, col].transAxes,  # Use subplot's coordinate system
                    fontsize=14,
                    fontweight='bold',
                    va='top',  # Vertical alignment
                    ha='right'  # Horizontal alignment
                )
                label_idx += 1
    

def plot_metric_along_k(ax, k_sizes, performance_nw, performance_wt, title, metric, model_name: str):
    p_val = one_tailed_t_test(performance_nw, performance_wt)
    plot_smooth_curve(ax, k_sizes, performance_wt, label ="%s (w/ O)"%(model_name), color = "red")
    plot_smooth_curve(ax, k_sizes, performance_nw, label =f"%s (w/o O, $\\mathit{{{p_val}}}$)"%(model_name), color = "blue")
    ax.legend(loc = 'lower right')
    ax.set_xlabel("k")
    ax.set_ylabel(metric)
    ax.set_title(title)

def one_tailed_t_test(control_performance: list, candidate_performance: list):

    # Perform paired t-test (two-tailed by default)
    t_stat, p_value_two_tailed = ttest_rel(candidate_performance, control_performance)

    # Convert to one-tailed p-value by dividing by 2
    p_value_one_tailed = p_value_two_tailed / 2

    # Check if the test statistic supports the hypothesis that candidate > control
    if t_stat <= 0:  # If t_stat is not positive, candidate does not outperform control
        return 'NS'

    # Determine the appropriate p-value threshold
    if p_value_one_tailed < 0.001:
        return 'p<0.001'
    elif p_value_one_tailed < 0.01:
        return 'p<0.01'
    elif p_value_one_tailed < 0.05:
        return 'p<0.05'
    else:
        return 'NS'
    
def plot_smooth_curve(ax, X_vals, y_vals, label, color, alpha = 1.0):
    # Fit a polynomial of the specified degree
    coeffs = np.polyfit(X_vals, y_vals, 5)
    poly_func = np.poly1d(coeffs)
    
    # Generate smooth x-values and compute corresponding y-values
    X_smooth = np.linspace(np.min(X_vals), np.max(X_vals), 500)
    y_smooth = poly_func(X_smooth)
    
    # Plot the fitted curve
    ax.plot(X_smooth, y_smooth, label=label, color=color, linewidth=1, alpha = alpha)
    
    # Plot the original points with markers
    ax.scatter(X_vals, y_vals, color=color, marker='.', s=30, label="_nolegend_", alpha = alpha)
    
    
def plot_optim_vs_controls(ax, k_sizes: list, control_performance: list, control_names: list, 
                           best_measure_name: str, title: str, metric: str) -> None:
    markersize = 7
    best_idx = control_names.index(best_measure_name)
    candidate_control_linecolor = ['deepskyblue', 'dodgerblue', 'skyblue', 'steelblue', 
                                   'cornflowerblue', 'royalblue', 'mediumblue', 'slateblue', 'darkblue']
    for i in range(len(control_names)):
        if i == best_idx:
            color = "red"
            alpha = 1.0
        else:
            color = candidate_control_linecolor[i]
            alpha = 0.3
        plot_smooth_curve(ax, k_sizes, control_performance[i], complete_distance_name(control_names[i]), color, alpha)
    
    ax.set_title(title + ": " + metric)
    ax.set_xlabel('k')
    ax.set_ylabel(metric)
    ax.legend(loc = 'lower right')

def complete_distance_name(name: str) -> str:
    if name == "DTW":
        return "DTW-AROW"
    elif name == "Euc":
        return "Euclidean"
    elif name == "Cos":
        return "Cosine"
    elif name == "Manh":
        return "Manhattan"
    
def plot_final_performance_metrics(ax, k_sizes, metric, base_glob_Euc_performance: dict, base_glob_Cos_performance: dict, base_fix_distance_performance_nw: dict, base_opt_distance_performance_nw: dict,
                                   final_model_performance_nw: dict, final_model_performance_wt: dict, model_name: str) -> None:
    # derive p values
    final_nw_p_val = one_tailed_t_test(final_model_performance_nw[metric], final_model_performance_wt[metric])
    base_opt_distance_nw_p_val = one_tailed_t_test(base_opt_distance_performance_nw[metric], final_model_performance_wt[metric])
    base_fix_distance_nw_p_val = one_tailed_t_test(base_fix_distance_performance_nw[metric], final_model_performance_wt[metric])
    base_glob_Cos_p_val = one_tailed_t_test(base_glob_Cos_performance[metric], final_model_performance_wt[metric])
    base_glob_Euc_p_val = one_tailed_t_test(base_glob_Euc_performance[metric], final_model_performance_wt[metric])
    
    
    plot_smooth_curve(ax, k_sizes, final_model_performance_wt[metric], label = "%s (D+W+O)"%(model_name), color = "red")
    plot_smooth_curve(ax, k_sizes, final_model_performance_nw[metric], label = f"%s (D+W, $\\mathit{{{final_nw_p_val}}}$)"%(model_name), color = "blue")
    plot_smooth_curve(ax, k_sizes, base_opt_distance_performance_nw[metric], label = f"%s (D, $\\mathit{{{base_opt_distance_nw_p_val}}}$)"%(model_name), color = "green")
    plot_smooth_curve(ax, k_sizes, base_fix_distance_performance_nw[metric], label = f"%s (base, $\\mathit{{{base_fix_distance_nw_p_val}}}$)"%(model_name), color = "cyan")
    plot_smooth_curve(ax, k_sizes, base_glob_Cos_performance[metric], label = f"%s (cosine, $\\mathit{{{base_glob_Cos_p_val}}}$)"%(model_name), color = "pink")
    plot_smooth_curve(ax, k_sizes, base_glob_Euc_performance[metric], label = f"%s (Euclidean, $\\mathit{{{base_glob_Euc_p_val}}}$)"%(model_name), color = "purple")
    
    ax.set_title("All Features-%s: %s"%(model_name, metric))
    ax.set_ylabel(metric)
    ax.set_xlabel('k')
    ax.legend(loc = 'lower right')
    

    
