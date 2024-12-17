import numpy as np

def plot_metric_along_k(ax, k_sizes, performance_nw, performance_wt, title, metric):
    plot_smooth_curve(ax, k_sizes, performance_nw, label ="No weighting", color = "blue")
    plot_smooth_curve(ax, k_sizes, performance_wt, label ="Data Overlap Rates weighting", color = "red")
    ax.legend()
    ax.set_xlabel("k")
    ax.set_ylabel(metric)
    ax.set_title(title)
    
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
        plot_smooth_curve(ax, k_sizes, control_performance[i], "Control - %s"%(control_names[i]), color, alpha)
    
    ax.set_title(title + ": " + metric)
    ax.set_xlabel('k')
    ax.set_ylabel(metric)
    ax.legend()
    
def plot_final_performance_metrics(ax, k_sizes, metric, base_glob_Euc_performance: dict, base_glob_Cos_performance: dict, base_fix_distance_performance_nw: dict, base_opt_distance_performance_nw: dict,
                                   final_model_performance_nw: dict, final_model_performance_wt: dict, model_name: str) -> None:
    plot_smooth_curve(ax, k_sizes, final_model_performance_wt[metric], label = "%s (D+W+O)"%(model_name), color = "red")
    plot_smooth_curve(ax, k_sizes, final_model_performance_nw[metric], label = "%s (D+W)"%(model_name), color = "blue")
    plot_smooth_curve(ax, k_sizes, base_opt_distance_performance_nw[metric], label = "%s (D)"%(model_name), color = "green")
    plot_smooth_curve(ax, k_sizes, base_fix_distance_performance_nw[metric], label = "%s (base)"%(model_name), color = "cyan")
    plot_smooth_curve(ax, k_sizes, base_glob_Cos_performance[metric], label = "%s (cosine)"%(model_name), color = "pink")
    plot_smooth_curve(ax, k_sizes, base_glob_Euc_performance[metric], label = "%s (Euclidean)"%(model_name), color = "purple")
    
    ax.set_title("All Features: %s"%(metric))
    ax.set_ylabel(metric)
    ax.set_xlabel('k')
    ax.legend()
    
    
