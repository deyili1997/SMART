import numpy as np

def plot_metric_along_k(ax, k_sizes, performance_nw, performance_wt, title, metric):
    plot_smooth_curve(ax, k_sizes, performance_nw, label ="No weighting", color = "blue")
    plot_smooth_curve(ax, k_sizes, performance_wt, label ="Data Overlap Rates weighting", color = "red")
    ax.legend()
    ax.set_xlabel("k")
    ax.set_ylabel(metric)
    ax.set_title(title)
    
def plot_smooth_curve(ax, X_vals, y_vals, label, color):
    # Fit a polynomial of the specified degree
    coeffs = np.polyfit(X_vals, y_vals, 5)
    poly_func = np.poly1d(coeffs)
    
    # Generate smooth x-values and compute corresponding y-values
    X_smooth = np.linspace(np.min(X_vals), np.max(X_vals), 500)
    y_smooth = poly_func(X_smooth)
    
    # Plot the fitted curve
    ax.plot(X_smooth, y_smooth, label=label, color=color, linewidth=1)
    
    # Plot the original points with markers
    ax.scatter(X_vals, y_vals, color=color, marker='.', s=30, label="_nolegend_")
    
    
def plot_final_performance_metrics(ax, k_sizes, metric, base_glob_Euc_performance: dict, base_glob_Cos_performance: dict, base_fix_distance_performance_nw: dict, base_opt_distance_performance_nw: dict,
                                   final_model_performance_nw: dict, final_model_performance_wt: dict, model_name: str) -> None:
    plot_smooth_curve(ax, k_sizes, final_model_performance_wt[metric], label = "%s (D+W+O)"%(model_name), color = "red")
    plot_smooth_curve(ax, k_sizes, final_model_performance_nw[metric], label = "%s (D+W)"%(model_name), color = "blue")
    plot_smooth_curve(ax, k_sizes, base_opt_distance_performance_nw[metric], label = "%s (D)"%(model_name), color = "green")
    plot_smooth_curve(ax, k_sizes, base_fix_distance_performance_nw[metric], label = "%s (base)"%(model_name), color = "cyan")
    plot_smooth_curve(ax, k_sizes, base_glob_Cos_performance[metric], label = "%s (Cosine)"%(model_name), color = "pink")
    plot_smooth_curve(ax, k_sizes, base_glob_Euc_performance[metric], label = "%s (Euclidean)"%(model_name), color = "purple")
    
    ax.set_title("All Features: %s"%(metric))
    ax.set_ylabel(metric)
    ax.set_xlabel('k')
    ax.legend()