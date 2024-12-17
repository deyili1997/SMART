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


def extract_CI_bounds(performance: list) -> tuple:
    mean = [perf[0] for perf in performance]
    low = [perf[1] for perf in performance]
    up = [perf[2] for perf in performance]
    return np.array(mean), np.array(low), np.array(up)

def plot_smooth_curve_with_CI(ax, X_vals, y_vals_CI, label, color, alpha=1.0):
    # Extract the mean, low, and up values
    y_mean, y_low, y_up = extract_CI_bounds(y_vals_CI)
    # Smooth the y_mean, y_low, and y_up curves
    coeffs_mean = np.polyfit(X_vals, y_mean, 5)  # Fit polynomial for y_mean
    coeffs_low = np.polyfit(X_vals, y_low, 5)    # Fit polynomial for y_low
    coeffs_up = np.polyfit(X_vals, y_up, 5)      # Fit polynomial for y_up

    poly_mean = np.poly1d(coeffs_mean)
    poly_low = np.poly1d(coeffs_low)
    poly_up = np.poly1d(coeffs_up)
    
    # Generate smooth x-values
    X_smooth = np.linspace(np.min(X_vals), np.max(X_vals), 500)
    y_mean_smooth = poly_mean(X_smooth)
    y_low_smooth = poly_low(X_smooth)
    y_up_smooth = poly_up(X_smooth)
    
    # Plot the mean curve
    ax.plot(X_smooth, y_mean_smooth, label=label, color=color, linewidth=1.5, alpha=alpha)
    
    # Plot the shaded 95% CI band
    ax.fill_between(X_smooth, y_low_smooth, y_up_smooth, color=color, alpha=0.2, label="_nolegend_")
    
    # Plot the original data points for y_mean (optional)
    ax.scatter(X_vals, y_mean, color=color, marker='.', s=30, alpha=alpha, label="_nolegend_")

    
    
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
    plot_smooth_curve_with_CI(ax, k_sizes, final_model_performance_wt[metric], label = "%s (D+W+O)"%(model_name), color = "red")
    plot_smooth_curve_with_CI(ax, k_sizes, final_model_performance_nw[metric], label = "%s (D+W)"%(model_name), color = "blue")
    plot_smooth_curve_with_CI(ax, k_sizes, base_opt_distance_performance_nw[metric], label = "%s (D)"%(model_name), color = "green")
    plot_smooth_curve_with_CI(ax, k_sizes, base_fix_distance_performance_nw[metric], label = "%s (base)"%(model_name), color = "cyan")
    plot_smooth_curve_with_CI(ax, k_sizes, base_glob_Cos_performance[metric], label = "%s (cosine)"%(model_name), color = "pink")
    plot_smooth_curve_with_CI(ax, k_sizes, base_glob_Euc_performance[metric], label = "%s (Euclidean)"%(model_name), color = "purple")
    
    ax.set_title("All Features: %s"%(metric))
    ax.set_ylabel(metric)
    ax.set_xlabel('k')
    ax.legend()
    
    
