import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
import dask.array as da

def translate_dist_mtx_to_simi(dist_arr):
    dist_arr_norm = min_max_normalization(dist_arr, axis = 1)
    simi_arr = 1 - dist_arr_norm
    return simi_arr

def min_max_normalization(data, axis = 1):
    scaler = MinMaxScaler()
    if axis == 1:
        data = data.T
    scaled_data = scaler.fit_transform(data)
    if axis == 1:
        scaled_data = scaled_data.T
    return scaled_data

def fast_argsort(arr, num_processors):
    #important here, the second shape must not change because we are sort an entire row
    darr = da.from_array(arr, chunks=(arr.shape[0]//num_processors, arr.shape[1]))
    result = darr.map_blocks(custom_argsort, dtype=np.int64)
    computed_result = result.compute()
    return computed_result

def slow_argsort(arr):
    sorted_indices = np.argsort(arr, axis=1)[:, ::-1]
    return sorted_indices

def custom_argsort(block):
    #Descending order
    sorted_indices = np.argsort(block, axis=1)[:, ::-1]
    return sorted_indices

def f1_score(precision, recall):
    return (2 * precision * recall) / (precision + recall)

def confidence_interval(y_test, y_pred_proba, n_bootstrap=1000, alpha=0.95):
    assert len(y_test) == len(y_pred_proba)
    rng = np.random.default_rng()  # Random number generator
    AUPRCs = []
    AUROCs = []

    for _ in (range(n_bootstrap)):
        # Resample with replacement
        indices = rng.choice(len(y_test), size=len(y_test), replace=True)
        y_test_resampled = np.array(y_test)[indices]
        y_pred_proba_resampled = np.array(y_pred_proba)[indices]
        
        # Compute AUPRC and AUROC
        AUPRC = average_precision_score(y_test_resampled, y_pred_proba_resampled)
        AUROC = roc_auc_score(y_test_resampled, y_pred_proba_resampled)
        AUPRCs.append(AUPRC)
        AUROCs.append(AUROC)
    
    # Calculate the mean and CI
    mean_AUPRC = np.mean(AUPRCs)
    lower_AUPRC = np.percentile(AUPRCs, (1 - alpha) / 2 * 100)
    upper_AUPRC = np.percentile(AUPRCs, (1 + alpha) / 2 * 100)
    
    mean_AUROC = np.mean(AUROCs)
    lower_AUROC = np.percentile(AUROCs, (1 - alpha) / 2 * 100)
    upper_AUROC = np.percentile(AUROCs, (1 + alpha) / 2 * 100)
    
    return mean_AUPRC, lower_AUPRC, upper_AUPRC, mean_AUROC, lower_AUROC, upper_AUROC