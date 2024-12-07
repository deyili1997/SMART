import numpy as np
from sklearn.preprocessing import MinMaxScaler

import dask.array as da

def translate_dist_mtx_to_simi(dist_arr):
    dist_arr_norm = min_max_normalization(dist_arr)
    simi_arr = 1 - dist_arr_norm
    return simi_arr

def min_max_normalization(data):
    scaler = MinMaxScaler()
    data_transposed = data.T
    scaled_data_transposed = scaler.fit_transform(data_transposed)
    scaled_data = scaled_data_transposed.T
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