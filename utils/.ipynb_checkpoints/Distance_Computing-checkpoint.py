import sys
sys.path.append('/home/lideyi/AKI_SMART/SMART/DTW_with_missing_values')
import dtw_missing.dtw_missing as dtw_m
from utils.Z_Helping_Functions import translate_dist_mtx_to_simi, fast_argsort
from sklearn.metrics.pairwise import pairwise_distances
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import numpy as np

def parallel_distance_matrix(df: pd.DataFrame, num_processes: int, metric: callable) -> np.ndarray:
    pool = Pool(num_processes)
    total = len(df)

    results = list(tqdm(pool.imap(calculate_overlap, [(i, df, metric) for i in range(total - 1)]), total=total - 1))
    pool.close()
    pool.join()
    distance_mtx = create_distance_matrix(results)
    return distance_mtx

def calculate_overlap(args: tuple) -> list:
    index, df, metric = args
    return [metric(df.iloc[index], df.iloc[j]) for j in range(index + 1, len(df))]

def create_distance_matrix(distance_list: list) -> np.array:
    n = len(distance_list[0]) + 1

    matrix = np.zeros((n, n))

    for i in range(n-1):
        matrix[i, i+1:i+1+len(distance_list[i])] = distance_list[i]

    for i in range(n):
        for j in range(i+1, n):
            matrix[j, i] = matrix[i, j]
    return matrix

def get_DTW_distance(u, v):
    u = np.array(u)
    v = np.array(v)
    d = dtw_m.warping_paths(u, v)[0]
    return d


def compute_similarity(feature_df: pd.DataFrame, metric: str, train_len: int, num_processors: int) -> \
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if metric in ['euclidean', 'cosine', 'manhattan']:
        # compute the distance matrix for the entire dataset
        dist_full_arr = pairwise_distances(feature_df, metric=metric, n_jobs=-1)
        simi_full_arr = translate_dist_mtx_to_simi(dist_full_arr)
        idx_full_arr = fast_argsort(simi_full_arr, num_processors)
        
        # we also need to compute the similarity matrix and idx rankings only for the train set
        dist_train_arr = dist_full_arr[:train_len, :train_len]
        simi_train_arr = translate_dist_mtx_to_simi(dist_train_arr)
        idx_train_arr = fast_argsort(simi_train_arr, num_processors)
        
    else:
        raise ValueError('Invalid metric')
    return simi_full_arr, idx_full_arr, simi_train_arr, idx_train_arr

def overlap_rates_weighting(overlap_arr, simi_arr, num_processors):
    weighted_simi_arr = overlap_arr * simi_arr
    weighted_idd_arr = fast_argsort(weighted_simi_arr, num_processors)
    return weighted_simi_arr, weighted_idd_arr