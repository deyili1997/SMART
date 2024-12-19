#parallel computing
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

def parallel_overlap_matrix_comp(df, num_processes, metric, norm_distr):
    pool = Pool(num_processes)
    total = len(df)

    results = list(tqdm(pool.imap(calculate_overlap, 
                                  [(i, df, metric, norm_distr) for i in range(total - 1)]), total=total - 1))
    pool.close()
    pool.join()
    return create_similarity_matrix(results)

def check_vec_overlap(u, v):
    assert(len(u) == len(v))
    u_arr = np.array(u)
    v_arr = np.array(v)
    overlap = np.logical_and(u_arr, v_arr)
    return overlap

def calculate_overlap_rate_SCR(u, v, norm_distr):
    #get the bool vec
    overlap_vec = check_vec_overlap(u, v)
    overlap_rate = overlap_rate_SCR(overlap_vec, norm_distr)
    return overlap_rate

def calculate_overlap_rate_LAB(u, v, norm_distr):
    #get the bool vec
    overlap_vec = check_vec_overlap(u, v)
    overlap_rate = overlap_rate_LAB(overlap_vec, norm_distr)
    return overlap_rate

def calculate_overlap(args):
    index, df, metric, norm_distr = args
    return [metric(df.iloc[index], df.iloc[j], norm_distr) for j in range(index + 1, len(df))]

#apply a Gaussian distribution on SCr overlap vec
def overlap_rate_SCR(overlap_vec, norm_distr):
    return np.sum(norm_distr * overlap_vec)

def overlap_rate_LAB(overlap_vec, norm_distr):
    return np.sum(norm_distr * overlap_vec)


def create_similarity_matrix(distance_list):
    n = len(distance_list[0]) + 1

    matrix = np.ones((n, n))

    for i in range(n-1):
        matrix[i, i+1:i+1+len(distance_list[i])] = distance_list[i]

    for i in range(n):
        for j in range(i+1, n):
            matrix[j, i] = matrix[i, j]
    return matrix

def check_matrix_sanity(matrix):
    assert(matrix.shape[0] == matrix.shape[1])
    assert(np.all((np.round(matrix, 3) >= 0) & (np.round(matrix, 3) <= 1)))
    