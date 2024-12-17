import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
import sys
sys.path.append('/home/lideyi/AKI_SMART/SMART/utils')
from Z_Helping_Functions import min_max_normalization, fast_argsort

def process_idx_arr_for_test(train_idx: list, test_idx: list, idx_arr_full: np.array, y_full: np.array) -> tuple[np.array, np.array]:
    assert len(y_full) == len(train_idx) + len(test_idx)
    assert len(idx_arr_full) == len(y_full)
    
    idx_arr_test_clean = remove_train_rows_and_test_cols(idx_arr_full, train_idx, test_idx)
    y_test_arr = sort_by_idx_arr(idx_arr_test_clean, y_full)
    return idx_arr_test_clean, y_test_arr

def remove_train_rows_and_test_cols(idx_arr_full: np.array , train_indices: list, test_indices: list) -> np.array:
    
    # drop the train rows
    idx_arr_test = idx_arr_full[test_indices, :]
    
    # remove the test index from rows
    idx_to_remove = np.array(test_indices)

    # Use a list to collect the filtered rows
    idx_arr_test_clean = []

    # Iterate over each row in the matrix
    for row in idx_arr_test:
        # Use boolean indexing to filter out the elements in elements_to_remove
        filtered_row = row[~np.isin(row, idx_to_remove)]
        idx_arr_test_clean.append(filtered_row)

    # Convert the list of filtered rows back to a NumPy array or a list of lists
    idx_arr_test_clean = np.array(idx_arr_test_clean)

    
    assert(idx_arr_test_clean.shape[0] == len(test_indices))
    assert(idx_arr_test_clean.shape[1] == len(train_indices))
    
    return idx_arr_test_clean

def sort_by_idx_arr(idx_arr: np.array, y_train: np.array) -> np.array:
    y_train_arr = np.tile(y_train, (idx_arr.shape[0], 1))
    # Use advanced indexing to select the elements from matrix_b
    row_indices = np.arange(idx_arr.shape[0])[:, None]  # Create an array of row indices
    y_train_arr = y_train_arr[row_indices, idx_arr]  # Use the row indices and matrix_a for advanced indexing
    
    return y_train_arr


# here we test grid-searched measures against all other base measures under 4 conditions
def evluate_on_test_set(SCR_train: pd.DataFrame, SCR_test: pd.DataFrame, LAB_train: pd.DataFrame, LAB_test: pd.DataFrame, SCR_control_measure: str, LAB_control_measure: str, y_test: np.array, k_sizes: list, SCR_idx_y_nw_dict_test: dict, SCR_idx_y_wt_dict_test: dict, LAB_idx_y_nw_dict_test: dict, LAB_idx_y_wt_dict_test: dict, base_model: str) -> tuple[dict, dict]:
    # fill in testing base measure
    control_measures = {"SCR NW": SCR_control_measure, "LAB NW": LAB_control_measure, "SCR WT": SCR_control_measure, "LAB WT": LAB_control_measure}
            
    SCR_control_performance = dict()
    LAB_control_performance = dict()
            
    # test under 4 conditions
    SCR_nw_control, LAB_nw_control = perform_evluation_on_test_set(SCR_train, SCR_test, LAB_train, LAB_test, SCR_idx_y_nw_dict_test, LAB_idx_y_nw_dict_test, 
                                                                   y_test, k_sizes, control_measures, weighting = False, base_model = base_model)
    SCR_wt_control, LAB_wt_control = perform_evluation_on_test_set(SCR_train, SCR_test, LAB_train, LAB_test, SCR_idx_y_wt_dict_test, LAB_idx_y_wt_dict_test,
                                                                   y_test, k_sizes, control_measures, weighting = True, base_model = base_model)
    
    SCR_control_performance["NW"] = SCR_nw_control
    SCR_control_performance["WT"] = SCR_wt_control
    
    LAB_control_performance["NW"] = LAB_nw_control
    LAB_control_performance["WT"] = LAB_wt_control
    
    return SCR_control_performance, LAB_control_performance



# here since query_grid_search_table give us 2 best method (SCR + LAB) at the same time,
# we test SCR and LAB under one condition at the same time
def perform_evluation_on_test_set(SCR_train: pd.DataFrame, SCR_test: pd.DataFrame, LAB_train: pd.DataFrame, LAB_test: pd.DataFrame,
                                  SCR_idx_y_dict_test: dict, LAB_idx_y_dict_test: dict, 
                                  y_test: np.array, k_sizes: list, best_distance_measures: dict, 
                                  weighting: bool, base_model: str) -> tuple[dict, dict]:
    
    SCR_performance = {"AUPRC": [], "AUROC": []}
    LAB_performance = {"AUPRC": [], "AUROC": []}
    
    best_SCR_measure, best_LAB_measure = get_best_method(best_distance_measures, weighting)
    SCR_arr_dict, LAB_arr_dict = SCR_idx_y_dict_test[best_SCR_measure], LAB_idx_y_dict_test[best_LAB_measure]
    
    for k in tqdm(k_sizes):
        if base_model == "KNN":
            SCR_AUPRC, SCR_AUROC = KNN(SCR_arr_dict, k, y_test)
            LAB_AUPRC, LAB_AUROC = KNN(LAB_arr_dict, k, y_test)
        elif base_model == "LR":
            SCR_AUPRC, SCR_AUROC = predict_by_LR(SCR_train, SCR_test, SCR_arr_dict, k, y_test)
            LAB_AUPRC, LAB_AUROC = predict_by_LR(LAB_train, LAB_test, LAB_arr_dict, k, y_test)
            
        SCR_performance["AUPRC"].append(SCR_AUPRC)
        SCR_performance["AUROC"].append(SCR_AUROC)
        LAB_performance["AUPRC"].append(LAB_AUPRC)
        LAB_performance["AUROC"].append(LAB_AUROC)
        
    return SCR_performance, LAB_performance


def get_best_method(best_distance_measures: dict, weighting: bool) -> tuple[str, str]:
    if not weighting:
        return best_distance_measures["SCR NW"], best_distance_measures["LAB NW"]
    else:
        return best_distance_measures["SCR WT"], best_distance_measures["LAB WT"]




def predict_by_LR(df_train: pd.DataFrame, df_test: pd.DataFrame, arr_dict: dict, k: int, y_test: np.array, 
                  report_pred: bool = False, report_CI: bool = False) -> tuple[float, float]:
    assert len(y_test) == arr_dict["idx"].shape[0]
    assert len(y_test) == arr_dict["label"].shape[0]
    y_pred_probs = []
    
    for i in range(len(y_test)):
        k_idx = arr_dict["idx"][i, :k]
        k_featrues = df_train.iloc[k_idx, :].values
        k_labels = arr_dict["label"][i, :k]
        featrues_test = df_test.iloc[i, :].values.reshape(1, -1)
        
        y_prob = LR(k_featrues, k_labels, featrues_test)
        y_pred_probs.append(y_prob)
    
    assert(not np.isnan(np.array(y_pred_probs)).any())

    if not report_pred:
        if not report_CI:
            AUPRC = average_precision_score(y_test, y_pred_probs)
            AUROC = roc_auc_score(y_test, y_pred_probs)
            return AUPRC, AUROC
        else:
            performance_CI = compute_CI(y_test, y_pred_probs)
            return performance_CI
    else:
        return y_pred_probs


def LR(k_featrues: np.array, k_labels: np.array, featrues_test: np.array) -> float:
    # if all labels are the same, we do not need to fit a LR model
    if len(np.unique(k_labels)) == 1:
        y_prob = k_labels[0]
    else:
        LR = LogisticRegression(random_state=0, max_iter=2000)
        LR.fit(k_featrues, k_labels)
        y_prob = LR.predict_proba(featrues_test)[0][1]
    return y_prob

#return AUROC and AUPRC at a certain size k
def KNN(arr_dict: dict, k: int, y_test: np.array, report_pred: bool = False, report_CI: bool = False) -> tuple[float, float]:
    y_train_arr = arr_dict["label"]
    # y_train_arr is a array of shape (len(y_test), len(full_data)), len(full_data) depends on wether it is one-vs-all training or testing
    # if it is one-vs-all training, len(full_data) = len(train_data) - 1, otherwise, if testing, len(full_data) = len(train_data)
    assert(len(y_test) == y_train_arr.shape[0])
    y_pred_probs = []
    
    for i in range(len(y_test)):
        k_labels = y_train_arr[i, :k]
        y_pred_prob = np.sum(k_labels) / len(k_labels)
        y_pred_probs.append(y_pred_prob)
        
    if not report_pred:
        if not report_CI:
            AUPRC = average_precision_score(y_test, y_pred_probs)
            AUROC = roc_auc_score(y_test, y_pred_probs)
            return AUPRC, AUROC
        else:
            performance_CI = compute_CI(y_test, y_pred_probs)
            return performance_CI
    else:
        return y_pred_probs


def test_final_personalized_model(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                  k_sizes: list, grid_search_table: pd.DataFrame, train_idx: list, 
                                  test_idx: list, y_full: np.array, y_test: np.array, 
                                  opt_measure_simi_full_dict: dict, num_processors: int, base_model, 
                                  weighting: bool = False, report_pred: bool = False, report_CI: bool = False) -> dict:
    # get the corresponding simi mtx
    SCR_simi_full, LAB_simi_full = opt_measure_simi_full_dict["SCR"], opt_measure_simi_full_dict["LAB"]
    
    if not report_pred:
        results = {"AUPRC": [], "AUROC": []}
        for k in tqdm(k_sizes):

            best_weights = eval(get_best_weights(grid_search_table, k, weighting))

            A = best_weights[0]
            B = best_weights[1]

            combined_weights_dict = combine_best_weights_for_test(SCR_simi_full, LAB_simi_full, A, B, train_idx, test_idx, y_full, num_processors)
            
            if report_CI:
                if base_model == "KNN":
                    performance_CI = KNN(combined_weights_dict, k, y_test, report_CI = report_CI)
                elif base_model == "LR":
                    performance_CI = predict_by_LR(X_train, X_test, combined_weights_dict, k, y_test, report_CI = report_CI)


                results["AUPRC"].append((performance_CI["AUPRC_mean"], performance_CI["AUPRC_low"], performance_CI["AUPRC_up"]))
                results["AUROC"].append((performance_CI["AUROC_mean"], performance_CI["AUROC_low"], performance_CI["AUROC_up"]))
            else:
                if base_model == "KNN":
                    AUPRC, AUROC = KNN(combined_weights_dict, k, y_test)
                elif base_model == "LR":
                    AUPRC, AUROC = predict_by_LR(X_train, X_test, combined_weights_dict, k, y_test)
                results["AUPRC"].append(AUPRC)
                results["AUROC"].append(AUROC)

        return results
    
    # for the subgroup performance analysis (at a single k)
    else:
        assert len(k_sizes) == 1
        k = k_sizes[0]
        best_weights = eval(get_best_weights(grid_search_table, k, weighting))
        A = best_weights[0]
        B = best_weights[1]
        combined_weights_dict = combine_best_weights_for_test(SCR_simi_full, LAB_simi_full, A, B, train_idx, test_idx, y_full, num_processors)
        if base_model == "KNN":
            y_pred_probs = KNN(combined_weights_dict, k, y_test, report_pred = True)
        elif base_model == "LR":
            y_pred_probs = predict_by_LR(X_train, X_test, combined_weights_dict, k, y_test, report_pred = True)
        assert len(y_pred_probs) == len(y_test)
        return y_pred_probs 

def combine_best_weights_for_test(SCR_simi_full: np.array, LAB_simi_full: np.array, A: float, B: float, 
                                  train_idx: list, test_idx: list, y_full: np.array, num_processors: int) -> dict:   # weighted sum of the simi mtxs and min-max normalization 
    combined_simi_full = A * SCR_simi_full + B * LAB_simi_full
    combined_simi_full = min_max_normalization(combined_simi_full, axis = 1)
    # get the ordered idx
    combined_idx_full = fast_argsort(combined_simi_full, num_processors)
    # organize the sorted idx into dict
    idx_arr_test_combined, y_test_arr_combined = process_idx_arr_for_test(train_idx, test_idx, combined_idx_full, y_full)
    idx_y_nw_dict_test_combined = {"idx": idx_arr_test_combined, "label": y_test_arr_combined}
    return idx_y_nw_dict_test_combined



def get_best_weights(grid_search_table: np.array, k: int, weighting: bool) -> str:
    if not weighting:
        return grid_search_table.loc[k, "COMBINE NW"]
    else:
        return grid_search_table.loc[k, "COMBINE WT"]
    

def compute_CI(y_true: np.array, y_pred_proba: np.array) -> dict:
    np.random.seed(888)
    AUPRC_scores = []
    AUROC_scores = []
    
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    n_bootstrap = 2000
    for i in range(n_bootstrap):
        # Stratified sampling: maintain label proportion
        y_true_resampled, y_pred_resampled = resample_stratified(y_true, y_pred_proba)
        
        # Calculate AUPRC
        AUPRC = average_precision_score(y_true_resampled, y_pred_resampled)
        AUPRC_scores.append(AUPRC)
        
        # Calculate AUROC
        AUROC = roc_auc_score(y_true_resampled, y_pred_resampled)
        AUROC_scores.append(AUROC)
    
    # Calculate mean and confidence intervals
    ci_low = 2.5
    ci_up = 97.5
    
    AUPRC_mean = np.mean(AUPRC_scores)
    AUPRC_low = np.percentile(AUPRC_scores, ci_low)
    AUPRC_up = np.percentile(AUPRC_scores, ci_up)
    
    AUROC_mean = np.mean(AUROC_scores)
    AUROC_low = np.percentile(AUROC_scores, ci_low)
    AUROC_up = np.percentile(AUROC_scores, ci_up)
    
    return {
        "AUPRC_mean": AUPRC_mean,
        "AUPRC_low": AUPRC_low,
        "AUPRC_up": AUPRC_up,
        "AUROC_mean": AUROC_mean,
        "AUROC_low": AUROC_low,
        "AUROC_up": AUROC_up
    }
    
def resample_stratified(y_true: np.array, y_pred_proba: np.array) -> tuple[np.array, np.array]:
    # Separate indices for positive and negative classes
    pos_indices = np.where(y_true == 1)[0]
    neg_indices = np.where(y_true == 0)[0]
    
    # Resample while maintaining proportions
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)
    
    pos_resampled = np.random.choice(pos_indices, n_pos, replace=True)
    neg_resampled = np.random.choice(neg_indices, n_neg, replace=True)
    
    # Combine resampled indices
    resampled_indices = np.concatenate([pos_resampled, neg_resampled])
    np.random.shuffle(resampled_indices)
    
    y_true_resampled = y_true[resampled_indices]
    y_pred_resampled = y_pred_proba[resampled_indices]
    
    return y_true_resampled, y_pred_resampled
