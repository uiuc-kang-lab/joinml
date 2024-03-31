from typing import List, Tuple
import numpy as np
import logging
from copy import deepcopy

def get_ci(z: np.ndarray, alpha: float) -> Tuple[float, float]:
    """
    Get confidence interval for the given z and alpha.
    """
    mean = np.mean(z)
    std = np.std(z)
    n = len(z)
    conf = np.sqrt(2 * np.log(1/alpha))
    return mean - conf * std / np.sqrt(n), mean + conf * std / np.sqrt(n)

def supg_recall_target_importance(target: float, oracle_results: np.ndarray, 
                                  sampling_weights: np.ndarray, dataset_size: int) -> Tuple[float, bool]:
    # unweighting sampling weights
    m = 1 / sampling_weights / dataset_size
    # estimate tau0 for the sample
    total_positive = np.sum(oracle_results)
    target_positive = np.ceil(target * total_positive)
    for i in range(len(m)-1, 0, -1):
        if np.sum(oracle_results[i:]) > target_positive:
            tau0 = sampling_weights[i]
            break
    # get estimated population
    z1 = deepcopy(m * oracle_results)
    z1[sampling_weights < tau0] = 0
    z2 = deepcopy(m * oracle_results)
    z2[sampling_weights >= tau0] = 0
    # get adjusted recall
    lb1, ub1 = get_ci(z1, 0.025)
    lb2, ub2 = get_ci(z2, 0.025)
    print(lb1, ub1, lb2, ub2)
    if ub1 / (ub1 + lb2) <= 0.99:
        gamma = ub1 / (ub1 + lb2)
        status = True
    else:
        gamma = 0.99
        status = False
    target_positive = np.ceil(gamma * total_positive)
    print(f"new target positives: {gamma}")
    assert gamma < 1
    for i in range(len(m)-1, 0, -1):
        if np.sum(oracle_results[i:]) > target_positive:
            return sampling_weights[i], status
    return min(sampling_weights), status

def supg_recall_target_uniform(target: float, oracle_results: np.ndarray, sampling_weights: np.ndarray, dataset_size: int) -> Tuple[float, bool]:
    target_positive = np.ceil(target * np.sum(oracle_results))
    for i in range(oracle_results.shape[0]-1, 0, -1):
        if np.sum(oracle_results[i:]) > target_positive:
            tau0 = sampling_weights[i]
            break
    z1 = deepcopy(oracle_results)
    z1[sampling_weights < tau0] = 0
    z2 = deepcopy(oracle_results)
    z2[sampling_weights >= tau0] = 0
    lb1, ub1 = get_ci(z1, 0.025)
    lb2, ub2 = get_ci(z2, 0.025)
    if ub1 / (ub1 + lb2) < 0.99:
        gamma = ub1 / (ub1 + lb2)
        status = True
    else:
        gamma = 0.99
        status = False
    logging.info(f"required new positive to bound recall {gamma}")
    target_positive = np.ceil(gamma * np.sum(oracle_results))
    for i in range(len(oracle_results)-1, 0, -1):
        if np.sum(oracle_results[i:]) >= target_positive:
            return sampling_weights[i], status
    return min(sampling_weights), status

def supg_precision_target_importance(target: float, oracle_results: np.ndarray, 
                                     sampling_weights: np.ndarray, dataset_size: int) -> Tuple[float, bool]:
    # unweight sampling weights
    m = 1 / sampling_weights / dataset_size
    candidates = []
    k = 10
    for i in range(len(m)-100*k, len(m)-k, k):
        tau = sampling_weights[i]
        curr_z = oracle_results[i:] * m[i:]
        lb, _ = get_ci(curr_z, 0.05/100)
        candidates.append((tau, lb))

    true_candidate = (100, 0)
    max_candidate = (100, 0)
    for candidate in candidates:
        tau, precision = candidate
        if precision > target and tau < true_candidate[0]:
            true_candidate = candidate
            print(f"true candidate {candidate}")
        if precision > max_candidate[1]:
            max_candidate = candidate

    if true_candidate[1] == 0:
        print(f"max candidate {max_candidate}")
        return max_candidate[0], False
    else:
        return true_candidate[0], True

def supg_precision_target_uniform(target: float, oracle_results: np.ndarray, sampling_weights: np.ndarray, dataset_size: int) -> Tuple[float, bool]:
    candidates = []
    m = 10
    for i in range(len(oracle_results)-100*m, len(oracle_results)-m, m):
        tau = sampling_weights[i]
        curr_z = oracle_results[i:]
        lb, _ = get_ci(curr_z, 0.05/100)
        candidates.append((tau, lb))

    true_candidate = (100, 0)
    max_candidate = (100, 0)
    for candidate in candidates:
        tau, precision = candidate
        if precision > target and tau < true_candidate[0]:
            true_candidate = candidate
            print(f"true candidate {true_candidate}")
        if precision > max_candidate[1]:
            max_candidate = candidate

    if true_candidate[1] == 0:
        print(f"max candidate {max_candidate}")
        return max(sampling_weights), False
    else:
        return true_candidate[0], True