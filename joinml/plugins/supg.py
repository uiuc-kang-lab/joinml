from typing import List, Tuple
import numpy as np

def get_ci(z: np.ndarray, alpha: float) -> Tuple[float, float]:
    """
    Get confidence interval for the given z and alpha.
    """
    mean = np.mean(z)
    std = np.std(z)
    n = len(z)
    conf = np.sqrt(2 * np.log(1/alpha))
    return mean - conf * std / np.sqrt(n), mean + conf * std / np.sqrt(n)

def supg_recall_target_importance(target: float, oracle_results: np.ndarray, sampling_weights: np.ndarray, dataset_size: int) -> float:
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
    z1 = m * oracle_results
    z1[sampling_weights < tau0] = 0
    z2 = m * oracle_results
    z2[sampling_weights >= tau0] = 0
    # get adjusted recall
    lb1, ub1 = get_ci(z1, 0.025)
    lb2, ub2 = get_ci(z2, 0.025)
    gamma = ub1 / (ub1 + ub2)
    target_positive = np.ceil(gamma * total_positive)
    assert gamma < 1
    for i in range(len(m)-1, 0, -1):
        if np.sum(m[i:]) > target_positive:
            tau = sampling_weights[i]
            break
    return tau

def supg_recall_target_uniform(target: float, oracle_results: np.ndarray, sampling_weights: np.ndarray, dataset_size: int) -> float:
    target_positive = np.ceil(target * np.sum(oracle_results))
    for i in range(oracle_results.shape[0]-1, 0, -1):
        if np.sum(oracle_results[i:]) > target_positive:
            tau0 = sampling_weights[i]
            break
    z1 = oracle_results
    z1[sampling_weights < tau0] = 0
    z2 = oracle_results
    z2[sampling_weights >= tau0] = 0
    lb1, ub1 = get_ci(z1, 0.025)
    lb2, ub2 = get_ci(z2, 0.025)
    gamma = ub1 / (ub1 + ub2)
    assert gamma < 1
    target_positive = np.ceil(gamma * np.sum(oracle_results))
    for i in range(len(oracle_results)-1, 0, -1):
        if np.sum(oracle_results[i:]) > target_positive:
            tau = sampling_weights[i]
            break
    return tau

def supg_precision_target_importance(target: float, oracle_results: np.ndarray, sampling_weights: np.ndarray, dataset_size: int) -> float:
    # unweight sampling weights
    m = 1 / sampling_weights / dataset_size
    candidates = []
    for i in range(0, len(m)-100, 100):
        tau = sampling_weights[i]
        curr_z = oracle_results[i:] * m[i:]
        lb, _ = get_ci(curr_z, 0.025)
        candidates.append((tau, lb))

    true_candidate = (100, 0)
    max_candidate = (100, 0)
    for candidate in candidates:
        tau, precision = candidate
        if precision > target and tau < true_candidate[0]:
            true_candidate = candidate
        if precision > max_candidate[1]:
            max_candidate = candidate

    if true_candidate[1] == 0:
        return max_candidate[0]
    else:
        return true_candidate[0]

def supg_precision_target_uniform(target: float, oracle_results: np.ndarray, sampling_weights: np.ndarray, dataset_size: int) -> float:
    candidates = []
    for i in range(0, len(m)-100, 100):
        tau = sampling_weights[i]
        curr_z = oracle_results[i:]
        lb, _ = get_ci(curr_z, 0.025)
        candidates.append((tau, lb))

    true_candidate = (100, 0)
    max_candidate = (100, 0)
    for candidate in candidates:
        tau, precision = candidate
        if precision > target and tau < true_candidate[0]:
            true_candidate = candidate
        if precision > max_candidate[1]:
            max_candidate = candidate

    if true_candidate[1] == 0:
        return max_candidate[0]
    else:
        return true_candidate[0]