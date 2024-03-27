import csv
from typing import List
import math
from sentence_transformers import SentenceTransformer
import numpy as np
import random
from numba import jit
import logging
import sys
from scipy import stats
from scipy.stats import norm, t
import pandas as pd
import json

csv.field_size_limit(sys.maxsize)

def read_csv(path):
    """Read a CSV file."""
    with open(path) as f:
        reader = csv.reader(f)
        return list(reader)
    
def divide_sample_rate(rate: float, table_sizes: List[int]):
    table_rate = math.pow(rate, 1 / len(table_sizes))
    return [int(table_rate * size) for size in table_sizes]

def get_sentence_transformer(model_name: str="all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)


"""
avoid select duplicate in self join
avoid scores out of the range of [0,1]
"""
def preprocess(x: np.ndarray, is_self_join: bool=False):
    # avoid select duplicate
    if is_self_join:
        assert len(x.shape) == 2 and x.shape[0] == x.shape[1]
        # set diagonal to min
        x[np.diag_indices(x.shape[0])] = 0

    if np.min(x) < 0:
        x -= np.min(x)
    if np.max(x) > 1:
        x /= np.max(x)

    return x

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

@jit(nopython=True)
def calculate_score_for_tuples(embeddings: np.ndarray) -> np.ndarray:
    scores = np.zeros(len(embeddings))
    for i in range(len(embeddings)):
        norm1 = np.linalg.norm(embeddings[i][0]).item()
        norm2 = np.linalg.norm(embeddings[i][1]).item()
        scores[i] = np.dot(embeddings[i][0], embeddings[i][1]) / (norm1 * norm2)
    # normalize scores from -1, 1 to 0, 1
    scores += 1
    scores /= 2
    return scores

@jit(nopython=True)
def calculate_score_for_tables(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    scores = np.ones((len(embeddings1), len(embeddings2)))
    for i in range(len(embeddings1)):
        for j in range(len(embeddings2)):
            norm1 = np.linalg.norm(embeddings1[i]).item()
            norm2 = np.linalg.norm(embeddings2[j]).item()
            scores[i][j] = np.dot(embeddings1[i], embeddings2[j]) / (norm1 * norm2)
    # normalize scores from -1, 1 to 0, 1
    scores += 1
    scores /= 2
    return scores

def set_up_logging(log_file: str, log_level: str="INFO"):
    import sys, os
    log_level = str.lower(log_level)
    if log_level == "info":
        level = logging.INFO
    elif log_level == "debug":
        level = logging.DEBUG
    else:
        raise NotImplementedError(f"Log level {log_level} not implemented.")

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filename=log_file,
        filemode="w",
        force=True
    )


def get_ci_gaussian(data, confidence_level=0.95):
    """Get the confidence interval of the data using Gaussian."""
    mean = np.average(data)
    std = np.std(data)
    n = len(data)
    z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    return mean - z * std / np.sqrt(n), mean + z * std / np.sqrt(n)

def get_ci_ttest(data, confidence_level=0.95):
    """Get the confidence interval of the data using ttest."""
    mean = np.average(data)
    std = np.std(data)
    n = len(data)
    t = stats.ttest_1samp(data, popmean=mean)
    confidence_interval = t.confidence_interval(confidence_level=confidence_level)
    return confidence_interval.low, confidence_interval.high

def get_ci_bootstrap(trial_results, estimation, ts, variance, confidence_levels: list=[0.95]):
    """Get the confidence interval of the data using bootstrap."""
    p_lbs, p_ubs = [], []
    e_lbs, e_ubs = [], []
    t_lbs, t_ubs = [], []
    for confidence_interval in confidence_levels:
        ts_lb = np.percentile(ts, (1 - confidence_interval) / 2 * 100)
        ts_ub = np.percentile(ts, (1 + confidence_interval) / 2 * 100)
        
        p_lb = np.percentile(trial_results, (1 - confidence_interval) / 2 * 100)
        p_ub = np.percentile(trial_results, (1 + confidence_interval) / 2 * 100)
        e_lb = estimation*2 - p_ub
        e_ub = estimation*2 - p_lb
        t_lb = estimation - ts_ub * np.sqrt(variance)
        t_ub = estimation - ts_lb * np.sqrt(variance)

        p_lbs.append(p_lb)
        p_ubs.append(p_ub)
        e_lbs.append(e_lb)
        e_ubs.append(e_ub)
        t_lbs.append(t_lb)
        t_ubs.append(t_ub)
    return p_lbs, p_ubs, e_lbs, e_ubs, t_lbs, t_ubs

def get_ci_bootstrap_ttest(estimation, ts, variance, confidence_level: float=0.95):
    """Get the confidence interval of the data using bootstrap-t."""
    ts_lb = np.percentile(ts, (1 - confidence_level) / 2 * 100)
    ts_ub = np.percentile(ts, (1 + confidence_level) / 2 * 100)
    
    t_lb = estimation - ts_ub * np.sqrt(variance)
    t_ub = estimation - ts_lb * np.sqrt(variance)

    return t_lb, t_ub

def get_ci_bootstrap_quantile(trial_results, confidence_level: float=0.95):
    """Get the confidence interval of the data using bootstrap."""
    p_lb = np.percentile(trial_results, (1 - confidence_level) / 2 * 100)
    p_ub = np.percentile(trial_results, (1 + confidence_level) / 2 * 100)
    return p_lb, p_ub

def get_ci_wilson_score_interval(data, confidence_level=0.95):
    """Get the confidence interval of the data using Wilson score interval."""
    mean = np.average(data)
    z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    center = 1 / (1+z**2/len(data)) * (mean + z**2/(2*len(data)))
    width = z / (1+z**2/len(data)) * np.sqrt(mean*(1-mean)/len(data) + z**2/(4*len(data)**2))
    return center - width, center + width

def normalize(array: np.ndarray, style="proportional"):
    if style == "proportional":
        array /= np.sum(array)
    elif style == "sqrt":
        array = np.sqrt(array)
        array /= np.sum(array)
    else:
        raise NotImplementedError(f"Style {style} not implemented.")
    return array

def defensive_mix(weights, ratio: float, mixture: str="random"):
    if mixture == "random":
        weights = (1-ratio) * weights + ratio * 1/len(weights)
    else:
        raise NotImplementedError(f"Defensive mixture {mixture} not implemented.")
    return weights

def weighted_sample_pd(weights: np.ndarray, size: int, replace: bool=False):
    """Sample from a weighted array using pandas."""
    return pd.Series(weights).sample(n=size, replace=replace, weights=weights).index

def get_non_positive_ci(max_statistics: float, 
                        confidence_level: float, 
                        n_positive_population_size: int, 
                        n_positive_sample_size: int):
    z = float(stats.norm.ppf(1 - (1 - confidence_level) / 2))
    pr_upper_bound = 1 / (1+z**2/n_positive_sample_size) * (z**2/n_positive_sample_size)
    n_positive_count_lower_bound = 0
    n_positive_count_upper_bound = pr_upper_bound * n_positive_population_size
    n_positive_sum_lower_bound = 0
    n_positive_sum_upper_bound = n_positive_count_upper_bound * max_statistics
    return n_positive_count_lower_bound, n_positive_count_upper_bound, n_positive_sum_lower_bound, n_positive_sum_upper_bound

def get_cutoff_score(dataset: str):
    with open("block_noci/valid_threshold.json") as f:
        thresholds = json.load(f)
    return thresholds[dataset]
    