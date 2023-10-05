from typing import List, Callable
import numpy as np
from scipy import stats
import csv
from tqdm import tqdm
import multiprocessing
import sys
import logging

def wander_join(table1: List[int], table2: List[int], condition_evaluator: Callable[[int, int], bool], 
                proxy_score: np.ndarray, lid2index: dict, rid2index: dict, sample_size: int, confidence_level: float=0.95):
    # sum over one axis of proxy score
    table1 = np.array(table1)
    table2 = np.array(table2)
    table1_weights = proxy_score.sum(axis=0)
    table1_weights = table1_weights / table1_weights.sum()
    aligned_table1_weights = [table1_weights[lid2index[lid]] for lid in table1]
    table1_samples_indexes = np.random.choice(len(table1), size=sample_size, replace=True, p=aligned_table1_weights)
    count_results = []
    for table1_sample_index in table1_samples_indexes:
        table2_weights = proxy_score[lid2index[table1[table1_sample_index]]]
        table2_weights /= table2_weights.sum()
        aligned_table2_weights = [table2_weights[rid2index[rid]] for rid in table2]
        table2_sample_index = np.random.choice(len(table2), size=1, replace=True, p=table2_weights)[0]
        table1_sample_weight = aligned_table1_weights[table1_sample_index]
        table2_sample_weight = aligned_table2_weights[table2_sample_index]
        sample_pair = [table1[table1_sample_index], table2[table2_sample_index]]
        condition = condition_evaluator(*sample_pair)
        if condition:
            count_results.append(1 / len(table1) / len(table2) / (table1_sample_weight * table2_sample_weight))
        else:
            count_results.append(0)
    count_results = np.array(count_results)
    count_mean = count_results.mean()
    ttest = stats.ttest_1samp(count_results, popmean=count_mean)
    ci = ttest.confidence_interval(confidence_level=confidence_level)
    ci_lower_bound = ci.low
    ci_upper_bound = ci.high

    return count_mean, ci_lower_bound, ci_upper_bound

def join(args) -> None:
    sample_ratio, ltable, rtable, labels, proxy_score, lid2index, rid2index, repeats, output_file, seed, confidence_level = args

    def condition_evaluator(id1, id2):
        if id1 == id2:
            return False
        
        pair1 = f"{id1}|{id2}"
        pair2 = f"{id2}|{id1}"
        if pair1 in labels or pair2 in labels:
            return True
        else:
            return False
    
    np.random.seed(seed)
    sample_size = int(len(ltable) * len(rtable) * sample_ratio)
    results, uppers, lowers = [], [], []
    for i in range(repeats):
        count_result, ci_upper, ci_lower  = \
            wander_join(ltable, rtable, condition_evaluator, proxy_score, lid2index, rid2index, sample_size, confidence_level)
        results.append(count_result)
        uppers.append(ci_upper)
        lowers.append(ci_lower)
        logging.info(f"sample ratio {sample_ratio} finishes {i+1}")
    
    logging.info(sample_ratio, np.average(results), np.average(uppers), np.average(lowers))

    with open(output_file, "a+") as f:
        writer = csv.writer(f)
        writer.writerow([sample_ratio, np.average(results), np.average(uppers), np.average(lowers)])

def to_matrix(proxy_score: List[List[int|str]], size1: int, size2: int):
    proxy_mat = np.ones(size1, size2) * 1e-2
    lid2index = {}
    rid2index = {}
    l_countid = 0
    r_countid = 0
    for row in proxy_score:
        id1, id2, score = row
        score = float(score)
        if id1 not in lid2index:
            lid2index[id1] = l_countid
            lindex = l_countid
            l_countid += 1
        else:
            lindex = lid2index[id1]
        if id2 not in rid2index:
            rid2index[id2] = r_countid
            rindex = r_countid
            r_countid += 1
        else:
            rindex = rid2index[id2]
        proxy_mat[lindex, rindex] = score
    return proxy_mat, lid2index, rid2index
            

def run_weighted_wander(sample_ratios: List[float], ltable: List[int|str], rtable: List[int|str], labels: set, proxy: List[List[str|int]], 
                        repeats: int, output_file: str, seed: int, num_worker: int):
    proxy_mat, lid2index, rid2index = to_matrix(proxy, len(ltable), len(rtable))

    with multiprocessing.Pool(num_worker) as pool:    
        job_args = [(sample_raito, ltable, rtable, labels, proxy_mat, lid2index, rid2index, repeats, output_file, seed, 0.95) for sample_raito in sample_ratios]
        pool.map(join, job_args)
    