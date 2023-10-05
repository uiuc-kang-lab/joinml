import csv
import numpy as np
from typing import List, Callable
from tqdm import tqdm
from scipy import stats
import os
from multiprocessing import Pool, Lock
import sys
import logging

# construct the proposal distribution by normalizing the proxy score
# construct tne likelihood ratio by dividing the desired distribution by proposal distribution
# desired distribution is the uniform distribution
def construct_proposal_likelihood(proxy_scores_list: List[List[str|int]], size1: int, size2: int):
    proxy_score = np.ones((size1, size2)) * 1e-2
    lid2index = {}
    rid2index = {}
    index2lid = {}
    index2rid = {}
    l_countid = 0
    r_countid = 0
    for row in proxy_scores_list:
        id1, id2, score = row
        score = float(score)
        if id1 not in lid2index:
            lid2index[id1] = l_countid
            index2lid[l_countid] = id1
            lindex = l_countid
            l_countid += 1
        else:
            lindex = lid2index[id1]
        if id2 not in rid2index:
            rid2index[id2] = r_countid
            index2rid[r_countid] = id2
            rindex = r_countid
            r_countid += 1
        else:
            rindex = rid2index[id2]
        proxy_score[lindex, rindex] = score
    proposal = proxy_score / np.sum(proxy_score)
    desired = 1. / (size1 * size2) * np.ones((size1, size2))
    likelihood_ratios = desired / proposal

    return proposal, likelihood_ratios, lid2index, rid2index, index2lid, index2rid

# sample a subset of the data based on the provided sample size and weights
# weights should be a (0,1) probability value
def weighted_sampling(full_size: int, weights: List[float], size: int):
    sample_indexes = np.random.choice(full_size, size=size, p=weights, replace=True)
    return sample_indexes

# join based on importance sampling
def join(args):
    table1, table2, labels, proposal, likelihood_ratios, lid2index, rid2index, sample_ratio, output_file, confidence_level, repeats, seed = args
    
    def condition_evaluator(id1, id2):
        if id1 == id2:
            return False
        
        pair1 = f"{id1}|{id2}"
        pair2 = f"{id2}|{id1}"
        if pair1 in labels or pair2 in labels:
            return True
        else:
            return False
    
    # generate a full join
    full_join = [[id1, id2] for id1 in table1 for id2 in table2]
    sample_size = int(sample_ratio * len(full_join))
    np.random.seed(seed)

    ci_lower_bounds = []
    ci_upper_bounds = []
    sample_means = []
    
    for i in range(repeats):
        # importance sampling
        # flatten join likelihood ratios
        flattened_proposal = []
        flattened_likelihood_ratios = []
        for pair in full_join:
            index1 = lid2index[pair[0]]
            index2 = rid2index[pair[1]]
            flattened_likelihood_ratios.append(likelihood_ratios[index1, index2])
            flattened_proposal.append(proposal[index1, index2])

        sample_indexes = weighted_sampling(len(full_join), flattened_proposal, size=sample_size)
        sample_pairs = np.array(full_join)[sample_indexes]
        sample_likelihoods = np.array(flattened_likelihood_ratios)[sample_indexes]

        sample_results = []
        for sample_pair, sample_likelihood in zip(sample_pairs, sample_likelihoods):
            if condition_evaluator(*sample_pair):
                sample_results.append(sample_likelihood)
            else:
                sample_results.append(0)
        
        # calculate stats
        sample_results = np.array(sample_results)
        sample_mean = sample_results.mean()
        ttest = stats.ttest_1samp(sample_results, popmean=sample_mean)
        ci = ttest.confidence_interval(confidence_level=confidence_level)
        ci_lower_bounds.append(ci.low)
        ci_upper_bounds.append(ci.high)
        sample_means.append(sample_mean)
        logging.info(f"sample ratio {sample_ratio} finishes {i}/{repeats}")

    sample_mean = np.average(sample_means)
    ci_lower = np.average(ci_lower_bounds)
    ci_upper = np.average(ci_upper_bounds)

    # lock.acquire()
    
    with open(output_file, 'a+') as f:
        writer = csv.writer(f)
        writer.writerow([sample_ratio, sample_mean, ci_lower, ci_upper])

    # lock.release()
    
# def init_pool_process(_lock):
#     global lock
#     lock = _lock

def run_importance(sample_ratios: List[float], rtable: List[int|str], ltable: List[int|str], labels: set,
                   proxy: List[List[str]], repeats: int, output_file: str, seed: int, num_worker: int):
    proposal, likelihood, lid2index, rid2index, index2lid, index2rid = construct_proposal_likelihood(proxy, len(rtable), len(ltable))
        
    # lock = Lock()
    # pool = Pool(4, initializer=init_pool_process, initargs=(lock,))
    with Pool(num_worker) as pool:
        job_args = [[ltable, rtable, labels, proposal, likelihood, lid2index, rid2index, sample_ratio, output_file, 0.95, repeats, seed] for sample_ratio in sample_ratios]
        pool.imap(join, job_args)



    
        
