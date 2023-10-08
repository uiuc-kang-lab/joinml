import csv
import numpy as np
from typing import List, Callable
from tqdm import tqdm
from scipy import stats
import os
from multiprocessing import Pool, Lock
import sys
import logging
from itertools import product
from dataset_loader import Dataset

# construct the proposal distribution by normalizing the proxy score
# construct tne likelihood ratio by dividing the desired distribution by proposal distribution
# desired distribution is the uniform distribution
def construct_proposal_likelihood(dataset: Dataset):
    proposal = dataset.get_proxy_matrix() / np.sum(dataset.get_proxy_matrix())
    sizes = [len(table) for table in dataset.tables]
    desired = 1. / np.prod(sizes) * np.ones(tuple(sizes))
    likelihood_ratios = desired / proposal
    return proposal, likelihood_ratios

# sample a subset of the data based on the provided sample size and weights
# weights should be a (0,1) probability value
def weighted_sampling(full_size: int, weights: List[float], size: int):
    sample_indexes = np.random.choice(full_size, size=size, p=weights, replace=True)
    return sample_indexes

# join based on importance sampling
def join(args):
    dataset, proposal, likelihood_ratios, sample_ratio, output_file, confidence_level, repeats, seed = args
    
    # generate a full join
    full_join_results = np.array(list(product(*[table.get_ids() for table in dataset.tables])))
    sample_size = int(sample_ratio * len(full_join_results))
    print(sample_size)
    np.random.seed(seed)

    ci_lower_bounds = []
    ci_upper_bounds = []
    sample_means = []
    for i in range(repeats):
        # importance sampling
        # flatten join likelihood ratios
        ## TODO: TEST if the following is correct
        flattened_proposal = np.reshape(proposal, -1)
        flattened_likelihood_ratios = np.reshape(likelihood_ratios, -1)
        sample_indexes = weighted_sampling(len(full_join_results), flattened_proposal, size=sample_size)
        sample_pairs = full_join_results[sample_indexes]
        sample_likelihoods = flattened_likelihood_ratios[sample_indexes]
        # join
        join_results = dataset.evaluate_conditions(sample_pairs) * sample_likelihoods
        # calculate stats
        sample_mean = join_results.mean()
        ttest = stats.ttest_1samp(join_results, popmean=sample_mean)
        ci = ttest.confidence_interval(confidence_level=confidence_level)
        ci_lower_bounds.append(ci.low)
        ci_upper_bounds.append(ci.high)
        sample_means.append(sample_mean)
        print(f"sample ratio {sample_ratio} finishes {i}/{repeats}")

    sample_mean = np.average(sample_means)
    ci_lower = np.average(ci_lower_bounds)
    ci_upper = np.average(ci_upper_bounds)

    # lock.acquire()
    
    with open(output_file, 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(["{:.2}".format(sample_ratio), sample_mean, ci_lower, ci_upper])

    # lock.release()
    
# def init_pool_process(_lock):
#     global lock
#     lock = _lock

def run_importance(sample_ratios: List[float], dataset: Dataset, repeats: int, output_file: str, seed: int, num_worker: int):
    proposal, likelihood = construct_proposal_likelihood(dataset)
    with Pool(num_worker) as pool:
        job_args = [[dataset, proposal, likelihood, sample_ratio, output_file, 0.95, repeats, seed] for sample_ratio in sample_ratios]
        pool.map(join, job_args)



    
        
