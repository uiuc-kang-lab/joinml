from typing import List
import numpy as np
from scipy import stats
import csv
import multiprocessing
import logging

from dataset_loader import Dataset


def wander_join(dataset: Dataset, sample_size: int, confidence_level: float=0.95):
    table1_weights = dataset.proxy.proxy_matrix.sum(axis=tuple(range(1, len(dataset.proxy.proxy_matrix.shape))))
    table1_weights = table1_weights / table1_weights.sum()
    samples = [[sample] for sample in np.random.choice(len(table1_weights), size=sample_size, 
                                                       p=table1_weights, replace=True)]
    sample_likelihoods = table1_weights[[sample[0] for sample in samples]]
    for _ in dataset.tables[1:]:
        for idx, sample in enumerate(samples):
            index = tuple(sample)
            table_weights = dataset.proxy.proxy_matrix[index]
            if len(table_weights.shape) > 1:
                table_weights = table_weights.sum(axis=tuple(range(1, len(table_weights.shape))))
            assert len(table_weights.shape) == 1
            table_weights = table_weights / table_weights.sum()
            new_sample = np.random.choice(len(table_weights), size=1, p=table_weights, replace=True)[0]
            samples[idx].append(new_sample)
            sample_likelihoods[idx] *= table_weights[new_sample]
    
    # join
    join_results = dataset.evaluate_conditions(samples) / np.prod([len(table) for table in dataset.tables]) / sample_likelihoods
    # calculate stats
    count_mean = join_results.mean()
    ttest = stats.ttest_1samp(join_results, popmean=count_mean)
    ci = ttest.confidence_interval(confidence_level=confidence_level)
    ci_lower_bound = ci.low
    ci_upper_bound = ci.high

    return count_mean, ci_lower_bound, ci_upper_bound


def join(args) -> None:
    sample_ratio, dataset, repeats, output_file, seed, confidence_level = args
    assert isinstance(dataset, Dataset)
    np.random.seed(seed)
    sample_size = int(np.prod([len(table) for table in dataset.tables]) * sample_ratio)
    results, uppers, lowers = [], [], []
    for i in range(repeats):
        count_result, ci_upper, ci_lower  = \
            wander_join(dataset, sample_size, confidence_level)
        results.append(count_result)
        uppers.append(ci_upper)
        lowers.append(ci_lower)
        logging.info(f"sample ratio {sample_ratio} finishes {i+1}")
    
    with open(output_file, "a+") as f:
        writer = csv.writer(f)
        writer.writerow(["{:.2}".format(sample_ratio), np.average(results), np.average(uppers), np.average(lowers)])


def run_weighted_wander(sample_ratios: List[float], dataset: Dataset, repeats: int, output_file: str, seed: int, num_worker: int):
    with multiprocessing.Pool(num_worker) as pool:    
        job_args = [(sample_raito, dataset, repeats, output_file, seed, 0.95) for sample_raito in sample_ratios]
        pool.map(join, job_args)
    