from typing import List
import numpy as np
from scipy import stats
import csv
import multiprocessing
import logging
from dataset_loader import Dataset

def wander_join(dataset: Dataset, sample_size: int, confidence_level: float=0.95):
    print("start sampling")
    sampled_indexes = np.random.choice(len(dataset.proxy.proxy_matrix), size=sample_size, p=dataset.proxy.proxy_matrix, replace=True)
    print("end sampling")
    sample_likelihoods = dataset.proxy.proxy_matrix[sampled_indexes]
    samples = [(i // (1000*1000), (i // 1000) % 1000, i % 1000) for i in sampled_indexes]
    
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
        print(f"sample ratio {sample_ratio} finishes {i+1}")
    
    with open(output_file, "a+") as f:
        writer = csv.writer(f)
        writer.writerow(["{:.2}".format(sample_ratio), np.average(results), np.average(uppers), np.average(lowers)])



def run_weighted_wander(sample_ratios: List[float], dataset: Dataset, repeats: int, output_file: str, seed: int, num_worker: int):
    dataset.proxy.process_for_weighted_wander_join(dataset.name)
    for sample_ratio in sample_ratios:
        args = (sample_ratio, dataset, repeats, output_file, seed, 0.95)
        join(args)
    