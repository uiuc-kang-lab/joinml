import csv
import random
from typing import List
import math
import numpy as np
from scipy import stats
import multiprocessing
from internal.table import Table
from dataset_loader import Dataset
from itertools import product

def join(args):
    dataset, sample_ratio, output_file, confidence_level, repeats, seed = args
    assert isinstance(dataset, Dataset)

    random.seed(seed)

    # calculate the sample rate for each table proportional to the size of the table
    # product of the sample rate is the sample rate for the whole join
    full_join_size = np.prod([len(table) for table in dataset.tables])
    sample_sizes = np.power(sample_ratio, 1 / len(dataset.tables)) / np.power(full_join_size, 1/len(dataset.tables)) * np.array([len(table)**2 for table in dataset.tables])
    if len(np.where(sample_sizes < 10)[0]) > 0:
        print(f"sample size {sample_sizes} is too small, skip")
        return
    ci_lower_bounds = []
    ci_upper_bounds = []
    sample_means = []

    for _ in range(repeats):
        # random sample one sample for each table
        table_samples = []
        for tid, sample_size in enumerate(sample_sizes):
            table_samples.append(random.sample(dataset.tables[tid].get_ids(), k=int(sample_size)))
        sample_pairs = product(*table_samples)
        # join
        join_results = dataset.evaluate_conditions(sample_pairs)
        # calculate stats
        ci_lower_bounds = []
        ci_upper_bounds = []
        sample_means = []
        sample_mean = join_results.mean()
        ttest = stats.ttest_1samp(join_results, popmean=sample_mean)
        ci = ttest.confidence_interval(confidence_level=confidence_level)
        ci_lower_bounds.append(ci.low)
        ci_upper_bounds.append(ci.high)
        sample_means.append(sample_mean)

    mean = np.average(sample_means)
    ci_lower_bound = np.average(ci_lower_bounds)
    ci_upper_bound = np.average(ci_upper_bounds)
    
    with open(output_file, "a+") as f:
        writer = csv.writer(f)
        writer.writerow([sample_ratio, mean, ci_lower_bound, ci_upper_bound])

def run_ripple(sample_ratios: List[float], dataset: Dataset, repeats: int, output_file: str, seed: int=2333, num_worker: int=1):
    with multiprocessing.Pool(processes=num_worker) as pool:
        job_args = [[dataset, sample_ratio, output_file, 0.95, repeats, seed] for sample_ratio in sample_ratios]
        pool.map(join, job_args)
