import csv
import random
from tqdm import tqdm
from typing import List
import numpy as np
from scipy import stats
import multiprocessing
from internal.table import Table
from dataset_loader import Dataset
from itertools import product
import logging

def join(args):
    dataset, sample_ratio, output_file, confidence_level, repeats, seed = args
    assert isinstance(dataset, Dataset)
    
    random.seed(seed)
    full_join_results = list(product(*[table.get_ids() for table in dataset.tables]))
    sample_size = int(sample_ratio * len(full_join_results))
    
    ci_lower_bounds = []
    ci_upper_bounds = []
    sample_means = []

    for i in range(repeats):
        # random sample
        sample_pairs = random.sample(full_join_results, k=sample_size)
        # join
        join_results = dataset.evaluate_conditions(sample_pairs)
        # calculate stats
        sample_mean = join_results.mean()
        ttest = stats.ttest_1samp(join_results, popmean=sample_mean)
        ci = ttest.confidence_interval(confidence_level=confidence_level)
        ci_lower_bounds.append(ci.low)
        ci_upper_bounds.append(ci.high)
        sample_means.append(sample_mean)
        print(f"sample ratio {sample_ratio} finishes {i+1}")
    
    print("done sample ratio {:.2}: sample_means: {}, ci_lows: {}, ci_highs: {}"
                 .format(sample_ratio, sample_means, ci_lower_bounds, ci_upper_bounds))

    mean = np.average(sample_means)
    ci_lower_bound = np.average(ci_lower_bounds)
    ci_upper_bound = np.average(ci_upper_bounds)
    
    with open(output_file, "a+") as f:
        writer = csv.writer(f)
        writer.writerow([f"{sample_ratio:.2}", mean, ci_lower_bound, ci_upper_bound])


def run_random(sample_ratios: List[float], dataset: Dataset, repeats: int, output_file: str, seed: int=2333, num_worker: int=1):
    # with multiprocessing.Pool(processes=num_worker) as pool:
    #     job_args = [[dataset, sample_ratio, output_file, 0.95, repeats, seed] for sample_ratio in sample_ratios]
    #     pool.map(join, job_args)
    dataset.proxy = None
    for sample_ratio in tqdm(sample_ratios):
        join([dataset, sample_ratio, output_file, 0.95, repeats, seed])



