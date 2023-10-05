import csv
import random
from typing import List
import math
import numpy as np
from scipy import stats
import multiprocessing

def join(args):
    table1, table2, labels, sample_ratio, output_file, confidence_level, repeats, seed = args
    def condition_evaluator(id1, id2):
        if id1 == id2:
            return False
        
        pair1 = f"{id1}|{id2}"
        pair2 = f"{id2}|{id1}"
        if pair1 in labels or pair2 in labels:
            return True
        else:
            return False

    random.seed(seed)
    sample_ratio1 = math.sqrt(sample_ratio * len(table1) / len(table2))
    sample_ratio2 = math.sqrt(sample_ratio * len(table2) / len(table1))
    sample_size1 = int(sample_ratio1 * len(table1))
    sample_size2 = int(sample_ratio2 * len(table2))

    ci_lower_bounds = []
    ci_upper_bounds = []
    sample_means = []

    for _ in range(repeats):
        # random sample
        table1_sample = random.sample(table1, k=sample_size1)
        table2_sample = random.sample(table2, k=sample_size2)
        sample_pairs = [[id1, id2] for id1 in table1_sample for id2 in table2_sample]
        sample_results = []

        ci_lower_bounds = []
        ci_upper_bounds = []
        sample_means = []

        for sample_pair in sample_pairs:
            if condition_evaluator(*sample_pair):
                sample_results.append(1.)
            else:
                sample_results.append(0.)
        
        # calculate stats
        sample_results = np.array(sample_results)
        sample_mean = sample_results.mean()
        ttest = stats.ttest_1samp(sample_results, popmean=sample_mean)
        ci = ttest.confidence_interval(confidence_level=confidence_level)
        ci_lower_bounds.append(ci.low)
        ci_upper_bounds.append(ci.high)
        sample_means.append(sample_mean)

    mean = np.average(sample_means)
    ci_lower_bound = np.average(ci_lower_bounds)
    ci_upper_bound = np.average(ci_upper_bounds)
    
    with open(output_file, "a+") as f:
        writer = csv.writer(f)
        writer.writerow([mean, ci_lower_bound, ci_upper_bound])

def run_ripple(sample_ratios: List[float], ltable: List[int|str], rtable: List[int|str],
               labels: set, repeats: int, output_file: str, seed: int=2333, num_worker: int=1):
    with multiprocessing.Pool(processes=num_worker) as pool:
        job_args = [[ltable, rtable, labels, sample_ratio, output_file, 0.95, repeats, seed] for sample_ratio in sample_ratios]
        pool.map(join, job_args)
