import csv
import random
import sys
from typing import List, Callable
import math
from tqdm import tqdm
import numpy as np
from scipy import stats
import os

sample_ratios = [x/100 for x in range(1, 30)]
table_size = 3500
seed = 233
random.seed(233)

qids = []
with open("../datasets/quora_questions.csv") as f:
    reader = csv.reader(f)
    _ = next(reader)
    count = 0
    for row in reader:
        id, _ = row
        qids.append(int(id))
        count += 1
        if count >= 3500:
            break

positive_pairs = []
with open("../datasets/oracle_positive_labels.csv") as f:
    reader = csv.reader(f)
    _ = next(reader)
    for row in reader:
        qid1, qid2 = row
        positive_pairs.append([int(qid1), int(qid2)])

outputs = []
for sample_ratio in sample_ratios:
    sample_size = int(sample_ratio * table_size)
    n_positive = 0
    run_times = 10
    for _ in range(10):
        table1 = random.sample(qids, k=sample_size)
        table2 = random.sample(qids, k=sample_size)
        for positive_pair in positive_pairs:
            qid1, qid2 = positive_pair
            if (qid1 in table1 and qid2 in table2) or (qid1 in table2 and qid2 in table1):
                n_positive += 1
    n_positive /= run_times
    estimated_n_postivies = int(n_positive * table_size**2 / sample_size ** 2)
    error = abs(estimated_n_postivies - len(positive_pairs)*2) / (len(positive_pairs)*2)
    n_oracle_calls = sample_size ** 2 / table_size ** 2
    outputs.append(["{:.2}".format(sample_ratio), n_positive, estimated_n_postivies, "{:.2}".format(error), "{:.2}".format(n_oracle_calls)])    

with open("ripple_join.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["sample ratio", "number of positives", "estimated number of positives", "error", "oracle calls"])
    writer.writerows(outputs)


def join(table1: List[int], table2: List[int], condition_evaluator: Callable[[int, int], bool],
         sample_ratio: float, confidence_level: float=0.95, repeats: int=10, seed: int=233):
    
    assert len(table1) == len(table2)
    random.seed(seed)
    sample_ratio_per_table = math.sqrt(sample_ratio)
    sample_size = int(sample_ratio_per_table * len(table1))

    ci_lower_bounds = []
    ci_upper_bounds = []
    sample_means = []

    for _ in tqdm(range(repeats)):
        # random sample
        table1_sample = random.sample(table1, k=sample_size)
        table2_sample = random.sample(table2, k=sample_size)
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
    
    return np.average(sample_means), np.average(ci_lower_bounds), np.average(ci_upper_bounds)

def run_ripple(sample_ratios: List[float], dataset: str="QQP", size: int=3500):
    if dataset == "QQP":
        # read data
        qids = []
        with open("../datasets/quora_questions.csv") as f:
            reader = csv.reader(f)
            _ = next(reader)
            count = 0
            for row in reader:
                qid_str, _ = row
                qids.append(int(qid_str))
                count += 1
                if count >= size:
                    break
        
        # define evaluator
        positive_labels = set()
        with open("../datasets/oracle_positive_labels.csv") as f:
            reader = csv.reader(f)
            _ = next(reader)
            for row in reader:
                qid1, qid2 = row
                positive_labels.add(f"{qid1}|{qid2}")
        
        groundtruth = len(positive_labels) * 2 / (3500 * 3500)
        print(f"query groundtruth is {groundtruth}, {len(positive_labels)*2}")

        def condition_evaluator(qid1, qid2):
            evaluate_pair = None
            if qid1 > qid2:
                evaluate_pair = f"{qid2}|{qid1}"
            elif qid1 < qid2:
                evaluate_pair = f"{qid1}|{qid2}"
            else:
                return False
            if evaluate_pair in positive_labels:
                return True
            else:
                return False
            
        # run join
        means, ci_lowers, ci_uppers = [], [], []
        for sample_ratio in sample_ratios:
            mean, ci_lower, ci_upper = join(qids, qids, condition_evaluator, sample_ratio, repeats=100)
            means.append(mean)
            ci_lowers.append(ci_lower)
            ci_uppers.append(ci_upper)
            print(sample_ratio, mean, ci_lower, ci_upper)

        # store results
        if os.path.exists("ripple.csv"):
            f = open("ripple.csv", 'a')
            writer = csv.writer(f)
        else:
            f = open("ripple.csv", 'w')
            writer = csv.writer(f)
            writer.writerow(["sample_ratio", "mean", "ci_lower", "ci_upper"])
        for sample_ratio, mean, ci_lower, ci_upper in zip(sample_ratios, means, ci_lowers, ci_uppers):
            writer.writerow([sample_ratio, mean, ci_lower, ci_upper])
        f.close()
        
if __name__ == '__main__':
    sample_ratios = [0.00001*x for x in range(1,100)] + [0.001*x for x in range(1,100)] + [0.2, 0.5, 0.75, 1]
    run_ripple(sample_ratios)