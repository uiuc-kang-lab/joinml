import csv
import random
import sys, os
from tqdm import tqdm
from typing import List, Callable
import numpy as np
from scipy import stats

def join(table1: List[int], table2: List[int], condition_evaluator: Callable[[int, int], bool], 
         sample_ratio: float, confidence_level: float=0.95, repeats: int=10, seed: int=233):
    
    random.seed(seed)
    all_pairs = [[id1, id2] for id1 in table1 for id2 in table2]
    print(sample_ratio)
    sample_size = int(sample_ratio * len(all_pairs))
    
    ci_lower_bounds = []
    ci_upper_bounds = []
    sample_means = []

    for _ in tqdm(range(repeats)):
        # random sample
        sample_pairs = random.sample(all_pairs, k=sample_size)
        sample_results = []

        # join
        for sample_pair in sample_pairs:
            if condition_evaluator(*sample_pair):
                sample_results.append(1)
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
    
    return np.average(sample_means), np.average(ci_lower_bounds), np.average(ci_upper_bounds)


def run_random(sample_ratios: List[float], dataset: str="QQP", size: int=3500):
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
        
        groundtruth = len(positive_labels) * 2 / (3500*3500)
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
        if os.path.exists("random_naive.csv"):
            f = open("random_naive.csv", 'a')
            writer = csv.writer(f)
        else:
            f = open("random_naive.csv", 'w')
            writer = csv.writer(f)
            writer.writerow(["sample_ratio", "mean", "ci_lower", "ci_upper"])
        for sample_ratio, mean, ci_lower, ci_upper in zip(sample_ratios, means, ci_lowers, ci_uppers):
            writer.writerow([sample_ratio, mean, ci_lower, ci_upper])
        f.close()
        
if __name__ == '__main__':
    sample_ratios = [0.00001*x for x in range(1,100)] + [0.001*x for x in range(1,100)] + [0.2, 0.5, 0.75, 1]
    run_random(sample_ratios)

