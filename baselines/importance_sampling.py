import csv
import numpy as np
from typing import List, Callable
from tqdm import tqdm
from scipy import stats
import os
from multiprocessing import Pool, Lock
import sys

# construct the proposal distribution by normalizing the proxy score
# construct tne likelihood ratio by dividing the desired distribution by proposal distribution
# desired distribution is the uniform distribution
def construct_proposal_likelihood(proxy_file: str, size: int=3500):
    proxy_score = np.zeros((size, size))
    with open(proxy_file) as f:
        reader = csv.reader(f)
        _ = next(reader)
        for row in reader:
            qid1, qid2, score = row
            score = float(score)
            if score == 0 or qid1 == qid2:
                score = 2e-2
            proxy_score[int(qid1)-1, int(qid2)-1] = score
    proposal = proxy_score / np.sum(proxy_score)
    desired = 1. / (size*size) * np.ones((size, size))
    likelihood_ratios = desired / proposal

    return proposal, likelihood_ratios

# sample a subset of the data based on the provided sample size and weights
# weights should be a (0,1) probability value
def weighted_sampling(full_size: int, weights: List[float], size: int):
    sample_indexes = np.random.choice(full_size, size=size, p=weights, replace=True)
    return sample_indexes

def join_job(args):
    table1, table2, condition_evaluator, proposal, likelihood_ratios, sample_ratio, confidence_level, repeats, seed = args
    join(table1=table1, table2=table2, condition_evaluator=condition_evaluator, proposal=proposal,
         likelihood_ratios=likelihood_ratios, sample_ratio=sample_ratio, confidence_level=confidence_level, repeats=repeats,
         seed=seed)

# join based on importance sampling
def join(table1: List[int], table2: List[int], condition_evaluator: Callable[[int, int], bool],
         proposal, likelihood_ratios,
         sample_ratio: float, confidence_level: float=0.95, repeats: int=10, seed: int=3):
    # generate a full join
    full_join = [[id1, id2] for id1 in table1 for id2 in table2]
    assert len(full_join) == 3500*3500
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
            flattened_likelihood_ratios.append(likelihood_ratios[pair[0]-1, pair[1]-1])
            flattened_proposal.append(proposal[pair[0]-1, pair[1]-1])

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
        print(f"sample ratio {sample_ratio} finishes {i}/{repeats}")

    sample_mean = np.average(sample_means)
    ci_lower = np.average(ci_lower_bounds)
    ci_upper = np.average(ci_upper_bounds)

    lock.acquire()

    print(sample_ratio, sample_mean, ci_lower, ci_upper)

    
    if os.path.exists("importance.csv"):
        f = open("importance.csv", 'a')
        writer = csv.writer(f)
    else:
        f = open("importance.csv", 'w')
        writer = csv.writer(f)
        writer.writerow(["sample_ratio", "mean", "ci_lower", "ci_upper"])

    writer.writerow([sample_ratio, sample_mean, ci_lower, ci_upper])
    f.close()
    lock.release()

    return np.average(sample_means), np.average(ci_lower_bounds), np.average(ci_upper_bounds)

# define evaluator
positive_labels = set()
with open("../datasets/oracle_positive_labels.csv") as f:
    reader = csv.reader(f)
    _ = next(reader)
    for row in reader:
        qid1, qid2 = row
        positive_labels.add(f"{qid1}|{qid2}")

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
    
def init_pool_process(_lock):
    global lock
    lock = _lock

def run_importance(sample_ratios: List[float], dataset: str="QQP", size: int=3500):
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

        proposal, likelihood = construct_proposal_likelihood("../datasets/proxy_cosine_qqp.csv")

        groundtruth = len(positive_labels) * 2 / (3500*3500)
        print(f"query groundtruth is {groundtruth}, {len(positive_labels)*2}")
            
        lock = Lock()

        pool = Pool(40, initializer=init_pool_process, initargs=(lock,))

        args = [[qids, qids, condition_evaluator, proposal, likelihood, sample_ratio, 0.95, 10, 233] for sample_ratio in sample_ratios]
        # run join
        for i, _ in enumerate(pool.map(join_job, args)):
            sys.stderr.write('\rpercentage of sample ratios completed: {0:%}'.format(i/len(args)))
        

if __name__ == "__main__":
    sample_ratios = [0.00001*x for x in range(1,100)] + [0.001*x for x in range(1,100)] + [0.2, 0.5, 0.75, 1]
    run_importance(sample_ratios)


    
        
