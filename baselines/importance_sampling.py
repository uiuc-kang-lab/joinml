import csv
import numpy as np
from typing import List, Callable
from tqdm import tqdm
from scipy import stats
import os

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
    
    for _ in tqdm(range(repeats)):
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

    return np.average(sample_means), np.average(ci_lower_bounds), np.average(ci_upper_bounds)

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
            mean, ci_lower, ci_upper = join(qids, qids, condition_evaluator, sample_ratio=sample_ratio, proposal=proposal, likelihood_ratios=likelihood, repeats=100)
            means.append(mean)
            ci_lowers.append(ci_lower)
            ci_uppers.append(ci_upper)
            print(sample_ratio, mean, ci_lower, ci_upper)

        # store results
        if os.path.exists("importance.csv"):
            f = open("importance.csv", 'a')
            writer = csv.writer(f)
        else:
            f = open("importance.csv", 'w')
            writer = csv.writer(f)
            writer.writerow(["sample_ratio", "mean", "ci_lower", "ci_upper"])
        for sample_ratio, mean, ci_lower, ci_upper in zip(sample_ratios, means, ci_lowers, ci_uppers):
            writer.writerow([sample_ratio, mean, ci_lower, ci_upper])
        f.close()

if __name__ == "__main__":
    sample_ratios = [0.00001*x for x in range(1,100)] + [0.001*x for x in range(1,100)] + [0.2, 0.5, 0.75, 1]
    run_importance(sample_ratios)


    
        