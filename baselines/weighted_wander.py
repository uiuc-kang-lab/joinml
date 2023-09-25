from typing import List, Callable
import numpy as np
from scipy import stats
import csv
from tqdm import tqdm
import multiprocessing
import sys

num_workers = int(sys.argv[1])

# read oracle results
oracle_results = set()
with open("../datasets/oracle_positive_labels.csv") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        qid1, qid2 = row
        qid1 = int(qid1)
        qid2 = int(qid2)
        assert qid1 < qid2
        oracle_results.add(f"{qid1}|{qid2}")

print(f"groundtruth: {len(oracle_results)*2 / 3500 / 3500}")

# prepare condition evaluator
def condition_evaluator(qid1: int, qid2: int) -> bool:
    evaluate_pair = ""
    if qid1 < qid2:
        evaluate_pair = f"{qid1}|{qid2}"
    elif qid1 > qid2:
        evaluate_pair = f"{qid2}|{qid1}"
    else:
        return False
    if evaluate_pair in oracle_results:
        return True
    else:
        return False

def wander_join(table1: List[int], table2: List[int], condition_evaluator: Callable[[int, int], bool], 
                proxy_score: np.ndarray, sample_size: int, confidence_level: float=0.95):
    # sum over one axis of proxy score
    table1 = np.array(table1)
    table2 = np.array(table2)
    table1_weights = proxy_score.sum(axis=0)
    table1_weights = table1_weights / table1_weights.sum()
    table1_samples_indexes = np.random.choice(len(table1), size=sample_size, replace=True, p=table1_weights)
    count_results = []
    for table1_sample_index in table1_samples_indexes:
        table2_weights = proxy_score[table1_sample_index]
        table2_weights /= table2_weights.sum()
        table2_sample_index = np.random.choice(len(table2), size=1, replace=True, p=table2_weights)[0]
        table1_sample_weight = table1_weights[table1_sample_index]
        table2_sample_weight = table2_weights[table2_sample_index]
        sample_pair = [table1[table1_sample_index], table2[table2_sample_index]]
        condition = condition_evaluator(*sample_pair)
        if condition:
            count_results.append(1 / 3500 / 3500 / (table1_sample_weight * table2_sample_weight))
        else:
            count_results.append(0)
    count_results = np.array(count_results)
    count_mean = count_results.mean()
    ttest = stats.ttest_1samp(count_results, popmean=count_mean)
    ci = ttest.confidence_interval(confidence_level=confidence_level)
    ci_lower_bound = ci.low
    ci_upper_bound = ci.high

    return count_mean, ci_lower_bound, ci_upper_bound

def run_weighted_wander(args) -> None:
    sample_ratio, table, proxy_score, repeats, seed, confidence_level = args
    np.random.seed(seed)
    sample_size = int(len(table) * len(table) * sample_ratio)
    results, uppers, lowers = [], [], []
    for i in range(repeats):
        count_result, ci_upper, ci_lower  = wander_join(table, table, condition_evaluator, proxy_score, sample_size, confidence_level)
        results.append(count_result)
        uppers.append(ci_upper)
        lowers.append(ci_lower)
        print(f"sample ratio {sample_ratio} finishes {i+1}")
    
    print(sample_ratio, np.average(results), np.average(uppers), np.average(lowers))

    with open("weighted_wander.csv", "a+") as f:
        writer = csv.writer(f)
        writer.writerow([sample_ratio, np.average(results), np.average(uppers), np.average(lowers)])

if __name__ == "__main__":
    # read proxy score into a numpy matrix
    proxy_score = np.zeros((3500, 3500))
    with open("../datasets/proxy_cosine_qqp.csv") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            qid1, qid2, score = row
            qid1 = int(qid1)
            qid2 = int(qid2)
            score = float(score)
            if qid1 == qid2:
                proxy_score[qid1-1, qid2-1] = 0
            else:
                proxy_score[qid1-1, qid2-1] = score
                proxy_score[qid2-1, qid1-1] = score
    
    # prepare data
    QQP = []
    with open("../datasets/quora_questions.csv") as f:
        reader = csv.reader(f)
        header = next(reader)
        count = 0
        for row in reader:
            qid, _ = row
            QQP.append(int(qid))
            count += 1
            if count >= 3500:
                break

    sample_ratios = [0.00001*x for x in range(1,100)] + [0.001*x for x in range(1,100)] + [0.2, 0.5, 0.75, 1]
    job_args = [(sample_raito, QQP, proxy_score, 100, 233, 0.95) for sample_raito in sample_ratios]
    pool = multiprocessing.Pool(num_workers)
    for i, _ in enumerate(pool.map(run_weighted_wander, job_args)):
        sys.stderr.write('\rpercentage of sample ratios completed: {0:%}'.format(i/len(job_args)))
    