import random
import bitarray
import numpy as np
import csv
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Tuple, Callable, List
import math
from scipy import stats
from tqdm import tqdm
import time
import multiprocessing
import sys


class BitSampleHash:
    def __init__(self, n: int, d: int, l: int, seed: int=233) -> None:
        random.seed(seed)
        self.seed = seed
        self.n = n
        self.d = d
        self.l = l
        self.bitmasks = [np.array(random.sample(range(n), k=d)) for _ in range(l)]

    def hash_int(self, x: str) -> int:
        ba = bitarray.bitarray()
        ba.frombytes(x.encode('utf-8'))
        x_bits = np.array(ba.tolist())
        x_hash_bits = [x_bits[bitmask] for bitmask in self.bitmasks]
        hash_value = 0
        count = 0
        for bit_list in x_hash_bits:
            for bit in bit_list:
                if bit:
                    hash_value += 2**count
                count += 1
        return hash_value

    def hash01(self, x:str) -> int:
        hash_int = self.hash_int(x)
        return float(hash_int) / 2 ** (self.d * self.l)
    
def check_dist():
    length = 1e10
    with open("../datasets/quora_questions.csv") as f:
        data = []
        reader = csv.reader(f)
        _ = next(reader)
        count = 0
        for row in reader:
            data.append(row[-1])
            if len(row[-1]) < length:
                length = len(row[-1])
            if length == 1:
                print(row)
                return
            count += 1
            if count >= 3500:
                break

    hasher = BitSampleHash(8*length, length, 10)
    print(length)
    hash_values = []
    for d in data:
        hash_values.append(hasher.hash_int(d))

    print(len(set(hash_values)))
    
    # # plt.hist(hash_values)
    # y = [0 for _ in data]
    # # print(hash_values)
    # plt.scatter(hash_values, y)
    # plt.show()

def get_universe_sampling_config(avd: dict, bvd: dict, e1: float, e2: float) -> Tuple[float]:
    gamma_11 = sum([avd[v] * bvd[v] for v in avd])
    gamma_12 = sum([avd[v] * bvd[v]**2 for v in avd])
    gamma_21 = sum([avd[v]**2 * bvd[v] for v in avd])
    gamma_22 = sum([avd[v]**2 * bvd[v]**2 for v in avd])
    if e1*e2*gamma_22 - gamma_12 - gamma_21 + gamma_11 < 0:
        return e1, 1, 1
    p = min(1, max(e1, e2, math.sqrt((e1*e2*gamma_22 - gamma_12 - gamma_21 + gamma_11)/gamma_11)))
    q1 = e1 / p
    q2 = e2 / p
    return p, q1, q2

def join(table: List[int], condition_evaluator: Callable[[int, int], bool], confidence_level: float=0.95):
    pairs = [[t1, t2] for t1 in table for t2 in table]
    results = []
    for pair in pairs:
        if condition_evaluator(*pair):
            results.append(1.)
        else:
            results.append(0.)
    print(results)
    results = np.array(results)
    mean = results.mean()
    ttest = stats.ttest_1samp(results, popmean=mean)
    ci = ttest.confidence_interval(confidence_level=confidence_level)
    return mean, ci.low, ci.high


def start_universe_sampling(jobid: int=0):
    sample_ratios = [0.00001*x for x in range(1,100)] + [0.001*x for x in range(1,100)] + [0.2, 0.5, 0.75, 1]
    length = 1e10
    qids = []
    with open("../datasets/quora_questions.csv") as f:
        data = []
        reader = csv.reader(f)
        _ = next(reader)
        count = 0
        for row in reader:
            data.append(row[-1])
            qids.append(int(row[0]))
            if len(row[-1]) < length:
                length = len(row[-1])
            if length == 1:
                print(row)
                return
            count += 1
            if count >= 3500:
                break

    with open("../datasets/oracle_positive_labels.csv") as f:
        positive_labels = set()
        reader = csv.reader(f)
        _ = next(reader)
        vd = defaultdict(int)
        for row in reader:
            qid1, qid2 = row
            assert(int(qid1) < int(qid2))
            positive_labels.add(f"{qid1}|{qid2}")
            vd[int(qid1)] += 1
            vd[int(qid2)] += 1
        
    
    def condition_evaluator(qid1: int, qid2: int) -> bool:
        if qid1 < qid2:
            evaluate_pair = f"{qid1}|{qid2}"
        elif qid1 > qid2:
            evaluate_pair = f"{qid2}|{qid1}"
        else:
            return False
        if evaluate_pair in positive_labels:
            return True
        else:
            return False

    hasher = BitSampleHash(8*length, length, 10, seed=time.time())
    hash_values_l = []
    for d in data:
        hash_values_l.append(hasher.hash_int(d))

    hash_value_dct = defaultdict(list)
    for qid, hash_value in zip(qids, hash_values_l):
        hash_value_dct[hash_value].append(qid)
    
    hash_value_sample_ratio = {}
    for idx, hash_value in enumerate(hash_value_dct.keys()):
        hash_value_sample_ratio[hash_value] = (idx+1) / len(hash_value_dct)
    
    for sample_ratio in sample_ratios:
        e1 = math.sqrt(sample_ratio)
        e2 = e1
        p, q1, q2 = get_universe_sampling_config(vd, vd, e1, e2)
        sampled_hash_values = []
        for hash_value in hash_value_sample_ratio:
            if hash_value_sample_ratio[hash_value] <= p:
                sampled_hash_values.append(hash_value)
        sample_qids = []
        for hash_value in sampled_hash_values:
            sample_qids += hash_value_dct[hash_value]

        print(len(sample_qids))

        if q1 == 1 and q2 == 1:
            mean, ci_lower, ci_upper = join(sample_qids, condition_evaluator)
        elif q1 != 1 and q2 != 1:
            assert q1 == q2
            sample_qids = random.choices(sample_qids, k=int(len(sample_qids)*q1))
            mean, ci_lower, ci_upper = join(sample_qids, condition_evaluator)
        else:
            raise Exception("exceptional case")
    
        # print(f"{sample_ratio}, {mean}, {ci_lower}, {ci_upper}")
        with open(f"usb/usb.csv{jobid}", "a+") as f:
            writer = csv.writer(f)
            writer.writerow([sample_ratio, mean, ci_lower, ci_upper])

if __name__ == "__main__":
    # check_dist()
    # job_id = list(range(100))
    # pool = multiprocessing.Pool(4)
    # for i in pool.map(start_universe_sampling, job_id):
    #     sys.stderr.write('\rpercentage of sample ratios completed: {0:%}'.format(i/len(job_id)))
        
    start_universe_sampling(0)
