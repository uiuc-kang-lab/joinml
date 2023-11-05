from joinml.oracle import Oracle
from joinml.config import Config

import numpy as np
from typing import List
from numba import njit

def sample_for_wander_join(scores: List[np.ndarray], sample_size: int):
    table_0_size = scores[0].shape[0]
    samples = np.random.choice(table_0_size, sample_size, replace=True)
    samples = np.array([[sample] + [0 for _ in range(len(scores))] for sample in samples])
    sample_probs = np.ones(sample_size) / table_0_size
    for i in range(sample_size):
        for j in range(len(scores)):
            left_table_entry = samples[i][-1]
            prob_for_right_table = scores[j][left_table_entry]
            prob_for_right_table /= np.sum(prob_for_right_table)
            samples[i][j+1] = np.random.choice(scores[j].shape[1], p=prob_for_right_table)
            sample_probs[i] *= prob_for_right_table[samples[i][j+1]]
    return samples, sample_probs

njit
def convert_samples_to_ids(samples: np.ndarray, ids: List[List[int]]):
    sample_ids = np.zeros(samples.shape)
    for i in range(samples.shape[0]):
        for j in range(samples.shape[1]):
            sample_ids[i][j] = ids[j][samples[i][j]]
    return sample_ids

def run_weighted_wander(scores: List[np.ndarray], oracle: Oracle, config: Config, ids: List[List[int]]):
    all_results = []
    for _ in range(config.repeats):
        # run sampling
        samples, sample_probs = sample_for_wander_join(scores, config.join_sample_size)
        # convert samples to ids
        sample_ids = convert_samples_to_ids(samples, ids)
        # run oracle
        results = np.zeros(config.join_sample_size)
        num_tuples = scores[0].shape[0]
        for i in range(len(scores)):
            num_tuples *= scores[i].shape[1]
        for i, sample_id in enumerate(sample_ids):
            if oracle.query(sample_id):
                results[i] = 1 / num_tuples / sample_probs[i]
        all_results.append(np.array(results))
    return all_results

if __name__ == "__main__":
    from joinml.utils import normalize
    import time
    # test for one table
    config = Config(
        dataset_name="quora",
        data_path="../../data",
        repeats=20,
        join_sample_size=50000,   
    )
    oracle = Oracle(config)
    ids = list(range(10000))
    print("generate fake scores...")
    score = np.random.rand(10000, 10000)
    print(f"normalize {score.shape}...")
    score = normalize(score, is_self_join=True)
    print("run weighted wander join...")
    start = time.time()
    results = run_weighted_wander([score], oracle, config, [ids, ids])
    end = time.time()
    print(f"cost {end - start} seconds")

    