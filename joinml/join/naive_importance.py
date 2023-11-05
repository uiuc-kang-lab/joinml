from joinml.oracle import Oracle
from joinml.config import Config

import numpy as np
from typing import List
from numba import njit

njit
def convert_table_ids_to_table_indices(samples: np.ndarray, ids: List[List[int]]):
    sample_ids = np.zeros(samples.shape)
    for i in range(samples.shape[0]):
        for j in range(samples.shape[1]):
            sample_ids[i][j] = ids[j][samples[i][j]]
    return sample_ids


def run_naive_importance(scores: np.ndarray, oracle: Oracle, config: Config, ids: List[List[int]]):
    all_results = []
    for _ in range(config.repeats):
        # run sampling
        samples = np.random.choice(np.prod(scores.shape), config.join_sample_size, replace=True, p=scores.flatten())
        # samples are flattened indices
        # convert them to table indices
        samples = np.unravel_index(samples, scores.shape)
        samples = np.array(samples).T
        # convert table indices to table ids
        sample_ids = convert_table_ids_to_table_indices(samples, ids)
        # run oracle
        results = []
        n_total_tuples = np.prod(scores.shape)
        for sample in sample_ids:
            if oracle.query(sample):
                results.append(1 / n_total_tuples / scores[sample])
            else:
                results.append(0)
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
    print("run naive importance sampling join...")
    start = time.time()
    results = run_naive_importance(score, oracle, config, [ids, ids])
    end = time.time()
    print(f"cost {end - start} seconds")

