from joinml.config import Config
from joinml.oracle import Oracle
from joinml.utils import divide_sample_size

import numpy as np
from typing import List
from itertools import product

def run_ripple(oracle: Oracle, config: Config, ids: List[List[int]]):
    all_results = []
    sample_sizes = divide_sample_size(config.join_sample_size, [len(id) for id in ids])
    for _ in range(config.repeats):
        # run sampling
        samples = []
        for sample_size, table_ids in zip(sample_sizes, ids):
            table_samples = np.random.choice(table_ids, sample_size, replace=True)
            samples.append(table_samples)
        samples = np.array(list(product(*samples)))
        # run oracle
        results = []
        for sample in samples:
            if oracle.query(sample):
                results.append(1.)
            else:
                results.append(0.)
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
    print("run ripple join...")
    start = time.time()
    results = run_ripple(oracle, config, [ids, ids])
    results = np.array(results)
    end = time.time()
    print(f"cost {end - start} seconds")
