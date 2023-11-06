from joinml.oracle import Oracle
from joinml.config import Config

from typing import List
import numpy as np
from itertools import product
import logging
from tqdm import tqdm

def run_uniform(oracle: Oracle, config: Config, ids: List[List[int]]):
    all_results = []
    # get all tuples from ids
    tuples = np.array(list(product(*ids)))
    for _ in range(config.repeats):
        # run sampling
        samples = np.random.choice(len(tuples), config.join_sample_size, replace=True)
        samples = tuples[samples]
        # run oracle
        results = []
        for sample in tqdm(samples):
            if oracle.query(sample):
                results.append(1)
            else:
                results.append(0)
        all_results.append(np.array(results))
        logging.info(f"uniform sampling join results: {np.average(all_results[-1])}" )
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
    print("run naive uniform sampling join...")
    start = time.time()
    results = run_uniform(oracle, config, [ids, ids])
    end = time.time()
    print(f"cost {end - start} seconds")

