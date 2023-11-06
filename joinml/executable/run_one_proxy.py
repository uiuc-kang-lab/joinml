from joinml.config import Config
from joinml.proxy.get_proxy import get_proxy
from joinml.utils import set_random_seed, set_up_logging, normalize
from joinml.dataset_loader import JoinDataset
from joinml.oracle import Oracle
from joinml.join.run_join import run as run_join
from joinml.meter import ErrorMeter

import logging
import os
import pickle

def run(config: Config):
    # set up
    set_up_logging(config.log_path)
    set_random_seed(config.seed)
    # dataset
    dataset = JoinDataset(config)
    dataset_ids = dataset.get_ids()
    dataset_join_cols = dataset.get_join_column_per_table(dataset_ids)
    # oracle
    oracle = Oracle(config)
    # error meter
    error_meter = ErrorMeter(dataset, oracle, config)
    # proxy
    proxy = get_proxy(config)
    scores = []
    if config.join_algorithm in ["weighted_wander", "naive_importance"]:
        # get scores
        proxy_cache_path = f"{config.data_path}/{config.dataset_name}/proxy_cache"
        if not os.path.exists(f"{proxy_cache_path}/{config.proxy}.pkl"):
            logging.info(f"Running {config.proxy}...")
            if config.is_self_join:
                scores = proxy.get_proxy_score_for_tables(dataset_join_cols[0], dataset_join_cols[0])
                scores = [normalize(scores, is_self_join=True)]
            else:
                scores = []
                for i in range(1,len(dataset_join_cols)):
                    score = proxy.get_proxy_score_for_tables(dataset_join_cols[i-1], dataset_join_cols[i])
                    scores.append(normalize(score))
                assert len(scores) == len(dataset_join_cols) - 1
        else:
            logging.info(f"Loading {config.proxy} from cache...")
            with open(f"{proxy_cache_path}/{config.proxy}.pkl", "rb") as f:
                scores = pickle.load(f)
        if config.proxy_cache and not os.path.exists(f"{proxy_cache_path}/{config.proxy}.pkl"):
            if not os.path.exists(proxy_cache_path):
                os.makedirs(proxy_cache_path)
            with open(f"{proxy_cache_path}/{config.proxy}.pkl", "wb") as f:
                pickle.dump(scores, f)
    # run join
    results = run_join(config, scores, dataset_ids, oracle)
    # report
    for result in results:
        error_meter.add_results(result)
    error_meter.report()
