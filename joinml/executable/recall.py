from joinml.proxy.get_proxy import get_proxy_rank
from joinml.dataset_loader import load_dataset
from joinml.oracle import Oracle
from joinml.config import Config
from joinml.utils import set_up_logging

import os
import logging
import numpy as np


def run(config: Config):
    set_up_logging(config.log_path)

    # log config
    logging.info(config)

    # dataset, oracle
    dataset = load_dataset(config)
    oracle = Oracle(config)

    # setup dataset
    dataset_sizes = dataset.get_sizes()
    if config.is_self_join:
        dataset_sizes = (dataset_sizes[0], dataset_sizes[0])
    gt = len(oracle.oracle_labels)
    logging.info(f"data size: {dataset_sizes}")

    logging.info(f"ground truth: {gt}")

    # check cache for proxy rank
    proxy_rank = get_proxy_rank(config, dataset)

    count_all = 0
    count_positive = 0
    recall_results = []
    proxy_rank = np.flip(proxy_rank)
    for sample_id in proxy_rank:
        count_all += 1
        sample_table_id = np.array(np.unravel_index(sample_id, dataset_sizes)).T
        sample_table_id = sample_table_id.reshape(-1).tolist()
        if oracle.query(sample_table_id):
            count_positive += 1
        recall_results.append(count_positive / gt)
        logging.info(f"{count_all} {count_positive} {count_positive / gt}")
    
    recall_results = np.array(recall_results)
    logging.info("Recall results: %s", recall_results)
    # save recall results
    recall_store_path = f"{config.cache_path}/{config.dataset_name}_{config.proxy.split('/')[-1]}_recall.npy"
    np.save(recall_store_path, recall_results)