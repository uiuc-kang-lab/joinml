from joinml.proxy.get_proxy import get_proxy
from joinml.dataset_loader import JoinDataset
from joinml.oracle import Oracle
from joinml.config import Config
from joinml.utils import set_up_logging, normalize

import numpy as np
import logging

def run(config: Config):
    set_up_logging(config.log_path)

    proxy = get_proxy(config)
    dataset = JoinDataset(config)
    oracle = Oracle(config)

    dataset_ids = dataset.get_ids()
    join_cols = dataset.get_join_column_per_table(dataset_ids)
    if config.is_self_join:
        dataset_ids = [dataset_ids[0], dataset_ids[0]]
        join_cols = [join_cols[0], join_cols[0]]
    dataset_shape = [len(dataset_id) for dataset_id in dataset_ids]
    scores = proxy.get_proxy_score_for_tables(join_cols[0], join_cols[1], is_self_join=config.is_self_join)
    logging.info("preprocessing")
    if config.is_self_join:
        assert len(scores.shape) == 2 and scores.shape[0] == scores.shape[1]
        # set diagonal to min
        scores[np.diag_indices(scores.shape[0])] = np.min(scores)
    logging.info("flattening")
    scores = scores.flatten()
    logging.info("sorting")
    sorted_indexes = scores.argsort()
    np.save("sorted_flatten_index.npy", sorted_indexes)
    sorted_indexes = np.flip(sorted_indexes)
    count = 0
    groundtruth = len(oracle.oracle_labels)
    for blocking_budget, flatten_index in enumerate(sorted_indexes):
        index = np.array(np.unravel_index(flatten_index, dataset_shape)).T
        index = index.tolist()
        if oracle.query(index):
            count += 1
        logging.info(f"Budget: {blocking_budget+1}, Index: {index}, Score: {scores[flatten_index]}, Positive pairs: {count}")
        if count == groundtruth:
            break

        