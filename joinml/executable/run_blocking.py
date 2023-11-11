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
    scores = proxy.get_proxy_score_for_tables(join_cols[0], join_cols[1])
    scores = normalize(scores, is_self_join=config.is_self_join)
    flattened_scores = scores.flatten()
    sorted_indexes = flattened_scores.argsort()
    budgets = [1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]
    for budget in budgets:
        unblocked_indexes = sorted_indexes[-budget:]
        unblocked_ids = np.unravel_index(unblocked_indexes, scores.shape)
        unblocked_ids = np.array(unblocked_ids).T
        unblocked_table_ids = [[dataset_ids[0][unblocked_ids[i][0]], dataset_ids[1][unblocked_ids[i][1]]] for i in range(len(unblocked_ids))]
        # run oracle
        count = 0
        for table_ids in unblocked_table_ids:
            if oracle.query(table_ids):
                count += 1
        logging.info(f"Budget: {budget}, Count: {count}")
        