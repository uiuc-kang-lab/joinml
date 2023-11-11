from joinml.proxy.get_proxy import get_proxy
from joinml.dataset_loader import JoinDataset
from joinml.oracle import Oracle
from joinml.config import Config
from joinml.utils import set_up_logging, normalize

import numpy as np
from scipy import stats
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
    # get blocked data
    blocked_indexes = sorted_indexes[:-config.blocking_budget]
    blocked_scores = flattened_scores[blocked_indexes]
    blocked_scores /= np.sum(blocked_scores)
    blocked_ids = np.unravel_index(blocked_indexes, scores.shape)
    blocked_ids = np.array(blocked_ids).T
    blocked_table_ids = [[dataset_ids[0][blocked_ids[i][0]], dataset_ids[1][blocked_ids[i][1]]] for i in range(len(blocked_ids))]
    # get unblocked data
    unblocked_indexes = sorted_indexes[-config.blocking_budget:]
    unblocked_ids = np.unravel_index(unblocked_indexes, scores.shape)
    unblocked_ids = np.array(unblocked_ids).T
    unblocked_table_ids = [[dataset_ids[0][unblocked_ids[i][0]], dataset_ids[1][unblocked_ids[i][1]]] for i in range(len(unblocked_ids))]
    # running oracle on unblocked samples
    count = 0
    for unblocked_table_id in unblocked_table_ids:
        if oracle.query(unblocked_table_id):
            count += 1
    # sampling
    groundtruth = len(oracle.oracle_labels) - count
    logging.info(f"groundtruth for sampling: {groundtruth}")
    sample_sizes = [1000, 5000, 10000, 50000, 100000]
    for sample_size in sample_sizes:
        upper_bounds = []
        for i in range(config.repeats):
            samples = np.random.choice(len(blocked_scores), size=sample_size, p=blocked_scores)
            sample_table_ids = [blocked_table_ids[i] for i in samples]
            results = []
            for sample_table_id, sample in zip(sample_table_ids, samples):
                if oracle.query(sample_table_id):
                    results.append(1 / len(blocked_scores) / blocked_scores[sample])
                else:
                    results.append(0)
            results = np.array(results)
            if sum(results) != 0:
                ttest = stats.ttest_1samp(results, popmean=results.mean())
                ci = ttest.confidence_interval(confidence_level=0.95)
                upper_bound = ci.high
                estimated_count = np.average(results) * len(blocked_scores)
                upper_bound_count = upper_bound * len(blocked_scores)
                upper_bounds.append(upper_bound_count)
                logging.info(f"sample size {sample_size}, run time {i}, results {estimated_count}")
                logging.info(f"sample size {sample_size}, run time {i}, CI upper bound {upper_bound_count}")
        error = (np.average(upper_bounds) - groundtruth) / groundtruth
        amortised_error = (np.average(upper_bounds) - groundtruth) / len(oracle.oracle_labels)
        logging.info(f"sample size {sample_size}, average upper bound {np.average(upper_bounds)}, error {error}, amortised error {amortised_error}")
            