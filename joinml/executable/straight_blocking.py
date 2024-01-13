from joinml.proxy.get_proxy import get_proxy_score, get_proxy_rank
from joinml.dataset_loader import load_dataset, JoinDataset
from joinml.oracle import Oracle
from joinml.config import Config
from joinml.utils import set_up_logging, normalize
from joinml.estimates import Estimates
from joinml.utils import get_ci_bootstrap, weighted_sample_pd

import logging
import numpy as np

def run(config: Config):
    set_up_logging(config.log_path, config.log_level)

    # log config
    logging.info(config)

    # dataset, oracle
    dataset = load_dataset(config)
    oracle = Oracle(config)

    # setup dataset
    dataset_sizes = dataset.get_sizes()
    if config.is_self_join:
        dataset_sizes = (dataset_sizes[0], dataset_sizes[0])

    count_gt, sum_gt, avg_gt = dataset.get_gt(oracle)

    sampling_budget = config.oracle_budget // 2
    blocking_budget = config.oracle_budget - sampling_budget
    proxy_weights = get_proxy_score(config, dataset)

    # get a uniform sampling of the whole dataset
    sample = np.random.choice(np.prod(dataset_sizes), size=sampling_budget)
    positive_sample_scores = []
    sample_ids = np.array(np.unravel_index(sample, dataset_sizes)).T
    for s, sample_id in zip(sample, sample_ids):
        if oracle.query(sample_id):
            positive_sample_scores.append(proxy_weights[s])
    if len(positive_sample_scores) == 0:
        return
    cutoff_score = min(positive_sample_scores)
    logging.debug(f"cutoff_score: {cutoff_score}")

    # get the id of data with a score larger than the cutoff score
    unblocked_population = np.argwhere(proxy_weights >= cutoff_score).reshape(-1)
    logging.debug(f"size of unblocked population {len(unblocked_population)}")

    # get a uniform sampling of the unblocked dataset
    unblocked_sample = np.random.choice(unblocked_population, size=blocking_budget)
    unblocked_sample_ids = np.array(np.unravel_index(unblocked_sample, dataset_sizes)).T
    sample_count = 0
    sample_sum = 0
    for sample_id in unblocked_sample_ids:
        if oracle.query(sample_id):
            sample_count += 1
            sample_sum += dataset.get_statistics(sample_id)
    # get the estimates
    count_estimate = sample_count / blocking_budget * len(unblocked_population)
    sum_estimate = sample_sum / blocking_budget * len(unblocked_population)
    avg_estimate = sample_sum / sample_count

    count_est = Estimates(config.oracle_budget, count_gt, count_estimate, 0, 0)
    sum_est = Estimates(config.oracle_budget, sum_gt, sum_estimate, 0, 0)
    avg_est = Estimates(config.oracle_budget, avg_gt, avg_estimate, 0, 0)
    count_est.log()
    sum_est.log()
    avg_est.log()
    count_est.save(output_file=config.output_file, surfix="_count")
    sum_est.save(output_file=config.output_file, surfix="_sum")
    avg_est.save(output_file=config.output_file, surfix="_avg")
