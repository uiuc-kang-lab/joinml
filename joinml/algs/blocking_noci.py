from joinml.proxy.get_proxy import get_proxy_score, get_proxy_rank
from joinml.dataset_loader import load_dataset, JoinDataset
from joinml.oracle import Oracle
from joinml.config import Config
from joinml.utils import set_up_logging, get_cutoff_score
from joinml.estimates import Estimates

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

    # sampling_budget = config.oracle_budget // 2
    # blocking_budget = config.oracle_budget - sampling_budget
    blocking_budget = config.oracle_budget

    proxy_weights = get_proxy_score(config, dataset)
    cutoff_score = get_cutoff_score(config.dataset_name)
    logging.debug(f"cutoff_score: {cutoff_score}")

    for _ in range(config.internal_loop):
        # get a uniform sampling of the whole dataset
        # sample = np.random.choice(np.prod(dataset_sizes), size=sampling_budget)
        # positive_sample_scores = []
        # sample_ids = np.array(np.unravel_index(sample, dataset_sizes)).T
        # for s, sample_id in zip(sample, sample_ids):
        #     if oracle.query(sample_id):
        #         positive_sample_scores.append(proxy_weights[s])
        # if len(positive_sample_scores) == 0:
        #     return
        # cutoff_score = min(positive_sample_scores)

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
                
        if config.aggregator == "count":
            estimate = sample_count / blocking_budget * len(unblocked_population)
            gt = count_gt
        elif config.aggregator == "sum":
            estimate = sample_sum / blocking_budget * len(unblocked_population)
            gt = sum_gt
        else:
            estimate = sample_sum / sample_count
            gt = avg_gt

        est = Estimates(config.oracle_budget, gt, estimate, [0], [0])
        est.log()
        est.save(output_file=config.output_file, surfix=f"_{config.aggregator}")

