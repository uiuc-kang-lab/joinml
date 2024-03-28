from joinml.proxy.get_proxy import get_proxy_score
from joinml.dataset_loader import load_dataset
from joinml.oracle import Oracle
from joinml.config import Config
from joinml.utils import set_up_logging, normalize, get_ci_gaussian, get_ci_bootstrap_ttest, get_non_positive_ci
from joinml.estimates import Selection
from joinml.plugins.supg import supg_recall_target_importance, supg_precision_target_importance

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
    logging.info(f"ground truth count {count_gt} sum {sum_gt} avg {avg_gt}")

    # check cache for proxy rank
    proxy_score = get_proxy_score(config, dataset, is_wanderjoin=True)
    proxy_weights = normalize(proxy_score)

    for _ in range(config.internal_loop):
        run_once(config, dataset, oracle, dataset_sizes, count_gt, sum_gt, avg_gt, proxy_weights)

def run_once(config, dataset, oracle, dataset_sizes, count_gt, sum_gt, avg_gt, proxy_weights):
    sample = np.random.choice(len(proxy_weights), size=config.oracle_budget, p=proxy_weights, replace=True)
    sample_ids = np.array(np.unravel_index(sample, dataset_sizes)).T
    sampling_weights = proxy_weights[sample]
    sample_oracle_results = []
    for s, sample_id in zip(sample, sample_ids):
        if oracle.query(sample_id):
            sample_oracle_results.append(1)
        else:
            sample_oracle_results.append(0)

    sorted_ids = np.argsort(sampling_weights)
    sorted_oracle_results = np.array(sample_oracle_results)[sorted_ids]
    sorted_sampling_weights = sampling_weights[sorted_ids]
    recall_threshold = supg_recall_target_importance(config.target, sorted_oracle_results, 
                                                     sorted_sampling_weights, np.prod(dataset_sizes))
    precision_threshold = supg_precision_target_importance(config.target, sorted_oracle_results,
                                                           sorted_sampling_weights, np.prod(dataset_sizes))
    # calculate the true recall
    sample = np.where(proxy_weights > recall_threshold)[0]
    sample_ids = np.array(np.unravel_index(sample, dataset_sizes)).T
    positives = 0
    negatives = 0
    for sample_id in sample_ids:
        if oracle.query(sample_id):
            positives += 1
        else:
            negatives += 1
    recall_results = Selection(config.budget, "recall", config.target, recall, precision)
    recall_results.log()
    recall_results.save(config.output_file, surfix="_recall")
    # calculate the true precision
    sample = np.where(proxy_weights > precision_threshold)[0]
    sample_ids = np.array(np.unravel_index(sample, dataset_sizes)).T
    positives = 0
    negatives = 0
    for sample_id in sample_ids:
        if oracle.query(sample_id):
            positives += 1
        else:
            negatives += 1
    precision_results = Selection(config.budget, "precision", config.target, recall, precision)
    precision_results.log()
    precision_results.save(config.output_file, surfix="_precision")

