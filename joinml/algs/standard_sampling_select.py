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
    proxy_score = get_proxy_score(config, dataset)
    proxy_weights = normalize(proxy_score)

    for _ in range(config.internal_loop):
        run_once(config, dataset, oracle, dataset_sizes, count_gt, sum_gt, avg_gt, proxy_weights)

def run_once(config, dataset, oracle, dataset_sizes, count_gt, sum_gt, avg_gt, proxy_weights):
    if config.task == "importance":
        sample = np.random.choice(len(proxy_weights), size=config.oracle_budget, p=proxy_weights, replace=True)
        sample_weights = proxy_weights[sample]
        sample_ids = np.array(np.unravel_index(sample, dataset_sizes)).T
        sample_oracle_results = []
        for s, sample_id in zip(sample, sample_ids):
            if oracle.query(sample_id):
                sample_oracle_results.append(1)
            else:
                sample_oracle_results.append(0)

    elif config.task == "uniform":
        sample = np.random.choice(np.prod(dataset_sizes), size=config.oracle_budget)
        sample_ids = np.array(np.unravel_index(sample, dataset_sizes)).T
        sample_oracle_results = []
        sample_weights = proxy_weights[sample]
        for sample_id in sample_ids:
            if oracle.query(sample_id):
                sample_oracle_results.append(1)
            else:
                sample_oracle_results.append(0)
    else:
        raise NotImplementedError(f"Task {config.task} not implemented for straight sampling.")

    sorted_ids = np.argsort(sample_weights)
    sorted_weights = sample_weights[sorted_ids]
    sorted_oracle_results = np.array(sample_oracle_results)[sorted_ids]

    if config.task == "importance":
        recall_threshold = supg_recall_target_importance(config.target, sorted_oracle_results, 
                                                         sorted_weights, np.prod(dataset_sizes))
        precision_threshold = supg_precision_target_importance(config.target, sorted_oracle_results,
                                                               sorted_weights, np.prod(dataset_sizes))
    elif config.task == "uniform":
        recall_threshold = supg_recall_target_uniform(config.target, sorted_oracle_results, 
                                                    sorted_weights, np.prod(dataset_sizes))
        precision_threshold = supg_precision_target_uniform(config.target, sorted_oracle_results, 
                                                            sorted_weights, np.prod(dataset_sizes))
    else:
        raise NotImplementedError(f"Task {config.task} not implemented for straight sampling.")

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
    recall = positives / count_gt
    precision = positives / (positives + negatives)
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
    recall = positives / count_gt
    precision = positives / (positives + negatives)
    precision_results = Selection(config.budget, "precision", config.target, recall, precision)
    precision_results.log()
    precision_results.save(config.output_file, surfix="_precision")