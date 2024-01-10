from joinml.proxy.get_proxy import get_proxy_score, get_proxy_rank
from joinml.dataset_loader import load_dataset
from joinml.oracle import Oracle
from joinml.config import Config
from joinml.utils import set_up_logging, normalize
from joinml.estimates import Estimates
from joinml.utils import get_ci_bootstrap

import os
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
    min_statistics, max_statistics = dataset.get_min_max_statistics()
    logging.debug(f"count gt: {count_gt}, sum gt: {sum_gt}, avg gt: {avg_gt}")
    logging.debug(f"min statistics: {min_statistics}, max statistics: {max_statistics}")

    # get proxy
    proxy_scores = get_proxy_score(config, dataset)
    proxy_rank = get_proxy_rank(config, dataset, proxy_scores)

    # allocate blocking and sampling
    blocking_size = int(config.oracle_budget * config.blocking_ratio)
    sampling_size = config.oracle_budget - blocking_size
    blocking_data = proxy_rank[-blocking_size:]
    sampling_data = proxy_rank[:-blocking_size]
    logging.debug(f"blocking size: {blocking_size}, sampling size: {len(sampling_data)}")

    # run sampling
    proxy_scores = proxy_scores[sampling_data]
    proxy_scores = normalize(proxy_scores, config.proxy_normalizing_style)
    sample = np.random.choice(len(sampling_data), size=sampling_size, replace=True, p=proxy_scores)
    sample_ids = sampling_data[sample]
    sample_ids = np.array(np.unravel_index(sample, dataset_sizes)).T
    sampling_results = []
    sampling_count_results = []
    sampling_avg_results = []
    for s_id, s in zip(sample_ids, sample):
        if oracle.query(s_id):
            statistics = dataset.get_statistics(s_id)
            sampling_results.append(statistics / len(sampling_data) / proxy_scores[s])
            sampling_count_results.append(1 / len(sampling_data) / proxy_scores[s])
            sampling_avg_results.append(statistics)
        else:
            sampling_results.append(0)
            sampling_count_results.append(0)
    sampling_results = np.array(sampling_results)
    sampling_count_results = np.array(sampling_count_results)

    # run blocking
    blocking_results = []
    blocking_count_results = []
    blocking_avg_results = []
    for b_id in blocking_data:
        if oracle.query(b_id):
            statistics = dataset.get_statistics(b_id)
            blocking_results.append(statistics)
            blocking_count_results.append(1)
            blocking_avg_results.append(statistics)
        else:
            blocking_results.append(0)
            blocking_count_results.append(0)

    count_estimate = np.mean(sampling_count_results) * len(sampling_data) + np.sum(blocking_count_results)
    sum_estimate = np.mean(sampling_results) * len(sampling_data) + np.sum(blocking_results)
    avg_estimate = (np.sum(sampling_avg_results) + np.sum(blocking_avg_results)) / (len(sampling_data) + len(blocking_avg_results))

    # run bootstrapping for the CI
    bootstrapping_count = []
    bootstrapping_sum = []
    bootstrapping_avg = []
    for trian in range(config.bootstrap_trials):
        # resample from the sampling data
        resample = np.random.choice(len(sampling_count_results), size=sampling_size, replace=True)
        resample_results = sampling_results[resample]
        resample_count_results = sampling_count_results[resample]
        resample_count = np.mean(resample_count_results) * len(sampling_data) + np.sum(blocking_count_results)
        resample_sum = np.mean(resample_results) * len(sampling_data) + np.sum(blocking_results)
        resample_avg = resample_sum / resample_count
        bootstrapping_count.append(resample_count)
        bootstrapping_sum.append(resample_sum)
        bootstrapping_avg.append(resample_avg)
    bootstrapping_count = np.array(bootstrapping_count)
    bootstrapping_sum = np.array(bootstrapping_sum)
    bootstrapping_avg = np.array(bootstrapping_avg)

    # calculate CI
    ci_count_lower, ci_count_upper = get_ci_bootstrap(bootstrapping_count, config.confidence_level)
    ci_sum_lower, ci_sum_upper = get_ci_bootstrap(bootstrapping_sum, config.confidence_level)
    ci_avg_lower, ci_sum_upper = get_ci_bootstrap(bootstrapping_avg, config.confidence_level)

    count_est = Estimates(config.oracle_budget, count_gt, count_estimate, ci_count_lower, ci_count_upper)
    sum_est = Estimates(config.oracle_budget, sum_gt, sum_estimate, ci_sum_lower, ci_sum_upper)
    avg_est = Estimates(config.oracle_budget, avg_gt, avg_estimate, ci_avg_lower, ci_sum_upper)

    count_est.log()
    sum_est.log()
    avg_est.log()

    count_est.save(config.output_file, "_count")
    sum_est.save(config.output_file, "_sum")
    avg_est.save(config.output_file, "_avg")
