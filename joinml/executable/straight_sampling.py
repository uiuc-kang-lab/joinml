from joinml.proxy.get_proxy import get_proxy_score
from joinml.dataset_loader import load_dataset
from joinml.oracle import Oracle
from joinml.config import Config
from joinml.utils import set_up_logging, normalize, get_ci_gaussian, get_ci_bootstrap
from joinml.estimates import Estimates
from joinml.executable.joinml_dep2 import get_non_positive_ci

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
    logging.info(f"ground truth count {count_gt} sum {sum_gt} avg {avg_gt}")

    if config.task == "is":
        # check cache for proxy rank
        proxy_score = get_proxy_score(config, dataset)
        proxy_weights = normalize(proxy_score)
    else:
        proxy_weights = None

    for exp_id in range(config.internal_loop):
        run_once(config, dataset, oracle, dataset_sizes, count_gt, sum_gt, avg_gt, proxy_weights)

def run_once(config, dataset, oracle, dataset_sizes, count_gt, sum_gt, avg_gt, proxy_weights):
    if config.task == "is":
        sample = np.random.choice(len(proxy_weights), size=config.oracle_budget, p=proxy_weights, replace=True)
        sample_ids = np.array(np.unravel_index(sample, dataset_sizes)).T
        sample_results = []
        sample_count_results = []
        sample_positive_results = []
        for s, sample_id in zip(sample, sample_ids):
            if oracle.query(sample_id):
                stats = dataset.get_statistics(sample_id)
                sample_results.append(stats / len(proxy_weights) / proxy_weights[s])
                sample_count_results.append(1 / len(proxy_weights) / proxy_weights[s])
                sample_positive_results.append(stats)
            else:
                sample_results.append(0)
                sample_count_results.append(0)

    elif config.task == "uniform":
        sample = np.random.choice(np.prod(dataset_sizes), size=config.oracle_budget)
        sample_ids = np.array(np.unravel_index(sample, dataset_sizes)).T
        sample_results = []
        sample_count_results = []
        sample_positive_results = []
        for sample_id in sample_ids:
            if oracle.query(sample_id):
                sample_results.append(dataset.get_statistics(sample_id))
                sample_positive_results.append(dataset.get_statistics(sample_id))
                sample_count_results.append(1)
            else:
                sample_results.append(0)
                sample_count_results.append(0)
    
    else:
        raise NotImplementedError(f"Task {config.task} not implemented for straight sampling.")

    if len(sample_positive_results) < 2:
        min_statistics, max_statistics = dataset.get_min_max_statistics()
        count_lower_bound, count_upper_bound, sum_lower_bound, sum_upper_bound = \
            get_non_positive_ci(max_statistics=max_statistics, confidence_level=config.confidence_level,
                                n_positive_population_size=np.prod(dataset_sizes).item(), n_positive_sample_size=config.oracle_budget)
        avg_lower_bound = min_statistics
        avg_upper_bound = max_statistics
        count_estimate = 0
        sum_estimate = 0
        avg_estimate = np.nan
        count_est = Estimates(config.oracle_budget, count_gt, count_estimate, count_lower_bound, count_upper_bound)
        sum_est = Estimates(config.oracle_budget, sum_gt, sum_estimate, sum_lower_bound, sum_upper_bound)
        avg_est = Estimates(config.oracle_budget, avg_gt, avg_estimate, avg_lower_bound, avg_upper_bound)
        count_est.log()
        sum_est.log()
        avg_est.log()
        count_est.save(config.output_file, "_count")
        sum_est.save(config.output_file, "_sum")
        avg_est.save(config.output_file, "_avg")
    else:
        get_straight_sampling_CI(config, dataset_sizes, count_gt, sum_gt, avg_gt, sample_results, sample_count_results, sample_positive_results)

def get_straight_sampling_CI(config, dataset_sizes, count_gt, sum_gt, avg_gt, sample_results, sample_count_results, sample_positive_results):
    sample_results = np.array(sample_results)
    sample_count_results = np.array(sample_count_results)
    sample_positive_results = np.array(sample_positive_results)
    if config.bootstrap_trials == 0:
        # calculate gaussian confidence interval for sum
        ci_lower, ci_upper = get_ci_gaussian(sample_results, config.confidence_level)
        m = np.mean(sample_results).item()
        sum_result = m * np.prod(dataset_sizes, dtype=np.float32).item()
        ci_upper *= np.prod(dataset_sizes)
        ci_lower *= np.prod(dataset_sizes)
        sum_est = Estimates(config.oracle_budget, sum_gt, sum_result, ci_lower, ci_upper)
        sum_est.log()
        sum_est.save(config.output_file, surfix="_sum")
        # calculate gaussian confidence interval for count
        ci_lower, ci_upper = get_ci_gaussian(sample_count_results, config.confidence_level)
        m = np.mean(sample_count_results).item()
        count_result = m * np.prod(dataset_sizes, dtype=np.float32).item()
        ci_upper *= np.prod(dataset_sizes)
        ci_lower *= np.prod(dataset_sizes)
        count_est = Estimates(config.oracle_budget, count_gt, count_result, ci_lower, ci_upper)
        count_est.log()
        count_est.save(config.output_file, surfix="_count")
        # calculate gaussian confidence interval for avg
        ci_lower, ci_upper = get_ci_gaussian(sample_positive_results, config.confidence_level)
        avg_result = np.mean(sample_positive_results).item()
        avg_est = Estimates(config.oracle_budget, avg_gt, avg_result, ci_lower, ci_upper)
        avg_est.log()
        avg_est.save(config.output_file, surfix="_avg")
    else:
        m_sums = []
        m_counts = []
        m_avgs = []
        for trial in range(config.bootstrap_trials):
            # resample
            resample = np.random.choice(len(sample_results), size=len(sample_results), replace=True)
            resample_results = sample_results[resample]
            resample_count_results = sample_count_results[resample]
            positive_resample = np.random.choice(len(sample_positive_results), size=len(sample_positive_results), replace=True)
            resample_positive_results = sample_positive_results[positive_resample]
            m_sum = np.mean(resample_results).item()
            m_count = np.mean(resample_count_results).item()
            m_avg = np.mean(resample_positive_results).item()
            m_sums.append(m_sum)
            m_counts.append(m_count)
            m_avgs.append(m_avg)
            logging.debug(f"trial {trial} avg {m_avg} sum {m_sum} count {m_count}")
        m_sums = np.array(m_sums)
        m_counts = np.array(m_counts)
        avg_results = np.array(m_avgs)
        sum_results = m_sums * np.prod(dataset_sizes)
        count_results = m_counts * np.prod(dataset_sizes)
        
        # statistics for avg
        avg_estimates = np.mean(avg_results).item()
        avg_lower, avg_upper = get_ci_bootstrap(avg_results, confidence_level=config.confidence_level)
        avg_est = Estimates(config.oracle_budget, avg_gt, avg_estimates, avg_lower, avg_upper)
        avg_est.log()
        avg_est.save(config.output_file, surfix="_avg")
        # statistics for sum
        sum_estimates = np.mean(sum_results).item()
        sum_lower, sum_upper = get_ci_bootstrap(sum_results, confidence_level=config.confidence_level)
        sum_est = Estimates(config.oracle_budget, sum_gt, sum_estimates, sum_lower, sum_upper)
        sum_est.log()
        sum_est.save(config.output_file, surfix="_sum")
        # statistics for count
        count_estimates = np.mean(count_results).item()
        count_lower, count_upper = get_ci_bootstrap(count_results, confidence_level=config.confidence_level)
        count_est = Estimates(config.oracle_budget, count_gt, count_estimates, count_lower, count_upper)
        count_est.log()
        count_est.save(config.output_file, surfix="_count")