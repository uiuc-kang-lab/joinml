from joinml.proxy.get_proxy import get_proxy_score
from joinml.dataset_loader import load_dataset
from joinml.oracle import Oracle
from joinml.config import Config
from joinml.utils import set_up_logging, normalize, get_ci_gaussian, get_ci_bootstrap
from joinml.estimates import Estimates

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
        proxy_weights = normalize(proxy_score, style="sqrt")

        # weighted sample
        p = np.array(proxy_weights).astype(np.float64)
        p /= np.sum(p)
        sample = np.random.choice(len(proxy_weights), size=config.oracle_budget, p=p, replace=True)
        sample_ids = np.array(np.unravel_index(sample, dataset_sizes)).T
        sample_results = []
        sample_count_results = []
        for s, sample_id in zip(sample, sample_ids):
            if oracle.query(sample_id):
                stats = dataset.get_statistics(sample_id)
                sample_results.append(stats / len(proxy_weights) / proxy_weights[s])
                sample_count_results.append(1 / len(proxy_weights) / proxy_weights[s])
            else:
                sample_results.append(0)
                sample_count_results.append(0)

    elif config.task == "uniform":
        sample = np.random.choice(np.prod(dataset_sizes), size=config.oracle_budget)
        sample_ids = np.array(np.unravel_index(sample, dataset_sizes)).T
        sample_results = []
        sample_count_results = []
        for sample_id in sample_ids:
            if oracle.query(sample_id):
                sample_results.append(dataset.get_statistics(sample_id))
                sample_count_results.append(1)
            else:
                sample_results.append(0)
                sample_count_results.append(0)
    
    else:
        raise NotImplementedError(f"Task {config.task} not implemented for straight sampling.")
        
    sample_results = np.array(sample_results)
    sample_count_results = np.array(sample_count_results)
    bootstrapping_results = {
        "count": [],
        "sum": []
    }
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
    else:
        m_sums = []
        m_counts = []
        avgs = []
        for trial in range(config.bootstrap_trials):
            # resample
            resample = np.random.choice(len(sample_results), size=len(sample_results), replace=True)
            resample_results = sample_results[resample]
            resample_count_results = sample_count_results[resample]
            m_sum = np.mean(resample_results).item()
            m_count = np.mean(resample_count_results).item()
            avg = m_sum / m_count if m_count > 0 else 0
            m_sums.append(m_sum)
            m_counts.append(m_count)
            avgs.append(avg)
            logging.debug(f"trial {trial} avg {avg} sum {m_sum} count {m_count}")
        m_sums = np.array(m_sums)
        m_counts = np.array(m_counts)
        avg_results = np.array(avgs)
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