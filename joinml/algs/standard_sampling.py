from joinml.proxy.get_proxy import get_proxy_score
from joinml.dataset_loader import load_dataset
from joinml.oracle import Oracle
from joinml.config import Config
from joinml.utils import set_up_logging, normalize, get_ci_gaussian, get_ci_bootstrap_ttest, get_non_positive_ci
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
    logging.info(f"ground truth count {count_gt} sum {sum_gt} avg {avg_gt}")

    if config.task == "importance":
        # check cache for proxy rank
        proxy_score = get_proxy_score(config, dataset)
        proxy_weights = normalize(proxy_score)
    else:
        proxy_weights = None

    for _ in range(config.internal_loop):
        run_once(config, dataset, oracle, dataset_sizes, count_gt, sum_gt, avg_gt, proxy_weights)

def run_once(config, dataset, oracle, dataset_sizes, count_gt, sum_gt, avg_gt, proxy_weights):
    if config.task == "importance":
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
        count_est = Estimates(config.oracle_budget, count_gt, count_estimate, [count_lower_bound], [count_upper_bound])
        sum_est = Estimates(config.oracle_budget, sum_gt, sum_estimate, [sum_lower_bound], [sum_upper_bound])
        avg_est = Estimates(config.oracle_budget, avg_gt, avg_estimate, [avg_lower_bound], [avg_upper_bound])
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
    count_estimation = np.mean(sample_count_results).item() * np.prod(dataset_sizes, dtype=np.float32).item()
    sum_estimation = np.mean(sample_results).item() * np.prod(dataset_sizes, dtype=np.float32).item()
    count_var = np.var(sample_count_results).item() * np.prod(dataset_sizes, dtype=np.float32).item()
    sum_var = np.var(sample_results).item() * np.prod(dataset_sizes, dtype=np.float32).item()
    if count_estimation == 0:
        avg_estimation = np.nan
        avg_var = np.nan
    else:
        avg_estimation = sum_estimation / count_estimation
        avg_estimation -= get_avg_correction(np.prod(dataset_sizes, dtype=np.float32).item(), 
                                            config.oracle_budget, count_estimation, avg_estimation, count_var)
        avg_var = get_avg_var(config.oracle_budget, count_var, count_estimation, sum_var, sum_estimation)
    if config.aggregator == "count":
        estimation = count_estimation
        variance = count_var
        gt = count_gt
    elif config.aggregator == "sum":
        estimation = sum_estimation
        variance = sum_var
        gt = sum_gt
    else:
        estimation = avg_estimation
        variance = avg_var
        gt = avg_gt

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
        ts = []
        for trial in range(config.bootstrap_trials):
            # resample
            resample = np.random.choice(len(sample_results), size=len(sample_results), replace=True)
            resample_results = sample_results[resample]
            resample_count_results = sample_count_results[resample]
            sum_estimate_bootstrap = np.mean(resample_results).item() * np.prod(dataset_sizes, dtype=np.float32).item()
            count_estimate_bootstrap = np.mean(resample_count_results).item() * np.prod(dataset_sizes, dtype=np.float32).item()
            count_var_bootstrap = np.var(resample_count_results).item() * np.prod(dataset_sizes, dtype=np.float32).item()
            sum_var_bootstrap = np.var(resample_results).item() * np.prod(dataset_sizes, dtype=np.float32).item()
            if count_estimate_bootstrap == 0:
                avg_estimate_bootstrap = np.nan
                avg_var_bootstrap = np.nan
            else:
                avg_estimate_bootstrap = sum_estimate_bootstrap / count_estimate_bootstrap
                avg_estimate_bootstrap -= get_avg_correction(np.prod(dataset_sizes, dtype=np.float32).item(), 
                                            config.oracle_budget, count_estimate_bootstrap, avg_estimate_bootstrap, count_var_bootstrap)
                avg_var_bootstrap = get_avg_var(config.oracle_budget, count_var_bootstrap, count_estimate_bootstrap, sum_var_bootstrap, sum_estimate_bootstrap)
            if config.aggregator == "count":
                ts.append((count_estimate_bootstrap - count_estimation) / np.sqrt(count_var_bootstrap))
            elif config.aggregator == "sum":
                ts.append((sum_estimate_bootstrap - sum_estimation) / np.sqrt(sum_var_bootstrap))
            else:
                ts.append((avg_estimate_bootstrap - avg_estimation) / np.sqrt(avg_var_bootstrap))

        # calculate CI
        lb, ub = get_ci_bootstrap_ttest(estimation, ts, variance, config.confidence_level)

        est = Estimates(config.oracle_budget, gt, estimation, lb, ub)
        est.log()
        est.save(config.output_file, f"_{config.aggregator}")

def get_avg_var(sample_size, count_var, count_mean, sum_var, sum_mean):
    return 1. / sample_size * (sum_var / count_mean**2 + count_var * sum_mean**2 / count_mean**4)

def get_avg_correction(population_size, sample_size, count_mean, avg_mean, count_var):
    return (population_size-sample_size)/(population_size-1) * avg_mean * count_var / sample_size / count_mean**2
