from joinml.proxy.get_proxy import get_proxy_score, get_proxy_rank
from joinml.dataset_loader import load_dataset, JoinDataset
from joinml.oracle import Oracle
from joinml.config import Config
from joinml.utils import set_up_logging, get_cutoff_score_unbiased, get_ci_bootstrap_ttest
from joinml.estimates import Estimates

import time
import logging
import numpy as np
import scipy.stats as stats

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

    count_gt, sum_gt, avg_gt, min_gt, max_gt = dataset.get_gt(oracle)

    blocking_budget = config.oracle_budget

    proxy_weights = get_proxy_score(config, dataset)
    cutoff_score = get_cutoff_score_unbiased(config.dataset_name)
    logging.debug(f"cutoff_score: {cutoff_score}")
    
    unblocked_population = np.argwhere(proxy_weights >= cutoff_score).reshape(-1)
    # get the population based on the score cutoff
    logging.debug(f"size of unblocked population {len(unblocked_population)}")
    
    if blocking_budget > len(unblocked_population):
        replace = True
    else:
        replace = False

    for _ in range(config.internal_loop):
        # get a uniform sampling of the unblocked dataset
        unblocked_sample = np.random.choice(unblocked_population, size=blocking_budget, replace=replace)
        # fpc = (len(unblocked_population) - blocking_budget) / (len(unblocked_population) - 1)
        # print("fpc", fpc)
        unblocked_sample_ids = np.array(np.unravel_index(unblocked_sample, dataset_sizes)).T
        sample_results = []
        sample_count_results = []
        sample_positive_results = []
        for sample_id in unblocked_sample_ids:
            if oracle.query(sample_id):
                sample_results.append(dataset.get_statistics(sample_id))
                sample_positive_results.append(dataset.get_statistics(sample_id))
                sample_count_results.append(1)
            else:
                sample_results.append(0)
                sample_count_results.append(0)
        print("done oracle")

        if config.aggregator == "count":
            gt = count_gt
            N = len(unblocked_population)
            n = len(sample_count_results)
            mean = np.mean(sample_count_results).item()
            std = np.std(sample_count_results, ddof=1)
            z = stats.norm.ppf(1 - (1 - config.confidence_level) / 2)
            fpc = (N-n) / (N-1) if not replace else 1
            mean_lb = mean - z * std / np.sqrt(n) * fpc
            mean_ub = mean + z * std / np.sqrt(n) * fpc
            lb = mean_lb * N
            ub = mean_ub * N
            estimate = mean * N
        elif config.aggregator == "sum":
            gt = sum_gt
            N = len(unblocked_population)
            n = len(sample_count_results)
            mean = np.mean(sample_results).item()
            std = np.std(sample_results, ddof=1)
            z = stats.norm.ppf(1 - (1 - config.confidence_level) / 2)
            fpc = (N-n) / (N-1) if not replace else 1
            mean_lb = mean - z * std / np.sqrt(n) * fpc
            mean_ub = mean + z * std / np.sqrt(n) * fpc
            lb = mean_lb * N
            ub = mean_ub * N
            estimate = mean * N
        else:
            gt = avg_gt
            N = len(oracle.oracle_labels)
            n = len(sample_positive_results)
            fpc = (N-n) / (N-1) if not replace else 1
            mean = np.mean(sample_positive_results).item()
            std = np.std(sample_positive_results, ddof=1)
            z = stats.norm.ppf(1 - (1 - config.confidence_level) / 2)
            mean_lb = mean - z * std / np.sqrt(n) * fpc
            mean_ub = mean + z * std / np.sqrt(n) * fpc
            lb = mean_lb
            ub = mean_ub
            estimate = mean
        
        # config.bootstrap_trials = 1000
        # print("running resampling ...")
        # start = time.time()
        # lb, ub, estimate = get_straight_sampling_CI(
        #     config, len(unblocked_population), blocking_budget,
        #     sample_results, sample_count_results, sample_positive_results)
        # print(f"Resampling time: {time.time() - start}")

        est = Estimates(config.oracle_budget, gt, estimate, [lb], [ub])
        est.log()
        est.save(output_file=config.output_file, surfix=f"_{config.aggregator}")



def get_straight_sampling_CI(config, population_size, blocking_budget, sample_results, sample_count_results, sample_positive_results):
    sample_count = np.sum(sample_count_results)
    sample_sum = np.sum(sample_results)
    sample_mean = np.mean(sample_positive_results)
    
    if config.aggregator == "count":
        estimate = sample_count / blocking_budget * population_size
    elif config.aggregator == "sum":
        estimate = sample_sum / blocking_budget * population_size
    else:
        estimate = sample_mean

    sample_results = np.array(sample_results)
    sample_count_results = np.array(sample_count_results)
    sample_positive_results = np.array(sample_positive_results)
    count_var = np.var(sample_count_results).item() * population_size
    sum_var = np.var(sample_results).item() * population_size
    avg_var = np.var(sample_positive_results).item()
    
    if config.aggregator == "count":
        variance = count_var
    elif config.aggregator == "sum":
        variance = sum_var
    else:
        variance = avg_var

    ts = []
    for trial in range(config.bootstrap_trials):
        # resample
        resample = np.random.choice(len(sample_results), size=len(sample_results), replace=True)
        resample_results = sample_results[resample]
        resample_count_results = sample_count_results[resample]
        resample_positive_results = sample_results[sample_count_results[resample] == 1]
        sum_estimate_bootstrap = np.mean(resample_results).item() * population_size
        count_estimate_bootstrap = np.mean(resample_count_results).item() * population_size
        count_var_bootstrap = np.var(resample_count_results).item() * population_size
        sum_var_bootstrap = np.var(resample_results).item() * population_size
        if count_estimate_bootstrap == 0:
            avg_estimate_bootstrap = np.nan
            avg_var_bootstrap = np.nan
        else:
            avg_estimate_bootstrap = sum_estimate_bootstrap / count_estimate_bootstrap
            avg_var_bootstrap = np.var(resample_positive_results).item()
        if config.aggregator == "count":
            ts.append((count_estimate_bootstrap - estimate) / np.sqrt(count_var_bootstrap))
        elif config.aggregator == "sum":
            ts.append((sum_estimate_bootstrap - estimate) / np.sqrt(sum_var_bootstrap))
        else:
            ts.append((avg_estimate_bootstrap - estimate) / np.sqrt(avg_var_bootstrap))

        # calculate CI
    lb, ub = get_ci_bootstrap_ttest(estimate, ts, variance, config.confidence_level)
        
    return lb, ub, estimate
