from joinml.proxy.get_proxy import get_proxy_score, get_proxy_rank
from joinml.scalable_dataset_loader import ScalableJoinDataset, load_dataset
from joinml.oracle import Oracle
from joinml.config import Config
from joinml.utils import set_up_logging, get_cutoff_score, get_ci_gaussian, get_ci_bootstrap_ttest
from joinml.estimates import Estimates

import time
import logging
import numpy as np
import scipy.stats as stats

def run(config: Config):
    set_up_logging(config.log_path, config.log_level)

    # log config
    logging.info(config)

    # dataset
    dataset = load_dataset(config)

    # setup dataset
    dataset_sizes = dataset.get_sizes()

    count_gt = dataset.get_gt()

    blocking_budget = config.oracle_budget

    for _ in range(config.internal_loop):
        # get a uniform sampling of the unblocked dataset
        # unblocked_sample = np.random.choice(unblocked_population, size=blocking_budget, replace=replace)
        # fpc = (len(unblocked_population) - blocking_budget) / (len(unblocked_population) - 1)
        # print("fpc", fpc)
        sample_results = dataset.wander_join_blocking(blocking_budget)

        gt = count_gt
        N = np.prod(dataset_sizes)
        n = len(sample_results)
        mean = np.mean(sample_results).item()
        std = np.std(sample_results, ddof=1)
        z = stats.norm.ppf(1 - (1 - config.confidence_level) / 2)
        fpc = (N-n) / (N-1)
        mean_lb = mean - z * std / np.sqrt(n) * fpc
        mean_ub = mean + z * std / np.sqrt(n) * fpc
        lb = mean_lb * N
        ub = mean_ub * N
        estimate = mean * N

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
