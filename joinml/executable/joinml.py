from joinml.proxy.get_proxy import get_proxy_score, get_proxy_rank
from joinml.dataset_loader import load_dataset, JoinDataset
from joinml.oracle import Oracle
from joinml.config import Config
from joinml.utils import set_up_logging, normalize
from joinml.estimates import Estimates
from joinml.utils import get_ci_bootstrap, weighted_sample_pd, calculate_ci_correction

import os
import logging
import numpy as np
from typing import Tuple, List
from scipy import stats

def empirical_best_blocking_allocation(strata_population: List[int], strata_sample_sizes: List[int],
                                       strata_sample_variance: List[float], objective: str="joinml-ci") -> Tuple[List[int], List[int]]:
    # blocking for all the non-positive data
    strata_to_decide = []
    for i in range(len(strata_population)):
        if not np.isnan(strata_sample_variance[i]):
            strata_to_decide.append(i)

    # calculate the total variance of the sub population
    total_variances = []
    for i in range(len(strata_to_decide)):
        variance = 0
        total_population = sum([strata_population[j] for j in strata_to_decide[:i+1]])
        for j in strata_to_decide[:i+1]:
            variance += (strata_population[j] / total_population)**2 * strata_sample_variance[j]
        total_variances.append(variance)
    
    logging.debug(f"total variances: {total_variances}")
    
    # calculate the utility based on the objective
    utility = []
    for i in range(len(strata_to_decide)):
        utility.append(total_variances[i])
    logging.debug(f"utility: {utility}")

    min_utility = np.min(utility)
    optimal_allocation = 1 + [i for i in range(len(utility)) if utility[i] == min_utility][-1]

    logging.debug(f"optimal allocation {optimal_allocation} utility {min_utility}")
    strata_for_sampling = strata_to_decide[:optimal_allocation]
    strata_for_blocking = []
    for i in range(len(strata_population)):
        if i not in strata_for_sampling:
            strata_for_blocking.append(i)
    return strata_for_sampling, strata_for_blocking

def run_once(config: Config, dataset, oracle, dataset_sizes, count_gt, sum_gt, avg_gt, proxy_scores, strata, strata_population, strata_sample_sizes):

    strata_sample_count_results = []    # O(x)
    strata_sample_sum_results = []      # f(x) * O(x)
    strata_count_mean_vars = []         # variance of O(x)
    strata_sum_mean_vars = []           # variance of f(x) * O(x)
    # run sampling with replacement for each stratum
    for i, (stratum_begin, stratum_end) in enumerate(strata):
        stratum_proxy_scores = proxy_scores[strata_population[i]]
        stratum_proxy_weights = normalize(stratum_proxy_scores)
        stratum_sample = weighted_sample_pd(stratum_proxy_scores, strata_sample_sizes[i], replace=True)

        # run oracle for statistics
        stratum_sample_ids = strata_population[i][stratum_sample]
        stratum_sample_ids = np.array(np.unravel_index(stratum_sample_ids, dataset_sizes)).T
        stratum_sample_sum_results = []
        stratum_sample_count_results = []
        for stratum_s_id, stratum_s in zip(stratum_sample_ids, stratum_sample):
            if oracle.query(stratum_s_id):
                statistics = dataset.get_statistics(stratum_s_id)
                sample_count_result = 1 / len(stratum_proxy_weights) / stratum_proxy_weights[stratum_s]
                sample_sum_result = statistics / len(stratum_proxy_weights) / stratum_proxy_weights[stratum_s]
                stratum_sample_count_results.append(sample_count_result)
                stratum_sample_sum_results.append(sample_sum_result)
            else:
                stratum_sample_count_results.append(0)
                stratum_sample_sum_results.append(0)

        strata_sample_count_results.append(stratum_sample_count_results)
        strata_sample_sum_results.append(stratum_sample_sum_results)

        # calculate variance of mean
        stratum_count_mean_var = np.var(stratum_sample_count_results, ddof=1).item() / len(stratum_sample_count_results)
        stratum_sum_mean_var = np.var(stratum_sample_sum_results, ddof=1).item() / len(stratum_sample_sum_results)
        strata_count_mean_vars.append(stratum_count_mean_var)
        strata_sum_mean_vars.append(stratum_sum_mean_var)

    logging.debug(f"strata count vars: {strata_count_mean_vars} sum vars: {strata_sum_mean_vars}")
    logging.debug(f"strata count means: {[np.mean(stratum_sample_count_result) for stratum_sample_count_result in strata_sample_count_results]} sum means: {[np.mean(stratum_sample_sum_result) for stratum_sample_sum_result in strata_sample_sum_results]}")

    strata_population_size = [len(stratum_population) for stratum_population in strata_population]

    if sum(strata_sample_count_results[0]) == 0:
        sampling_strata = [0]
        blocking_strata = [i for i in range(1, len(strata_population))]
    elif config.dataset_name == "VeRi":
        sampling_strata = [i for i in range(len(strata_population))]
        blocking_strata = []
    else:
        sampling_strata, blocking_strata = empirical_best_blocking_allocation(strata_population_size, strata_sample_sizes, strata_count_mean_vars)

    logging.debug(f"{config.aggregator} sampling strata: {sampling_strata}, {config.aggregator} blocking strata: {blocking_strata}")

    # allocate extra sample size
    reserved_blocking_cost = np.sum([len(stratum_population) for stratum_population in strata_population[1:]])
    total_extra_sample_size = reserved_blocking_cost - np.sum([strata_population_size[i] for i in blocking_strata])
    logging.debug(f"extra sample size after blocking decision, {config.aggregator}: {total_extra_sample_size}")

    # sample size allocation
    original_count_sample_size = np.array([strata_sample_sizes[i] for i in sampling_strata])
    extra_sample_size = np.array(total_extra_sample_size * original_count_sample_size / np.sum(original_count_sample_size), dtype=int)
    logging.debug(f"extra sample size after blocking decision per sampling strata, {config.aggregator}: {extra_sample_size}")

    # sample more
    if np.sum(extra_sample_size) > 0:
        for strata_id, extra_size in zip(sampling_strata, extra_sample_size):
            if extra_size == 0:
                continue
            stratum_proxy_scores = proxy_scores[strata_population[strata_id]]
            stratum_proxy_weights = normalize(stratum_proxy_scores)
            extra_sample = weighted_sample_pd(stratum_proxy_scores, extra_size, replace=True)
            extra_sample_ids = strata_population[strata_id][extra_sample]
            extra_sample_ids = np.array(np.unravel_index(extra_sample_ids, dataset_sizes)).T
            for extra_count_s_id, extra_count_s in zip(extra_sample_ids, extra_sample):
                if oracle.query(extra_count_s_id):
                    statistics = dataset.get_statistics(extra_count_s_id)
                    sample_count_result = 1 / len(stratum_proxy_weights) / stratum_proxy_weights[extra_count_s]
                    sample_sum_result = statistics / len(stratum_proxy_weights) / stratum_proxy_weights[extra_count_s]
                    strata_sample_count_results[strata_id].append(sample_count_result)
                    strata_sample_sum_results[strata_id].append(sample_sum_result) 
                else:
                    strata_sample_count_results[strata_id].append(0)
                    strata_sample_sum_results[strata_id].append(0)
        logging.debug("after sampling more, strata count means: {} sum means: {}".format([np.mean(stratum_sample_count_result) for stratum_sample_count_result in strata_sample_count_results], 
                                                                                        [np.mean(stratum_sample_sum_result) for stratum_sample_sum_result in strata_sample_sum_results]))
        logging.debug("after sampling more, strata count vars: {} sum vars: {}".format([np.var(stratum_sample_count_result, ddof=1).item() / len(stratum_sample_count_result) for stratum_sample_count_result in strata_sample_count_results],
                                                                                    [np.var(stratum_sample_sum_result, ddof=1).item() / len(stratum_sample_sum_result) for stratum_sample_sum_result in strata_sample_sum_results]))

    # run sampling with replacement for blocking strata
    if len(blocking_strata) > 0:
        for i in blocking_strata:
            blocking_population = strata_population[i]
            blocking_sample = np.random.choice(blocking_population, len(blocking_population), replace=True)
            blocking_sample_ids = np.array(np.unravel_index(blocking_sample, dataset_sizes)).T
            for blocking_s_id, blocking_s in zip(blocking_sample_ids, blocking_sample):
                if oracle.query(blocking_s_id):
                    statistics = dataset.get_statistics(blocking_s_id)
                    sample_count_result = 1
                    sample_sum_result = statistics
                    strata_sample_count_results[i].append(sample_count_result)
                    strata_sample_sum_results[i].append(sample_sum_result)
                else:
                    strata_sample_count_results[i].append(0)
                    strata_sample_sum_results[i].append(0)
        logging.debug("after blocking more, strata count means: {} sum means: {}".format([np.mean(stratum_sample_count_result) for stratum_sample_count_result in strata_sample_count_results],
                                                                                         [np.mean(stratum_sample_sum_result) for stratum_sample_sum_result in strata_sample_sum_results]))
        logging.debug("after blocking more, strata count vars: {} sum vars: {}".format([np.var(stratum_sample_count_result, ddof=1).item() / len(stratum_sample_count_result) for stratum_sample_count_result in strata_sample_count_results],
                                                                                       [np.var(stratum_sample_sum_result, ddof=1).item() / len(stratum_sample_sum_result) for stratum_sample_sum_result in strata_sample_sum_results]))

    # vectorize the sample results
    strata_sample_count_results = [np.array(stratum_sample_count_result) for stratum_sample_count_result in strata_sample_count_results]
    strata_sample_sum_results = [np.array(stratum_sample_sum_result) for stratum_sample_sum_result in strata_sample_sum_results]

    # debug info
    logging.debug(f"total sample size {[len(stratum_sample_result) for stratum_sample_result in strata_sample_count_results]}")
    logging.debug(f"strata gts: {get_gt_strata(config, strata_population, dataset_sizes, dataset, oracle)}")

    # get the estimation
    count_estimation, sum_estimation, avg_estimation = stats_func(strata_population, strata_sample_count_results, strata_sample_sum_results)
    count_var = get_stratified_var(strata_population, strata_sample_count_results)
    sum_var = get_stratified_var(strata_population, strata_sample_sum_results)
    sample_size = sum([len(stratum_sample_result) for stratum_sample_result in strata_sample_count_results])
    population_size = sum([len(stratum_population) for stratum_population in strata_population])
    avg_var = get_avg_var(sample_size, count_var/population_size**2, count_estimation/population_size, sum_var/population_size**2, sum_estimation/population_size)
    logging.debug(f"count estimation {count_estimation} sum estimation {sum_estimation} avg estimation {avg_estimation}")
    logging.debug(f"count var {count_var} sum var {sum_var} avg var {avg_var}")

    # correct bootstrapping ci
    all_count_sample = np.concatenate(strata_sample_count_results)
    all_sum_sample = np.concatenate(strata_sample_sum_results)
    count_ci_correction = calculate_ci_correction(all_count_sample, population_size)
    sum_ci_correction = calculate_ci_correction(all_sum_sample, population_size)
    logging.debug(f"count ci correction {count_ci_correction} sum ci correction {sum_ci_correction}")

    # run bootstrapping
    count_bootstraps = []
    sum_bootstraps = []
    avg_bootstraps = []
    count_ts = []
    sum_ts = []
    avg_ts = []
    for i in range(config.bootstrap_trials):
        # sample with replacement
        strata_sample_count_results_bootstrap = []
        strata_sample_sum_results_bootstrap = []
        for i, stratum_sample_count_result in enumerate(strata_sample_count_results):
            stratum_resample = np.random.choice(len(stratum_sample_count_result), len(stratum_sample_count_result), replace=True)
            strata_sample_count_results_bootstrap.append(strata_sample_count_results[i][stratum_resample])
            strata_sample_sum_results_bootstrap.append(strata_sample_sum_results[i][stratum_resample])
        count_estimation_bootstrap, sum_estimation_bootstrap, avg_estimation_bootstrap = stats_func(strata_population, strata_sample_count_results_bootstrap, strata_sample_sum_results_bootstrap)
        count_bootstraps.append(count_estimation_bootstrap)
        sum_bootstraps.append(sum_estimation_bootstrap)
        avg_bootstraps.append(avg_estimation_bootstrap)
        count_var_bootstrap = get_stratified_var(strata_population, strata_sample_count_results_bootstrap)
        sum_var_bootstrap = get_stratified_var(strata_population, strata_sample_sum_results_bootstrap)
        avg_var_bootstrap = get_avg_var(sample_size, count_var_bootstrap, count_estimation_bootstrap, sum_var_bootstrap, sum_estimation_bootstrap)
        count_ts.append((count_estimation_bootstrap - count_estimation) / np.sqrt(count_var_bootstrap))
        sum_ts.append((sum_estimation_bootstrap - sum_estimation) / np.sqrt(sum_var_bootstrap))
        avg_ts.append((avg_estimation_bootstrap - avg_estimation) / np.sqrt(avg_var_bootstrap))

    # get confidence interval
    log_ci(config, count_gt, sum_gt, avg_gt, count_estimation, sum_estimation, avg_estimation, 
           count_var, sum_var, avg_var, count_bootstraps, sum_bootstraps, avg_bootstraps, 
           count_ts, sum_ts, avg_ts)

def log_ci(config, count_gt, sum_gt, avg_gt, count_estimation, sum_estimation, avg_estimation, count_var, sum_var, avg_var, count_bootstraps, sum_bootstraps, avg_bootstraps, count_ts, sum_ts, avg_ts):
    p_lbs, p_ubs, e_lbs, e_ubs, t_lbs, t_ubs = \
        get_ci_bootstrap(count_bootstraps, count_estimation, count_ts, count_var, confidence_levels=[0.95, 0.96, 0.97, 0.98, 0.99])
    est = Estimates(config.oracle_budget, count_gt, count_estimation, p_lbs, p_ubs)
    est.log()
    est.save(config.output_file, "_count_percentile")
    est = Estimates(config.oracle_budget, count_gt, count_estimation, e_lbs, e_ubs)
    est.log()
    est.save(config.output_file, "_count_empirical")
    est = Estimates(config.oracle_budget, count_gt, count_estimation, t_lbs, t_ubs)
    est.log()
    est.save(config.output_file, "_count_ttest")

    p_lbs, p_ubs, e_lbs, e_ubs, t_lbs, t_ubs = \
        get_ci_bootstrap(sum_bootstraps, sum_estimation, sum_ts, sum_var, confidence_levels=[0.95, 0.96, 0.97, 0.98, 0.99])
    est = Estimates(config.oracle_budget, sum_gt, sum_estimation, p_lbs, p_ubs)
    est.log()
    est.save(config.output_file, "_sum_percentile")
    est = Estimates(config.oracle_budget, sum_gt, sum_estimation, e_lbs, e_ubs)
    est.log()
    est.save(config.output_file, "_sum_empirical")
    est = Estimates(config.oracle_budget, sum_gt, sum_estimation, t_lbs, t_ubs)
    est.log()
    est.save(config.output_file, "_sum_ttest")

    p_lbs, p_ubs, e_lbs, e_ubs, t_lbs, t_ubs = \
        get_ci_bootstrap(avg_bootstraps, avg_estimation, avg_ts, avg_var, confidence_levels=[0.95, 0.96, 0.97, 0.98, 0.99])
    est = Estimates(config.oracle_budget, avg_gt, avg_estimation, p_lbs, p_ubs)
    est.log()
    est.save(config.output_file, "_avg_percentile")
    est = Estimates(config.oracle_budget, avg_gt, avg_estimation, e_lbs, e_ubs)
    est.log()
    est.save(config.output_file, "_avg_empirical")
    est = Estimates(config.oracle_budget, avg_gt, avg_estimation, t_lbs, t_ubs)
    est.log()
    est.save(config.output_file, "_avg_ttest")




def stats_func(strata_population, strata_sample_count_results, strata_sample_sum_results) -> Tuple[float, float, float]:
    population_size = sum([len(stratum_population) for stratum_population in strata_population])
    sample_size = sum([len(stratum_sample_result) for stratum_sample_result in strata_sample_count_results])
    count_estimation, sum_estimation, avg_estimation = get_estimation(strata_population, strata_sample_count_results, strata_sample_sum_results)
    count_var = get_stratified_var(strata_population, strata_sample_count_results)
    avg_estimation -= get_avg_correction(population_size, sample_size, count_estimation, avg_estimation, count_var)
    return count_estimation, sum_estimation, avg_estimation


def get_estimation(strata_population, strata_sample_count_results, strata_sample_sum_results):
    count_estimation = 0
    sum_estimation = 0
    avg_estimation = 0
    for stratum_sample_count_result, stratum_sample_sum_result, stratum_population in zip(strata_sample_count_results, strata_sample_sum_results, strata_population):
        count_estimation += np.mean(stratum_sample_count_result) * len(stratum_population)
        sum_estimation += np.mean(stratum_sample_sum_result) * len(stratum_population)
    avg_estimation = sum_estimation / count_estimation
    return count_estimation, sum_estimation, avg_estimation

def get_avg_correction(population_size, sample_size, count_mean, avg_mean, count_var):
    return (population_size-sample_size)/(population_size-1) * avg_mean * count_var / sample_size / count_mean**2

def get_stratified_var(strata_population, strata_sample_results):
    variance = 0
    population_size = sum([len(stratum_population) for stratum_population in strata_population])
    for stratum_population, stratum_sample_result in zip(strata_population, strata_sample_results):
        sample_size = len(stratum_sample_result)
        stratum_size = len(stratum_population)
        stratum_var = np.var(stratum_sample_result, ddof=1).item()
        variance += (stratum_size / population_size)**2 * stratum_var / sample_size
    variance *= population_size**2
    return variance

def get_avg_var(sample_size, count_var, count_mean, sum_var, sum_mean):
    return 1. / sample_size * (sum_var / count_mean**2 + count_var * sum_mean**2 / count_mean**4)


def get_gt_strata(config, strata_population: List[List[int]], dataset_sizes, dataset: JoinDataset, oracle: Oracle) -> List[int]:
    gts = [-1, ]
    for i, stratum_population in enumerate(strata_population[1:]):
        data_ids = np.array(np.unravel_index(stratum_population, dataset_sizes)).T
        result = []
        for data_id in data_ids:
            if oracle.query(data_id):
                if config.aggregator == "count":
                    result.append(1)
                elif config.aggregator == "sum":
                    result.append(dataset.get_statistics(data_id))
                else:
                    result.append(dataset.get_statistics(data_id) / len(stratum_population))
        gts.append(sum(result))
    return gts

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

    logging.debug(f"count gt: {count_gt}, sum gt: {sum_gt}, avg gt: {avg_gt}")

    # get proxy
    proxy_scores = get_proxy_score(config, dataset)
    proxy_rank = get_proxy_rank(config, dataset, proxy_scores)
    proxy_sum = proxy_scores.sum()

    # divide population into strata:
    # 1. a big stratum to only run sampling algorithm
    # 2. a series of strata to potentially run blocking algorithm
    sample_size = int((1-config.max_blocking_ratio) * config.oracle_budget)
    blocking_size_upperbound = config.oracle_budget - sample_size
    strata = [[0, int(proxy_rank.shape[0]) - blocking_size_upperbound]]

    # allocate sample size and strata size: sample size is proportional to the sum of proxy scores in the strata
    # 1. first split the population into two parts: sampling & blocking
    strata_population = [proxy_rank[strata[0][0]:strata[0][1]]]
    # strata_proxy_scores = [proxy_scores[strata_population[0]]]
    strata_sample_sizes = [int(proxy_scores[strata_population[0]].sum() / proxy_scores.sum() * sample_size)]
    sample_size_remaining = sample_size - strata_sample_sizes[0]
    # 2. get the number of strata by making sure each strata have at least 1000 samples
    num_strata = max(int(sample_size_remaining / 1000), 1)
    blocking_stratum_size = int(blocking_size_upperbound / num_strata)
    for i in range(num_strata):
        if i != num_strata - 1:
            strata.append([strata[i][1], strata[i][1] + blocking_stratum_size])
        else:
            strata.append([strata[i][1], int(proxy_rank.shape[0])])
        strata_population.append(proxy_rank[strata[i+1][0]:strata[i+1][1]])
        strata_sample_sizes.append(max(1000, int(proxy_scores[strata_population[i+1]].sum() / proxy_sum * sample_size)))
    logging.debug(f"{num_strata + 1} strata sample sizes: {strata_sample_sizes}")

    for exp_id in range(config.internal_loop):
        logging.info(f"running {exp_id} experiments")
        run_once(config, dataset, oracle, dataset_sizes, count_gt, sum_gt, avg_gt,
                 proxy_scores, strata, strata_population, strata_sample_sizes)
