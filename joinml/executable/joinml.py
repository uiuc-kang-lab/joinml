from joinml.proxy.get_proxy import get_proxy_score, get_proxy_rank
from joinml.dataset_loader import load_dataset, JoinDataset
from joinml.oracle import Oracle
from joinml.config import Config
from joinml.utils import set_up_logging, normalize
from joinml.estimates import Estimates
from joinml.utils import get_ci_bootstrap, weighted_sample_pd

import os
import logging
import numpy as np
from typing import Tuple, List
from scipy import stats

def get_non_positive_ci(max_statistics: float, 
                        min_statistics: float,
                        confidence_level: float, 
                        n_positive_population_size: int, 
                        n_positive_sample_size: int):
    z = float(stats.norm.ppf(1 - (1 - confidence_level) / 2))
    pr_upper_bound = 1 / (1+z**2/n_positive_sample_size) * (z**2/n_positive_sample_size)
    n_positive_count_lower_bound = 0
    n_positive_count_upper_bound = pr_upper_bound * n_positive_population_size
    n_positive_sum_lower_bound = 0
    n_positive_sum_upper_bound = n_positive_count_upper_bound * max_statistics
    n_positive_avg_lower_bound = min_statistics
    n_positive_avg_upper_bound = max_statistics
    return n_positive_count_lower_bound, n_positive_count_upper_bound, \
           n_positive_sum_lower_bound, n_positive_sum_upper_bound, \
           n_positive_avg_lower_bound, n_positive_avg_upper_bound

def empirical_best_blocking_allocation(strata_population: List[int], strata_sample_sizes: List[int],
                                       strata_sample_variance: List[float], objective: str="CI") -> Tuple[List[int], List[int]]:
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
            variance += (strata_population[j] / total_population)**2 * \
                        (strata_population[j] - strata_sample_sizes[j]) / strata_population[j] * \
                        strata_sample_variance[j]
        total_variances.append(variance)
    
    logging.debug(f"total variances: {total_variances}")
    
    # calculate the utility based on the objective
    utility = []
    if objective == "joinml-ci":
        for i in range(len(strata_to_decide)):
            total_sample_size = sum([strata_sample_sizes[j] for j in strata_to_decide[:i+1]])
            utility.append(total_variances[i] / total_sample_size)
    elif objective == "joinml-mse":
        for i in range(len(strata_to_decide)):
            total_nonblocking_ratio = sum([strata_population[j] for j in strata_to_decide[:i+1]]) / sum(strata_population)
            utility.append(total_variances[i] * (total_nonblocking_ratio)**2)
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

def calculate_mean_for_strata(strata_population: List[int], strata_sample_sizes: List[int], 
                              strata_sample_means: List[float], strata_gts: List[float|int],
                              strata_count_gts: List[float|int],
                              sampling_strata: List[int], blocking_strata: List[int],
                              aggregator: str="count") -> float:
    # make sure the strata are all allocated in either sampling or blocking
    assert len(set(sampling_strata).intersection(set(blocking_strata))) == 0
    assert len(sampling_strata) + len(blocking_strata) == len(strata_population)

    sampling_population_size = sum([strata_population[i] for i in sampling_strata])
    sampling_mean = 0
    for i in sampling_strata:
        sampling_mean += strata_sample_means[i] * strata_population[i] / sampling_population_size
    
    logging.debug(f"sampling mean: {sampling_mean}")

    blocking_population_size = sum([strata_population[i] for i in blocking_strata])
    blocking_mean = calculate_mean_blocking(strata_gts, strata_count_gts, blocking_strata, aggregator, blocking_population_size)
    
    total_population_size = sum(strata_population)

    mean = sampling_mean * sampling_population_size / total_population_size + \
        blocking_mean * blocking_population_size / total_population_size

    return mean

def calculate_mean_blocking(strata_gts, strata_count_gts, blocking_strata, aggregator, blocking_population_size):
    blocking_total_count = 0
    blocking_total_sum = .0
    for i in blocking_strata:
        blocking_total_count += strata_count_gts[i]
        blocking_total_sum += strata_gts[i]
    if aggregator == "count":
        blocking_mean = blocking_total_count / blocking_population_size
        logging.debug(f"blocking count mean: {blocking_mean}, blocking total count: {blocking_total_count}")
    elif aggregator == "sum":
        blocking_mean = blocking_total_sum / blocking_population_size
        logging.debug(f"blocking sum mean: {blocking_mean}, blocking total sum: {blocking_total_sum}")
    elif aggregator == "avg":
        blocking_mean = blocking_total_sum / blocking_total_count
        logging.debug(f"blocking avg mean: {blocking_mean}")
    else:
        raise ValueError(f"aggregator {aggregator} is not supported")
    return blocking_mean

def run_once(config, dataset, oracle, dataset_sizes, max_statistics, min_statistics,
             count_gt, sum_gt, avg_gt, 
             proxy_scores, strata, strata_population, strata_sample_sizes):

    strata_sample_results = []
    strata_means = []
    strata_mean_vars = []
    strata_sample_count_results = []
    strata_count_means = []
    strata_count_mean_vars = []
    strata_sample_avg_results = []
    strata_avg_means = []
    strata_avg_mean_vars = []
    cache_ids = set()

    # IS
    for i, (stratum_begin, stratum_end) in enumerate(strata):
        # IS
        stratum_proxy_scores = proxy_scores[strata_population[i]]
        stratum_proxy_weights = normalize(stratum_proxy_scores)
        stratum_sample = weighted_sample_pd(stratum_proxy_scores, strata_sample_sizes[i], replace=True)
        stratum_sample_ids = strata_population[i][stratum_sample]
        # run oracle for statistics
        for i in stratum_sample_ids:
            cache_ids.add(i)
        stratum_sample_ids = np.array(np.unravel_index(stratum_sample_ids, dataset_sizes)).T
        stratum_sample_count_results = []
        stratum_sample_results = []
        stratum_sample_avg_results = []
        for stratum_s_id, stratum_s in zip(stratum_sample_ids, stratum_sample):
            if oracle.query(stratum_s_id):
                stats = dataset.get_statistics(stratum_s_id)
                stratum_sample_results.append(stats / len(stratum_proxy_weights) / stratum_proxy_weights[stratum_s])
                stratum_sample_count_results.append(1 / len(stratum_proxy_weights) / stratum_proxy_weights[stratum_s])
                stratum_sample_avg_results.append(stats)
            else:
                stratum_sample_results.append(0)
                stratum_sample_count_results.append(0)
        stratum_sample_results = np.array(stratum_sample_results)
        stratum_sample_count_results = np.array(stratum_sample_count_results)
        stratum_sample_avg_results = np.array(stratum_sample_avg_results)
        strata_sample_results.append(stratum_sample_results)
        strata_sample_count_results.append(stratum_sample_count_results)
        strata_sample_avg_results.append(stratum_sample_avg_results)
            
        # calculate mean and variance of mean
        if sum(stratum_sample_count_results) > 0:
            strata_means.append(np.mean(stratum_sample_results))
            strata_mean_vars.append(np.var(stratum_sample_results, ddof=1).item() / len(stratum_sample_results))
            strata_count_means.append(np.mean(stratum_sample_count_results))
            strata_count_mean_vars.append(np.var(stratum_sample_count_results, ddof=1).item() / len(stratum_sample_count_results))
        else:
            strata_means.append(np.nan)
            strata_mean_vars.append(np.nan)
            strata_count_means.append(np.nan)
            strata_count_mean_vars.append(np.nan)

        if len(stratum_sample_avg_results) <= 1:
            strata_avg_means.append(np.nan)
            strata_avg_mean_vars.append(np.nan)
        else:
            strata_avg_means.append(np.mean(stratum_sample_avg_results))
            strata_avg_mean_vars.append(np.var(stratum_sample_avg_results, ddof=1).item() / len(stratum_sample_avg_results))

    logging.debug(f"strata means: {strata_means}")
    logging.debug(f"strata vars: {strata_mean_vars}")
    logging.debug(f"strata count means: {strata_count_means}")
    logging.debug(f"strata count vars: {strata_count_mean_vars}")
    logging.debug(f"strata avg means: {strata_avg_means}")
    logging.debug(f"strata avg vars: {strata_avg_mean_vars}")

    strata_gt, strata_count_gt = get_gt(dataset, oracle, dataset_sizes, strata, strata_population)

    # calculate the mean
    strata_population_size = [len(stratum_population) for stratum_population in strata_population]
    count_mean, sum_mean, avg_mean = get_means(config, strata, strata_sample_sizes, 
                                               strata_means, strata_mean_vars, 
                                               strata_count_means, strata_count_mean_vars, 
                                               strata_avg_means, strata_avg_mean_vars, 
                                               strata_gt, strata_count_gt, 
                                               strata_population_size)

    logging.debug(f"count mean {count_mean}")
    logging.debug(f"sum mean {sum_mean}")
    logging.debug(f"avg mean {avg_mean}")

    # calculate the estimation of statistics
    count_result = count_mean * sum(strata_population_size)
    sum_result = sum_mean * sum(strata_population_size)
    avg_result = avg_mean

    # calculate the confidence interval of the statistics if necessary
    if config.need_ci:
        if sum(strata_sample_results[0]) > 0:
            btstp_sums, btstp_counts, btstp_avgs = \
                get_bootstrapping_ci(config, strata, strata_sample_sizes, 
                                    strata_sample_results, strata_sample_count_results, strata_sample_avg_results, 
                                    strata_gt, strata_count_gt, strata_population_size)
            count_lower, count_upper = get_ci_bootstrap(btstp_counts)
            sum_lower, sum_upper = get_ci_bootstrap(btstp_sums)
            avg_lower, avg_upper = get_ci_bootstrap(btstp_avgs)
        else:
            count_lower, count_upper, sum_lower, sum_upper, avg_lower, avg_upper = \
                get_non_positive_ci(max_statistics, min_statistics, config.confidence_level, strata_population_size[0], strata_sample_sizes[0])

    else:
        count_lower, count_upper, sum_lower, sum_upper, avg_lower, avg_upper = -1, -1, -1, -1, -1, -1


    count_est = Estimates(config.oracle_budget, count_gt, count_result, count_lower, count_upper)
    sum_est = Estimates(config.oracle_budget, sum_gt, sum_result, sum_lower, sum_upper)
    avg_est = Estimates(config.oracle_budget, avg_gt, avg_result, avg_lower, avg_upper)

    count_est.log()
    sum_est.log()
    avg_est.log()
    count_est.save(config.output_file, "_count")
    sum_est.save(config.output_file, "_sum")
    avg_est.save(config.output_file, "_avg")

def get_bootstrapping_ci(config, strata, strata_sample_sizes, strata_sample_results, strata_sample_count_results, strata_sample_avg_results, strata_gt, strata_count_gt, strata_population_size):
    btstp_sums = []
    btstp_counts = []
    btstp_avgs = []
    for _ in range(config.bootstrap_trials):
            # resample
        strata_resample_means = []
        strata_resample_mean_vars = []
        strata_count_resample_means = []
        strata_count_resample_mean_vars = []
        strata_avg_resample_means = []
        strata_avg_resample_mean_vars = []
        for i in range(len(strata)):
            stratum_resample = np.random.choice(strata_sample_results[i], len(strata_sample_results[i]), replace=True)
            stratum_count_resample = np.random.choice(strata_sample_count_results[i], len(strata_sample_count_results[i]), replace=True)
                
            stratum_resample_mean = np.mean(stratum_resample)
            stratum_resample_mean_var = np.var(stratum_resample).item() / len(stratum_resample)
            stratum_count_resample_mean = np.mean(stratum_count_resample)
            stratum_count_resample_mean_var = np.var(stratum_count_resample).item() / len(stratum_count_resample)

            strata_resample_means.append(stratum_resample_mean)
            strata_resample_mean_vars.append(stratum_resample_mean_var)
            strata_count_resample_means.append(stratum_count_resample_mean)
            strata_count_resample_mean_vars.append(stratum_count_resample_mean_var)
            
            if len(strata_sample_avg_results[i]) >= 1:
                stratum_avg_resample = np.random.choice(strata_sample_avg_results[i], len(strata_sample_avg_results[i]), replace=True)
                stratum_avg_resample_mean = np.mean(stratum_avg_resample)
                stratum_avg_resample_mean_var = np.var(stratum_avg_resample).item() / len(stratum_avg_resample)
                strata_avg_resample_means.append(stratum_avg_resample_mean)
                strata_avg_resample_mean_vars.append(stratum_avg_resample_mean_var)
            else:
                strata_avg_resample_means.append(np.nan)
                strata_avg_resample_mean_vars.append(np.nan)

        count_mean, sum_mean, avg_mean = get_means(config, strata, strata_sample_sizes, 
                                                    strata_resample_means, strata_resample_mean_vars, 
                                                    strata_count_resample_means, strata_count_resample_mean_vars, 
                                                    strata_avg_resample_means, strata_avg_resample_mean_vars, 
                                                    strata_gt, strata_count_gt, 
                                                    strata_population_size)
        count_result = count_mean * sum(strata_population_size)
        sum_result = sum_mean * sum(strata_population_size)
        avg_result = avg_mean

        btstp_sums.append(count_result)
        btstp_counts.append(sum_result)
        btstp_avgs.append(avg_result)

    return btstp_sums,btstp_counts,btstp_avgs

def get_means(config, strata, strata_sample_sizes, strata_means, strata_mean_vars, strata_count_means, strata_count_mean_vars, strata_avg_means, strata_avg_mean_vars, strata_gt, strata_count_gt, strata_population_size):
    if np.isnan(strata_avg_means[0]):
        blocking_strata = list(range(1, len(strata)))
        logging.debug(f"count sampling strata: [0], count blocking strata: {blocking_strata}")
        logging.debug(f"sum sampling strata: [0], sum blocking strata: {blocking_strata}")
        logging.debug(f"avg sampling strata: [0], avg blocking strata: {blocking_strata}")

        blocking_population_size = sum([strata_population_size[i] for i in range(1, len(strata))])
        count_mean = calculate_mean_blocking(strata_gt, strata_count_gt, blocking_strata, "count", blocking_population_size)
        sum_mean = calculate_mean_blocking(strata_gt, strata_count_gt, blocking_strata, "sum", blocking_population_size)
        avg_mean = calculate_mean_blocking(strata_gt, strata_count_gt, blocking_strata, "avg", blocking_population_size)

        total_population_size = sum(strata_population_size)
        count_mean = count_mean * blocking_population_size / total_population_size
        sum_mean = sum_mean * blocking_population_size / total_population_size
        avg_mean = avg_mean * blocking_population_size / total_population_size

    else:
        count_sampling_strata, count_blocking_strata = empirical_best_blocking_allocation(strata_population_size, strata_sample_sizes, strata_count_mean_vars, objective=config.task)
        sum_sampling_strata, sum_blocking_strata = empirical_best_blocking_allocation(strata_population_size, strata_sample_sizes, strata_mean_vars, objective=config.task)
        avg_sampling_strata, avg_blocking_strat = empirical_best_blocking_allocation(strata_population_size, strata_sample_sizes, strata_avg_mean_vars, objective=config.task)

        logging.debug(f"count sampling strata: {count_sampling_strata}, count blocking strata: {count_blocking_strata}")
        logging.debug(f"sum sampling strata: {sum_sampling_strata}, sum blocking strata: {sum_blocking_strata}")
        logging.debug(f"avg sampling strata: {avg_sampling_strata}, avg blocking strata: {avg_blocking_strat}")

        count_mean = calculate_mean_for_strata(strata_population_size, strata_sample_sizes, strata_count_means, strata_gt, strata_count_gt, count_sampling_strata, count_blocking_strata, aggregator="count")
        sum_mean = calculate_mean_for_strata(strata_population_size, strata_sample_sizes, strata_means, strata_gt, strata_count_gt, sum_sampling_strata, sum_blocking_strata, aggregator="sum")
        avg_mean = calculate_mean_for_strata(strata_population_size, strata_sample_sizes, strata_avg_means, strata_gt, strata_count_gt, avg_sampling_strata, avg_blocking_strat, aggregator="avg")
    return count_mean,sum_mean,avg_mean

def get_gt(dataset, oracle, dataset_sizes, strata, strata_population):
    strata_gt = [-1.]
    strata_count_gt = [-1.]
    for i in range(1, len(strata)):
        stratum_population = strata_population[i]
        stratum_ids = np.array(np.unravel_index(stratum_population, dataset_sizes)).T
        stratum_gt = 0
        stratum_count_gt = 0
        for stratum_id in stratum_ids:
            if oracle.query(stratum_id):
                stratum_gt += dataset.get_statistics(stratum_id)
                stratum_count_gt += 1
        strata_gt.append(stratum_gt)
        strata_count_gt.append(stratum_count_gt)
    return strata_gt,strata_count_gt

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
    logging.debug("finish proxy loading")
    proxy_sum = proxy_scores.sum()
    logging.debug(f"sum of proxy scores {proxy_sum}")

    # divide population into strata
    sample_size = int((1-config.max_blocking_ratio) * config.oracle_budget)
    blocking_size_upperbound = config.oracle_budget - sample_size
    strata = [[0, int(proxy_rank.shape[0]) - blocking_size_upperbound]]
    # allocate sample size and strata size
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
        # strata_proxy_scores.append(proxy_scores[strata_population[i+1]])
        # strata_sample_sizes.append(max(int(sum(strata_proxy_scores[i+1]) / proxy_sum * sample_size), 1))
        strata_sample_sizes.append(max(1, int(proxy_scores[strata_population[i+1]].sum() / proxy_sum * sample_size)))
    logging.debug(f"{num_strata + 1} strata sample sizes: {strata_sample_sizes}")

    for exp_id in range(config.internal_loop):
        logging.info(f"running {exp_id} experiments")
        run_once(config, dataset, oracle, dataset_sizes, max_statistics, min_statistics,
                 count_gt, sum_gt, avg_gt, 
                 proxy_scores, strata, strata_population, strata_sample_sizes)
