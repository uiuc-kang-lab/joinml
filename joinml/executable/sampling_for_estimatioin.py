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

def bootstrap_blocking_sampling(config, dataset, oracle, dataset_sizes, 
                                count_gt, sum_gt, avg_gt, 
                                strata, strata_sample_sizes, strata_population, 
                                strata_sample_results, strata_sample_count_results,
                                cache_ids: set):
    # get the groundtruth for each stratum other than the first one
    strata_gt = [-1]
    strata_count_gt = [-1]
    strata_ids = [[-1]]
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
        strata_ids.append(stratum_population)
        strata_count_gt.append(stratum_count_gt)
    logging.debug(f"strata gt: {strata_gt}")
    logging.debug(f"strata count gt: {strata_count_gt}")
    # bootstrap
    bootstrap_results = {
            "count": [],
            "sum": [],
            "avg": []
        }
    for trial in range(config.bootstrap_trials):
        # resampling
        strata_resample_results = []
        strata_resample_count_results = []
        for i in range(len(strata)):
            resample = np.random.choice(len(strata_sample_results[i]), size=len(strata_sample_results[i]), replace=True)
            resample_results = strata_sample_results[i][resample]
            resample_count_results = strata_sample_count_results[i][resample]
            resample_var = np.var(resample_results, ddof=1)
            resample_mean = np.mean(resample_results)
            resample_count_var = np.var(resample_count_results, ddof=1)
            resample_count_mean = np.mean(resample_count_results)
            strata_resample_results.append({
                    "results": resample_results,
                    "var": resample_var,
                    "mean": resample_mean
                })
            strata_resample_count_results.append({
                    "results": resample_count_results,
                    "var": resample_count_var,
                    "mean": resample_count_mean
                })
        # calculate variance of subpopulation
        subpopulation_utility = []
        subpopulation_count_utility = []
        for i in range(1, len(strata)):
            all_strata_sizes = sum([len(stratum_population) for stratum_population in strata_population[:i+1]])
            variance = 0
            count_variance = 0
            subpopulation_size = 0
            all_sample_size = 0
            for j in range(i+1):
                stratum_population_size = len(strata_population[j])
                subpopulation_size += stratum_population_size
                all_sample_size += strata_sample_sizes[j]
                stratum_variance = (stratum_population_size / all_strata_sizes)**2 * \
                        (stratum_population_size - strata_sample_sizes[j]) / stratum_population_size * \
                        strata_resample_results[j]["var"] / strata_sample_sizes[j]
                stratum_count_variance = (stratum_population_size / all_strata_sizes)**2 * \
                        (stratum_population_size - strata_sample_sizes[j]) / stratum_population_size * \
                        strata_resample_count_results[j]["var"] / strata_sample_sizes[j]
                variance += stratum_variance
                count_variance += stratum_count_variance
            subpopulation_utility.append(variance / all_sample_size)
            subpopulation_count_utility.append(count_variance / all_sample_size)
        subpopulation_utility = np.array(subpopulation_utility)
        subpopulation_count_utility = np.array(subpopulation_count_utility)
        # get the best allocation
        min_var = np.min(subpopulation_utility)
        min_count_var = np.min(subpopulation_count_utility)
        if np.isnan(min_var):
            optimal_allocation = len(subpopulation_utility)
        else:
            optimal_allocation = 1 + [i for i in range(subpopulation_utility.shape[0]) if subpopulation_utility[i] == min_var][-1]
        if np.isnan(min_count_var):
            optimal_count_allocation = len(subpopulation_count_utility)
        else:
            optimal_count_allocation = 1 + [i for i in range(subpopulation_count_utility.shape[0]) if subpopulation_count_utility[i] == min_count_var][-1]
        # ====== debug ======
        logging.debug(f"resample variance {[strata_resample_results[i]['var'] for i in range(len(strata_resample_results))]}")
        logging.debug(f"optimal allocation {optimal_allocation} subpopulation variance {subpopulation_utility.tolist()}")
        logging.debug(f"resample count variance {[strata_resample_count_results[i]['var'] for i in range(len(strata_resample_count_results))]}")
        logging.debug(f"optimal count allocation {optimal_count_allocation} subpopulation count variance {subpopulation_count_utility.tolist()}")
        # calculate the statistics for sampling
        sampling_sum_mean = 0
        sampling_population_size = sum([len(stratum_population) for stratum_population in strata_population[:optimal_allocation+1]])
        for i in range(optimal_allocation+1):
            sum_mean = strata_resample_results[i]["mean"] * len(strata_population[i]) / sampling_population_size
            sampling_sum_mean += sum_mean
        sampling_sum = sampling_sum_mean * sampling_population_size

        sampling_count_mean = 0
        sampling_population_size_count = sum([len(stratum_population) for stratum_population in strata_population[:optimal_count_allocation+1]])
        for i in range(optimal_count_allocation+1):
            count_mean = strata_resample_count_results[i]["mean"] * len(strata_population[i]) / sampling_population_size_count
            sampling_count_mean += count_mean
        sampling_count = sampling_count_mean * sampling_population_size_count

        # calculate the statistics for blocking
        blocking_sum = 0
        for i in range(optimal_allocation+1, len(strata)):
            blocking_sum += strata_gt[i]
        blocking_count = 0
        for i in range(optimal_count_allocation+1, len(strata)):
            blocking_count += strata_count_gt[i]
        logging.debug(f"sampling count: {sampling_count}, sampling sum: {sampling_sum} blocking count: {blocking_count}, blocking sum: {blocking_sum}")

        # calculate the cost
        allocation = min(optimal_allocation, optimal_count_allocation)
        for stratum_ids in strata_ids[allocation+1: len(strata)]:
            for stratum_id in stratum_ids:
                cache_ids.add(stratum_id)
        logging.debug(f"current cost {len(cache_ids)}")
        # combine the statistics
        count_result = sampling_count + blocking_count
        sum_result = sampling_sum + blocking_sum
        avg_result = sum_result / count_result
        bootstrap_results["count"].append(count_result)
        bootstrap_results["sum"].append(sum_result)
        bootstrap_results["avg"].append(avg_result)
        logging.debug(f"bootstrap count: {count_result} ({(count_result - count_gt)/count_gt*100}%), bootstrap sum: {sum_result} ({(sum_result - sum_gt)/sum_gt*100}%), bootstrap avg: {avg_result} ({(avg_result - avg_gt)/avg_gt*100}%)")

    cost = len(cache_ids)

    # calculate the CI
    count_mean = float(np.mean(bootstrap_results["count"]))
    count_lb, count_ub = get_ci_bootstrap(bootstrap_results["count"], config.confidence_level)
    sum_mean = float(np.mean(bootstrap_results["sum"]))
    sum_lb, sum_ub = get_ci_bootstrap(bootstrap_results["sum"], config.confidence_level)
    avg_mean = float(np.mean(bootstrap_results["avg"]))
    avg_lb, avg_ub = get_ci_bootstrap(bootstrap_results["avg"], config.confidence_level)
    # log the results
    count_est = Estimates(cost, count_gt, count_mean, count_lb, count_ub)
    sum_est = Estimates(cost, sum_gt, sum_mean, sum_lb, sum_ub)
    avg_est = Estimates(cost, avg_gt, avg_mean, avg_lb, avg_ub)
    count_est.log()
    sum_est.log()
    avg_est.log()
    count_est.save(config.output_file, "_count")
    sum_est.save(config.output_file, "_sum")
    avg_est.save(config.output_file, "_avg")

    # calculate the total cost
    logging.info(f"total cost {len(cache_ids)}")

def empirical_best_blocking_allocation(strata_population: List[int], strata_sample_sizes: List[int],
                                       strata_sample_variance: List[float], objective: str="CI") -> Tuple[List[int], List[int]]:
    # blocking for all the non-positive data
    strata_to_decide = []
    for i in range(len(strata_population)):
        if not np.isnan(strata_sample_variance[i]):
            strata_to_decide.append(i)
    
    # calculate the total variance of the sub population
    total_variances = []
    for i in range(1, len(strata_to_decide)):
        variance = 0
        total_population = sum([len(strata_population[j]) for j in strata_to_decide[:i+1]])
        for j in strata_to_decide[:i+1]:
            variance += (strata_population[j] / total_population)**2 * \
                (strata_population[j] - strata_sample_sizes[j]) / strata_population[j] * \
                strata_sample_variance[j] / strata_sample_sizes[j]
        total_variances.append(variance)
    
    logging.debug(f"total variances: {total_variances}")
    
    # calculate the utility based on the objective
    utility = []
    if objective == "CI":
        for i in range(1, len(strata_to_decide)):
            total_sample_size = sum([strata_sample_sizes[j] for j in strata_to_decide[:i+1]])
            utility.append(total_variances[i-1] / total_sample_size)
    elif objective == "MSE":
        for i in range(1, len(strata_to_decide)):
            total_nonblocking_ratio = sum([strata_population[j] for j in strata_to_decide[:i+1]]) / sum(strata_population)
            utility.append(total_variances[i-1] * (total_nonblocking_ratio)**2)
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
    blocking_total_count = 0
    blocking_total_sum = .0
    for i in blocking_strata:
        blocking_total_count += strata_count_gts[i]
        blocking_total_sum += strata_gts[i]
    if aggregator == "count":
        blocking_mean = blocking_total_count / blocking_population_size
        logging.debug(f"blocking count mean: {blocking_total_count}, blocking total count: {blocking_total_count}")
    elif aggregator == "sum":
        blocking_mean = blocking_total_sum / blocking_population_size
        logging.debug(f"blocking sum mean: {blocking_total_sum}, blocking total sum: {blocking_total_sum}")
    elif aggregator == "avg":
        blocking_mean = blocking_total_sum / blocking_total_count
        logging.debug(f"blocking avg mean: {blocking_total_sum}")
    else:
        raise ValueError(f"aggregator {aggregator} is not supported")
    
    total_population_size = sum(strata_population)

    mean = sampling_mean * sampling_population_size / total_population_size + \
        blocking_mean * blocking_population_size / total_population_size

    return mean


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

    # divide population into strata
    sample_size = int((1-config.max_blocking_ratio) * config.oracle_budget)
    blocking_size_upperbound = config.oracle_budget - sample_size
    strata = [[0, int(proxy_rank.shape[0]) - blocking_size_upperbound]]
    # allocate sample size and strata size
    # 1. first split the population into two parts: sampling & blocking
    strata_population = [proxy_rank[strata[0][0]:strata[0][1]]]
    strata_proxy_scores = [proxy_scores[strata_population[0]]]
    strata_sample_sizes = [int(sum(strata_proxy_scores[0]) / sum(proxy_scores) * sample_size)]
    sample_size_remaining = sample_size - strata_sample_sizes[0]
    # 2. get the number of strata by making sure each strata have at least 500 samples
    num_strata = max(int(sample_size_remaining / 500), 1)
    blocking_stratum_size = int(blocking_size_upperbound / num_strata)
    for i in range(num_strata):
        if i != num_strata - 1:
            strata.append([strata[i][1], strata[i][1] + blocking_stratum_size])
        else:
            strata.append([strata[i][1], int(proxy_rank.shape[0])])
        strata_population.append(proxy_rank[strata[i+1][0]:strata[i+1][1]])
        strata_proxy_scores.append(proxy_scores[strata_population[i+1]])
        strata_sample_sizes.append(max(int(sum(strata_proxy_scores[i+1]) / sum(proxy_scores) * sample_size), 1))
    logging.debug(f"{num_strata + 1} strata sample sizes: {strata_sample_sizes}")

    # IS for each stratum
    strata_means = []
    strata_vars = []
    strata_count_means = []
    strata_count_vars = []
    strata_avg_means = []
    strata_avg_vars = []
    cache_ids = set()
    for i, (stratum_begin, stratum_end) in enumerate(strata):
        stratum_proxy_scores = strata_proxy_scores[i]
        stratum_proxy_weights = normalize(stratum_proxy_scores)
        stratum_sample = weighted_sample_pd(stratum_proxy_scores, strata_sample_sizes[i], replace=True)
        stratum_sample_ids = strata_population[i][stratum_sample]
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
        strata_means.append(np.mean(stratum_sample_results))
        strata_vars.append(np.var(stratum_sample_results, ddof=1))
        strata_count_means.append(np.mean(stratum_sample_count_results))
        strata_count_vars.append(np.var(stratum_sample_count_results, ddof=1))
        if len(stratum_sample_avg_results) <= 0:
            strata_avg_means.append(np.nan)
            strata_avg_vars.append(np.nan)
        else:
            strata_avg_means.append(np.mean(stratum_sample_avg_results))
            strata_avg_vars.append(np.var(stratum_sample_avg_results, ddof=1))

    logging.debug(f"strata means: {strata_means}")
    logging.debug(f"strata vars: {strata_vars}")
    logging.debug(f"strata count means: {strata_count_means}")
    logging.debug(f"strata count vars: {strata_count_vars}")
    logging.debug(f"strata avg means: {strata_avg_means}")
    logging.debug(f"strata avg vars: {strata_avg_vars}")

    # calculate the optimal allocation
    strata_population_size = [len(stratum_population) for stratum_population in strata_population]
    count_sampling_strata, count_blocking_strata = empirical_best_blocking_allocation(strata_population_size, strata_sample_sizes, strata_count_vars, objective="MSE")
    sum_sampling_strata, sum_blocking_strata = empirical_best_blocking_allocation(strata_population_size, strata_sample_sizes, strata_vars, objective="MSE")
    avg_sampling_strata, avg_blocking_strat = empirical_best_blocking_allocation(strata_population_size, strata_sample_sizes, strata_avg_vars, objective="MSE")

    logging.debug(f"count sampling strata: {count_sampling_strata}, count blocking strata: {count_blocking_strata}")
    logging.debug(f"sum sampling strata: {sum_sampling_strata}, sum blocking strata: {sum_blocking_strata}")
    logging.debug(f"avg sampling strata: {avg_sampling_strata}, avg blocking strata: {avg_blocking_strat}")

    strata_gt = [-1.]
    strata_count_gt = [-1.]
    strata_ids = [[-1]]
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
        strata_ids.append(stratum_population.tolist())
        strata_count_gt.append(stratum_count_gt)

    count_mean = calculate_mean_for_strata(strata_population_size, strata_sample_sizes, strata_means, strata_gt, strata_count_gt, count_sampling_strata, count_blocking_strata, aggregator="count")
    sum_mean = calculate_mean_for_strata(strata_population_size, strata_sample_sizes, strata_means, strata_gt, strata_count_gt, sum_sampling_strata, sum_blocking_strata, aggregator="sum")
    avg_mean = calculate_mean_for_strata(strata_population_size, strata_sample_sizes, strata_means, strata_gt, strata_count_gt, avg_sampling_strata, avg_blocking_strat, aggregator="avg")

    logging.debug(f"count mean: {count_mean}")
    logging.debug(f"sum mean: {sum_mean}")
    logging.debug(f"avg mean: {avg_mean}")

    # combine the statistics
    count_result = count_mean * sum(strata_population_size)
    sum_result = sum_mean * sum(strata_population_size)
    avg_result = avg_mean

    count_est = Estimates(config.oracle_budget, count_gt, count_result, 0, 0)
    sum_est = Estimates(config.oracle_budget, sum_gt, sum_result, 0, 0)
    avg_est = Estimates(config.oracle_budget, avg_gt, avg_result, 0, 0)

    count_est.log()
    sum_est.log()
    avg_est.log()
    count_est.save(config.output_file, "_count")
    sum_est.save(config.output_file, "_sum")
    avg_est.save(config.output_file, "_avg")
