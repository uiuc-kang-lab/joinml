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
            variance += (strata_population[j] / total_population)**2 * \
                        (strata_population[j] - strata_sample_sizes[j]) / strata_population[j] * \
                        strata_sample_variance[j]
        total_variances.append(variance)
    
    logging.debug(f"total variances: {total_variances}")
    
    # calculate the utility based on the objective
    utility = []
    if objective == "joinml-ci":
        for i in range(len(strata_to_decide)):
            utility.append(total_variances[i])
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

def run_once(config, dataset, oracle, dataset_sizes, gt, proxy_scores, strata, strata_population, strata_sample_sizes):

    strata_sample_results = []          # f(x) * O(x)
    strata_sample_oracle_results = []   # O(x)
    strata_mean_vars = []               # variance of f(x) * O(x)
    total_n_positive = 0                # |O(x) = 1|
    # run sampling with replacement for each stratum
    for i, (stratum_begin, stratum_end) in enumerate(strata):
        if config.aggregator in ["count", "sum"]:
            # run importance sampling for sum and count
            stratum_proxy_scores = proxy_scores[strata_population[i]]
            stratum_proxy_weights = normalize(stratum_proxy_scores)
            stratum_sample = weighted_sample_pd(stratum_proxy_scores, strata_sample_sizes[i], replace=True)
        else:
            # uniform for avg and other aggregators
            stratum_proxy_weights = []
            stratum_sample = np.random.choice(len(strata_population[i]), strata_sample_sizes[i], replace=True)
        
        # run oracle for statistics
        stratum_sample_ids = strata_population[i][stratum_sample]
        stratum_sample_ids = np.array(np.unravel_index(stratum_sample_ids, dataset_sizes)).T
        stratum_sample_results = []
        stratum_sample_oracle_results = []
        n_positives = 0
        for stratum_s_id, stratum_s in zip(stratum_sample_ids, stratum_sample):
            if oracle.query(stratum_s_id):
                stratum_sample_oracle_results.append(1)
                n_positives += 1
                stats = dataset.get_statistics(stratum_s_id)
                if config.aggregator == "count":
                    sample_result = 1 / len(stratum_proxy_weights) / stratum_proxy_weights[stratum_s]
                elif config.aggregator == "sum":
                    sample_result = stats / len(stratum_proxy_weights) / stratum_proxy_weights[stratum_s]
                elif config.aggregator == "avg":
                    sample_result = stats
                else:
                    raise ValueError(f"Unknown aggregator {config.aggregator}")
                stratum_sample_results.append(sample_result)
            else:
                stratum_sample_oracle_results.append(0)
                stratum_sample_results.append(0)

        strata_sample_results.append(stratum_sample_results)
        strata_sample_oracle_results.append(stratum_sample_oracle_results)

        # calculate mean and variance of mean
        if len(stratum_sample_results) <= 1:
            strata_mean_vars.append(np.nan)
        else:
            if n_positives == 0:
                strata_mean_vars.append(np.nan)
            else:
                strata_mean_vars.append(np.var(stratum_sample_results, ddof=1).item() / len(stratum_sample_results))
        total_n_positive += n_positives
    

    logging.debug(f"{config.aggregator} strata vars: {strata_mean_vars}")
    logging.debug(f"total number of positive samples in stage 1 sampling {total_n_positive}")

    strata_population_size = [len(stratum_population) for stratum_population in strata_population]

    if np.isnan(strata_mean_vars[0]):
        blocking_strata = list(range(1, len(strata)))
        sampling_strata = [0]
    else:
        sampling_strata, blocking_strata = empirical_best_blocking_allocation(strata_population_size, strata_sample_sizes, strata_mean_vars)

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
    for strata_id, extra_size in zip(sampling_strata, extra_sample_size):
        if extra_size == 0:
            continue
        if config.aggregator in ["count", "sum"]:
            stratum_proxy_scores = proxy_scores[strata_population[strata_id]]
            stratum_proxy_weights = normalize(stratum_proxy_scores)
            extra_count_sample = weighted_sample_pd(stratum_proxy_scores, extra_size, replace=True)
        else:
            stratum_proxy_weights = []
            extra_count_sample = np.random.choice(strata_population[strata_id], extra_size, replace=True)
        extra_count_sample_ids = strata_population[strata_id][extra_count_sample]
        extra_count_sample_ids = np.array(np.unravel_index(extra_count_sample_ids, dataset_sizes)).T
        original_stratum_mean = np.mean(strata_sample_results[strata_id])
        for extra_count_s_id, extra_count_s in zip(extra_count_sample_ids, extra_count_sample):
            if oracle.query(extra_count_s_id):
                strata_sample_oracle_results[strata_id].append(1)
                total_n_positive += 1
                stats = dataset.get_statistics(extra_count_s_id)
                if config.aggregator == "count":
                    sample_result = 1 / len(stratum_proxy_weights) / stratum_proxy_weights[extra_count_s]
                elif config.aggregator == "sum":
                    sample_result = stats / len(stratum_proxy_weights) / stratum_proxy_weights[extra_count_s]
                elif config.aggregator == "avg":
                    sample_result = stats
                else:
                    raise ValueError(f"Unknown aggregator {config.aggregator}")
                strata_sample_results[strata_id].append(sample_result) 
            else:
                strata_sample_oracle_results[strata_id].append(0)
                strata_sample_results[strata_id].append(0)
        new_stratum_mean = np.mean(strata_sample_results[strata_id])
        logging.debug(f"after sampling more: stratum {strata_id}, original mean: {original_stratum_mean}, new mean: {new_stratum_mean}")            
    logging.debug(f"total number of positive data after sampling more {total_n_positive}")

    # run sampling with replacement for blocking strata
    for i in blocking_strata:
        blocking_population = strata_population[i]
        blocking_sample = np.random.choice(blocking_population, len(blocking_population), replace=True)
        blocking_sample_ids = np.array(np.unravel_index(blocking_sample, dataset_sizes)).T
        original_stratum_mean = np.mean(strata_sample_results[i])
        for blocking_s_id, blocking_s in zip(blocking_sample_ids, blocking_sample):
            stats = dataset.get_statistics(blocking_s_id)
            if oracle.query(blocking_s_id):
                strata_sample_oracle_results[i].append(1)
                total_n_positive += 1
                if config.aggregator == "count":
                    sample_result = 1
                elif config.aggregator == "sum":
                    sample_result = stats
                else:
                    sample_result = stats
                strata_sample_results[i].append(sample_result)
            else:
                strata_sample_oracle_results[i].append(0)
                strata_sample_results[i].append(0)
        new_stratum_mean = np.mean(strata_sample_results[i])
        logging.debug(f"after blocking sampling: stratum {i}, original {config.aggregator} mean: {original_stratum_mean}, new {config.aggregator} mean: {new_stratum_mean}")
    logging.debug(f"total number of positive data after blocking sampling {total_n_positive}")

    logging.debug(f"total sample size {[len(stratum_sample_result) for stratum_sample_result in strata_sample_results]}")
    logging.debug(f"strata gts: {get_gt_strata(config, strata_population, dataset_sizes, dataset, oracle)}")
    # get the estimation
    estimation = get_estimation(config, strata_population, strata_sample_results, strata_sample_oracle_results)
    
    logging.debug(f"{config.aggregator} estimation {estimation}")

    # run bootstrapping to get the confidence interval
    bootstrapping_estimations = []
    for i in range(config.bootstrap_trials):
        # resample
        strata_resample = []
        strata_resample_oracle_results = []
        for sample, sample_oracle_results in zip(strata_sample_results, strata_sample_oracle_results):
            resample_ids = np.random.choice(len(sample), len(sample), replace=True)
            resample = np.array(sample)[resample_ids]
            resample_oracle_results = np.array(sample_oracle_results)[resample_ids]
            strata_resample.append(resample)
            strata_resample_oracle_results.append(resample_oracle_results)
        # get the estimation
        bootstrapping_estimation = get_estimation(config, strata_population, strata_resample, strata_resample_oracle_results)
        bootstrapping_estimations.append(bootstrapping_estimation)

    # get the confidence interval
    lower, upper = get_ci_bootstrap(bootstrapping_estimations, config.confidence_level)

    est = Estimates(config.oracle_budget, gt, estimation, lower, upper)
    est.log()
    est.save(config.output_file, f"_{config.aggregator}")

def get_estimation(config, strata_population, strata_sample_results, strata_sample_oracle_results=None):
    if config.aggregator in ["count", "sum"]:
        assert len(strata_population) == len(strata_sample_results)
        estimation = 0
        strata_count = 0
        for stratum_sample_result, stratum_population in zip(strata_sample_results, strata_population):
            logging.debug(f"stratum {strata_count} {config.aggregator} result {np.mean(stratum_sample_result) * len(stratum_population)}")
            estimation += np.mean(stratum_sample_result) * len(stratum_population)
            strata_count += 1
    elif config.aggregator == "avg":
        assert strata_sample_oracle_results is not None and len(strata_sample_oracle_results) == len(strata_sample_results)
        estimation = 0
        positive_rates = []
        strata_sample_positive_results = []
        for stratum_sample_result, stratum_sample_oracle_result in zip(strata_sample_results, strata_sample_oracle_results):
            pr = sum(stratum_sample_oracle_result) / len(stratum_sample_result)
            positive_rates.append(pr)
            stratum_sample_positive_results = np.array(stratum_sample_result)[np.array(stratum_sample_oracle_result) == 1].tolist()
            strata_sample_positive_results.append(stratum_sample_positive_results)
            assert len(stratum_sample_positive_results) == sum(stratum_sample_oracle_result)
        assert len(positive_rates) == len(strata_sample_positive_results)
        for stratum_sample_positive_results, positive_rate in zip(strata_sample_positive_results, positive_rates):
            estimation += np.mean(stratum_sample_positive_results) * positive_rate
        estimation /= np.sum(positive_rates)
    else:
        raise ValueError(f"Unknown aggregator {config.aggregator}")
    return estimation

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
    if config.aggregator == "count":
        gt = count_gt
    elif config.aggregator == "sum":
        gt = sum_gt
    elif config.aggregator == "avg":
        gt = avg_gt
    else:
        raise ValueError(f"Unknown aggregator {config.aggregator}")

    logging.debug(f"count gt: {count_gt}, sum gt: {sum_gt}, avg gt: {avg_gt}")

    # get proxy
    proxy_scores = get_proxy_score(config, dataset)
    proxy_rank = get_proxy_rank(config, dataset, proxy_scores)
    logging.debug("finish proxy loading")
    proxy_sum = proxy_scores.sum()
    logging.debug(f"sum of proxy scores {proxy_sum}")

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
        strata_sample_sizes.append(max(1, int(proxy_scores[strata_population[i+1]].sum() / proxy_sum * sample_size)))
    logging.debug(f"{num_strata + 1} strata sample sizes: {strata_sample_sizes}")

    for exp_id in range(config.internal_loop):
        logging.info(f"running {exp_id} experiments")
        run_once(config, dataset, oracle, dataset_sizes, gt, proxy_scores, strata, strata_population, strata_sample_sizes)
