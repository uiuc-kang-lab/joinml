from joinml.proxy.get_proxy import get_proxy
from joinml.dataset_loader import JoinDataset
from joinml.oracle import Oracle
from joinml.config import Config
from joinml.utils import set_up_logging, normalize

import numpy as np
from scipy import stats
import logging

def run(config: Config):
    set_up_logging(config.log_path)

    proxy = get_proxy(config)
    dataset = JoinDataset(config)
    oracle = Oracle(config)

    dataset_ids = dataset.get_ids()
    join_cols = dataset.get_join_column_per_table(dataset_ids)
    if config.is_self_join:
        dataset_ids = [dataset_ids[0], dataset_ids[0]]
        join_cols = [join_cols[0], join_cols[0]]
    scores = proxy.get_proxy_score_for_tables(join_cols[0], join_cols[1])
    scores = normalize(scores, is_self_join=config.is_self_join)
    flattened_scores = scores.flatten()
    for cutoff in config.dataset_cutoff:
        for upper_sample_size in config.upper_sample_size:
            for lower_sample_size in config.lower_sample_size:
                logging.info(f"config: cutoff= {cutoff} upper sample size= {upper_sample_size} lower sample size= {lower_sample_size}")
                sorted_indexes = flattened_scores.argsort()
                upper_dataset = sorted_indexes[-cutoff:]
                lower_dataset = sorted_indexes[:-cutoff]
                lower_dataset_score = flattened_scores[lower_dataset]

                # get groundtruth
                upper_data_gt = 0
                for upper_data in upper_dataset:
                    upper_data_id = np.array(np.unravel_index(upper_data, scores.shape)).T
                    upper_data_id = [dataset_ids[0][upper_data_id[0]], dataset_ids[1][upper_data_id[1]]]
                    if oracle.query(upper_data_id):
                        upper_data_gt += 1
                lower_data_gt = len(oracle.oracle_labels) - upper_data_gt

                logging.info(f"groundtruth at the lower dataset: {lower_data_gt}")
                logging.info(f"groundtruth at the upper dataset: {upper_data_gt}")

                # random sample on the upper dataset
                upper_dataset_count_upperbounds = []
                for i in range(config.repeats):
                    upper_sample = np.random.choice(upper_dataset, size=upper_sample_size)
                    sample_ids = np.array(np.unravel_index(upper_sample, scores.shape)).T
                    results = []
                    for sample_id in sample_ids:
                        sample_id = [dataset_ids[0][sample_id[0]], dataset_ids[1][sample_id[1]]]
                        if oracle.query(sample_id):
                            results.append(1.)
                        else:
                            results.append(0.)
                    if sum(results) != 0:
                        ttest = stats.ttest_1samp(results, popmean=np.average(results))
                        ci_high = ttest.confidence_interval(confidence_level=config.confidence_level).high
                        count_upperbound = ci_high * cutoff
                        upper_dataset_count_upperbounds.append(count_upperbound)
                        error = (count_upperbound - upper_data_gt) / upper_data_gt
                        logging.info(f"uniformlly sample the upper dataset: error= {error*100}%")

                upper_dataset_count_upperbound_avg = np.average(upper_dataset_count_upperbounds) if len(upper_dataset_count_upperbounds) > 0 else 0
                upper_dataset_count_upperbound_std = np.std(upper_dataset_count_upperbounds) if len(upper_dataset_count_upperbounds) > 0 else 0
                if upper_data_gt != 0:
                    upper_dataset_error_avg = (upper_dataset_count_upperbound_avg - upper_data_gt) / upper_data_gt
                    upper_error_std = upper_dataset_count_upperbound_std / upper_data_gt
                else:
                    upper_dataset_error_avg = 0
                    upper_error_std = 0
                logging.info(f"upper dataset error= {upper_dataset_error_avg*100}% std= {upper_error_std}")

                # importance sampling on the lower dataset
                lower_data_count_upperbounds = []
                lower_dataset_score /= np.sum(lower_dataset_score)
                for i in range(config.repeats):
                    results = []
                    lower_samples = np.random.choice(len(lower_dataset_score), size=lower_sample_size, p=lower_dataset_score)
                    lower_sample_ids = lower_dataset[lower_samples]
                    lower_sample_table_ids = np.array(np.unravel_index(lower_sample_ids, scores.shape)).T
                    for lower_sample, lower_sample_table_id in zip(lower_samples, lower_sample_table_ids):
                        lower_sample_table_id = [dataset_ids[0][lower_sample_table_id[0]], dataset_ids[1][lower_sample_table_id[1]]]
                        if oracle.query(lower_sample_table_id):
                            results.append(1 / len(lower_dataset_score) / lower_dataset_score[lower_sample])
                        else:
                            results.append(0)
                    if sum(results) != 0:
                        ttest = stats.ttest_1samp(results, popmean=np.average(results))
                        ci_high = ttest.confidence_interval(confidence_level=config.confidence_level).high
                        count_upperbound = ci_high * len(lower_dataset_score)
                        lower_data_count_upperbounds.append(count_upperbound)
                        error = (count_upperbound - lower_data_gt) / lower_data_gt
                        logging.info(f"importance sampling on the lower dataset: error= {error*100}%")
                lower_dataset_count_upperbound_avg = np.average(lower_data_count_upperbounds) if len(lower_data_count_upperbounds) > 0 else 0
                lower_dataset_count_upperbound_std = np.std(lower_data_count_upperbounds) if len(lower_data_count_upperbounds) > 0 else 0
                if lower_data_gt != 0:
                    lower_dataset_error_avg = (lower_dataset_count_upperbound_avg - lower_data_gt) / lower_data_gt
                    lower_error_std = lower_dataset_count_upperbound_std / lower_data_gt
                else:
                    lower_dataset_error_avg = 0
                    lower_error_std = 0

                logging.info(f"lower dataset error: {lower_dataset_error_avg*100}% std= {lower_error_std}")
                overall_error = (upper_dataset_count_upperbound_avg + lower_dataset_count_upperbound_avg - lower_data_gt - upper_data_gt) / (lower_data_gt + upper_data_gt)
                logging.info(f"overall error {overall_error}")