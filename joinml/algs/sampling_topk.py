from joinml.proxy.get_proxy import get_proxy_score, get_proxy_rank
from joinml.scalable_dataset_loader import ScalableJoinDataset, load_dataset
from joinml.oracle import Oracle
from joinml.config import Config
from joinml.utils import set_up_logging, normalize
from joinml.estimates import TopK
from joinml.utils import weighted_sample_pd, get_ci_bootstrap_ttest

import logging
import numpy as np
from typing import Tuple, List

def run(config: Config):
    set_up_logging(config.log_path, config.log_level)

    # log config
    logging.info(config)

    # dataset, oracle
    dataset = load_dataset(config)
    gt= dataset.get_gt(config.top_k)
    print(f"gt: {gt}")
    
    for _ in range(config.internal_loop):

        if config.task == "uniform-topk":
            sample_results = dataset.wander_join(config.oracle_budget)
        else:
            sample_results = dataset.sample(0, config.oracle_budget, replace=True)

        # print(f"sample results: {sample_results}")

        stats = []

        for fabric in sample_results:
            mean = np.mean(sample_results[fabric])
            std = np.std(sample_results[fabric], ddof=1)
            lb = mean - 1.96 * std / np.sqrt(len(sample_results[fabric]))
            ub = mean + 1.96 * std / np.sqrt(len(sample_results[fabric]))
            stats.append((fabric, mean, lb, ub))

        stats = sorted(stats, key=lambda x: x[1], reverse=True)
        topk_result = [stat[0] for stat in stats[:config.top_k]]
        biggest_lb_top5 = max([stat[2] for stat in stats[:5]])
        for stat in stats[5:]:
            if stat[3] < biggest_lb_top5:
                continue
            topk_result.append(stat[0])

        print(f"topk result: {topk_result}")

        est = TopK(config.oracle_budget, set(gt), set(topk_result))
        est.log()
        est.save(config.output_file, f"_topk")
