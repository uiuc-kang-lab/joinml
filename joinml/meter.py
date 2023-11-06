from joinml.dataset_loader import JoinDataset
from joinml.oracle import Oracle
from joinml.config import Config
import numpy as np
import logging
from scipy import stats

class ErrorMeter:
    def __init__(self, dataset: JoinDataset, oracle: Oracle, config: Config) -> None:
        if config.is_self_join:
            self.total_pairs = len(dataset.get_ids()[0]) * len(dataset.get_ids()[0])
        else:
            self.total_pairs = np.prod([len(ids) for ids in dataset.get_ids()])
        self.positive_pairs = len(oracle.oracle_labels)
        self.confidence_level = config.confidence_level
        self.count_results = []
        self.ci_uppers = []
        self.ci_lower = []
        self.errors = []
        logging.info(f"ErrorMeter reports groundtruth positive pairs: {self.positive_pairs}/{self.total_pairs}")

    def add_results(self, results: np.ndarray) -> None:
        mean = results.mean()
        ttest = stats.ttest_1samp(results, popmean=mean)
        ci = ttest.confidence_interval(confidence_level=self.confidence_level)
        ci_lower_bound = ci.low
        ci_upper_bound = ci.high
        count_result = mean * self.total_pairs
        count_upper = ci_upper_bound * self.total_pairs
        count_lower = ci_lower_bound * self.total_pairs
        count_error = np.abs(count_upper - self.positive_pairs) / self.positive_pairs
        self.count_results.append(count_result)
        self.ci_uppers.append(count_upper)
        self.ci_lower.append(count_lower)
        self.errors.append(count_error)
        running_error = np.abs(np.average(self.ci_uppers) - self.positive_pairs) / self.positive_pairs
        logging.info(f"ErrorMeter reports running error {running_error}")
    
    def report(self) -> None:
        logging.info(f"ErrorMeter reports count results {self.count_results}")
        logging.info(f"ErrorMeter reports count upper bounds {self.ci_uppers}")
        logging.info(f"ErrorMeter reports count lower bounds {self.ci_lower}")
        logging.info(f"ErrorMeter reports count errors {self.errors}")
        logging.info(f"ErrorMeter reports average count errors {100*np.average(self.errors)}%")
        logging.info(f"ErrorMeter reports count errors std {np.std(self.errors)}")

    def reset(self):
        self.count_results = []
        self.ci_uppers = []
        self.ci_lower = []
        self.errors = []

        