from joinml.config import Config
from joinml.utils import read_csv
import glob
import logging
from typing import Tuple

class Oracle:
    def __init__(self, config: Config) -> None:
        # load oracle_labels
        oracle_label_files = glob.glob(f"{config.data_path}/{config.dataset_name}/oracle_labels/[0-9]*.csv")
        assert len(oracle_label_files) == 1, "There should be exactly one oracle label file."
        oracle_label_file = oracle_label_files[0]
        logging.info("Loading oracle labels from %s", oracle_label_file)
        raw_oracle_label_list = read_csv(oracle_label_file)
        self.oracle_labels = set([tuple(row) for row in raw_oracle_label_list])
        logging.info("find %d oracle (positive) labels", len(raw_oracle_label_list))
        self.cache = set()
    
    def query(self, data: Tuple[int]):
        """Query the oracle for a data point."""
        data_str = tuple([str(int(d)) for d in data])
        if data_str not in self.cache:
            self.cache.add(data_str)
        return data_str in self.oracle_labels
    
    def get_cost(self):
        """Get the cost of the oracle."""
        return len(self.cache)
    
    def get_positive_rate(self, rows):
        """Get the positive rate of the oracle."""
        return len([row for row in rows if self.query(row)]) / len(rows)

