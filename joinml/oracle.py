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
        if raw_oracle_label_list[0][0] == "id1":
            raw_oracle_label_list = raw_oracle_label_list[1:]
        self.n_cols = len(raw_oracle_label_list[0])
        self.oracle_labels = set([tuple(row) for row in raw_oracle_label_list])
    
    def query(self, data: Tuple[int]):
        """Query the oracle for a data point."""
        data_str = tuple([str(int(d)) for d in data])
        return data_str in self.oracle_labels
    
    def get_positive_rate(self, rows):
        """Get the positive rate of the oracle."""
        return len(self.oracle_labels) / len(rows)

