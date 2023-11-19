from joinml.config import Config
from joinml.utils import read_csv
import glob
from typing import List, Tuple

class JoinDataset:
    def __init__(self, config: Config):
        self.tables = []
        table_files = glob.glob(f"{config.data_path}/{config.dataset_name}/data/table*.csv")
        for table_file in table_files:
            table_data = read_csv(table_file)
            self.tables.append(table_data)
        self.dataset = config.dataset_name
        self.path = f"{config.data_path}/{config.dataset_name}"

    def get_sizes(self) -> Tuple[int, int]:
        return tuple(len(table) for table in self.tables)
    
    def get_join_column(self):
        join_column_per_table = []
        if self.dataset == "city_vehicle_2":
            table_sizes = self.get_sizes()
            for i, table_size in enumerate(table_sizes):
                join_column = []
                for Id in range(table_size):
                    join_column.append(f"{self.path}/imgs/table{i}/{Id}.jpg")
                join_column_per_table.append(join_column)
        else:
            for table in self.tables:
                join_column = []
                for row in table:
                    join_column.append(row[-1])
                join_column_per_table.append(join_column)
        return join_column_per_table
