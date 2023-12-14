from joinml.config import Config
from joinml.utils import read_csv
import glob
from typing import List, Tuple
import pandas as pd
from joinml.commons import dataset2modality


class TextDataset:
    def __init__(self, config: Config):
        self.tables = []
        table_files = glob.glob(f"{config.data_path}/{config.dataset_name}/data/table*.csv")
        for table_file in table_files:
            table_data = pd.read_csv(table_file)
            self.tables.append(table_data)
        self.dataset = config.dataset_name
        self.path = f"{config.data_path}/{config.dataset_name}"

    def get_sizes(self) -> Tuple[int, int]:
        return tuple(len(table) for table in self.tables)
    
    def get_join_column(self):
        join_column_per_table = []
        for table in self.tables:
            join_column_per_table.append(table["join_col"])
        return join_column_per_table

class VideoDataset:
    def __init__(self, config: Config):
        self.tables = []
        table_files = glob.glob(f"{config.data_path}/{config.dataset_name}/data/table*.csv")
        self.dataset = config.dataset_name
        self.path = f"{config.data_path}/{config.dataset_name}"
        for i, table_file in enumerate(table_files):
            table_data = pd.read_csv(table_file)
            table_data["img_path"] = table_data["id"].apply(lambda x: f"{self.path}/imgs/table{i}/{x}.jpg")
            self.tables.append(table_data)
    
    def get_sizes(self) -> Tuple[int, int]:
        return tuple(len(table) for table in self.tables)
    
    def get_join_column(self):
        join_column_per_table = []
        for table in self.tables:
            join_column_per_table.append(table["img_path"])
        return join_column_per_table

class MultiModalDataset:
    def __init__(self, config: Config):
        self.tables = []
        table_files = glob.glob(f"{config.data_path}/{config.dataset_name}/data/table*.csv")
        self.dataset = config.dataset_name
        self.path = f"{config.data_path}/{config.dataset_name}"
        for i, table_file in enumerate(table_files):
            table_data = pd.read_csv(table_file)
            table_data["img_path"] = table_data["id"].apply(lambda x: f"{self.path}/imgs/table{i}/{x}.jpg")
            self.tables.append(table_data)
    
    def get_sizes(self) -> Tuple[int, int]:
        return tuple(len(table) for table in self.tables)
    
    def get_join_column(self):
        join_column_per_table = []
        for table in self.tables:
            join_column_per_table.append(table[['img_path', 'join_col']])
        return join_column_per_table


def load_dataset(config: Config):
    modality = dataset2modality(config.dataset_name)
    if modality == "text":
        return TextDataset(config)
    elif modality == "images":
        return VideoDataset(config)
    elif modality == "multimodal":
        return MultiModalDataset(config)