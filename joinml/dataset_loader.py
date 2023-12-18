from joinml.config import Config
from joinml.utils import read_csv
import glob
from typing import List, Tuple, Any
import pandas as pd
from joinml.commons import dataset2modality
from joinml.oracle import Oracle

class JoinDataset:
    def __init__(self) -> None:
        self.tables = []

    def get_sizes(self) -> Tuple[int, ...]:
        return tuple(len(table) for table in self.tables)

    def get_join_column(self) -> List[Any]:
        return []

    def get_statistics(self, table_ids: List[int]) -> float:
        return 1

    def get_min_max_statistics(self)-> Tuple[float, float]:
        return 1, 1

    def get_gt(self, oracle: Oracle):
        dataset_sizes = self.get_sizes()
        count_gt = 0
        sum_gt = 0
        for (i, j) in oracle.oracle_labels:
            count_gt += 1
            sum_gt += self.get_statistics([int(i), int(j)])
        avg_gt = sum_gt / count_gt
        return count_gt, sum_gt, avg_gt
    

class TextDataset(JoinDataset):
    def __init__(self, config: Config):
        self.tables = []
        table_files = glob.glob(f"{config.data_path}/{config.dataset_name}/data/table*.csv")
        for table_file in table_files:
            table_data = pd.read_csv(table_file)
            table_data["length"] = table_data["join_col"].apply(lambda x: len(x.split(" ")))
            self.tables.append(table_data)
        self.dataset = config.dataset_name
        self.path = f"{config.data_path}/{config.dataset_name}"

    def get_sizes(self) -> Tuple[int, ...]:
        return tuple(len(table) for table in self.tables)
    
    def get_join_column(self):
        join_column_per_table = []
        for table in self.tables:
            join_column_per_table.append(table["join_col"])
        return join_column_per_table
    
    def get_statistics(self, table_ids: List[int]) -> float:
        id1, id2 = table_ids
        if self.dataset == "quora":
            length1 = self.tables[0]["length"][id1].split(" ")
            length2 = self.tables[0]["length"][id2].split(" ")
            return abs(length1 - length2)
        elif self.dataset == "twitter":
            length1 = self.tables[0]["length"][id1].split(" ")
            length2 = self.tables[0]["length"][id2].split(" ")
            return length1 / length2
        elif self.dataset == "stackoverflow":
            view_count = self.tables[0]["view_count"][id1]
            return view_count
        else:
            return 1

    def get_min_max_statistics(self):
        if self.dataset == "quora":
            return self.tables[0]["length"].min() - self.tables[0]["length"].max(), \
                   self.tables[0]["length"].max() - self.tables[0]["length"].min()
        elif self.dataset == "twitter":
            return self.tables[0]["length"].min() / self.tables[0]["length"].max(), \
                   self.tables[0]["length"].max() / self.tables[0]["length"].min()
        elif self.dataset == "stackoverflow":
            return self.tables[0]["view_count"].min(), self.tables[0]["view_count"].max()
        else:
            return 0, 1
        

class VideoDataset(JoinDataset):
    def __init__(self, config: Config):
        self.tables = []
        table_files = glob.glob(f"{config.data_path}/{config.dataset_name}/data/table*.csv")
        self.dataset = config.dataset_name
        self.path = f"{config.data_path}/{config.dataset_name}"
        for i, table_file in enumerate(table_files):
            table_data = pd.read_csv(table_file)
            table_data["img_path"] = table_data["id"].apply(lambda x: f"{self.path}/imgs/table{i}/{x}.jpg")
            self.tables.append(table_data)
    
    def get_sizes(self) -> Tuple[int, ...]:
        return tuple(len(table) for table in self.tables)
    
    def get_join_column(self):
        join_column_per_table = []
        for table in self.tables:
            join_column_per_table.append(table["img_path"])
        return join_column_per_table
    
    def get_statistics(self, table_ids: List[int]) -> float:
        id1, id2 = table_ids
        x_pos1 = self.tables[0]["x_pos"][id1]
        x_pos2 = self.tables[1]["x_pos"][id2]
        return abs(x_pos1 - x_pos2)

    def get_min_max_statistics(self):
        return self.tables[0]["x_pos"].min() - self.tables[1]["x_pos"].max(), self.tables[0]["x_pos"].max() - self.tables[1]["x_pos"].min()

class MultiModalDataset(JoinDataset):
    def __init__(self, config: Config):
        self.tables = []
        table_files = sorted(glob.glob(f"{config.data_path}/{config.dataset_name}/data/table*.csv"))
        self.dataset = config.dataset_name
        self.path = f"{config.data_path}/{config.dataset_name}"
        for table_file in table_files:
            table_data = pd.read_csv(table_file)
            if "join_col" not in table_data.columns:
                table_data["img_path"] = table_data["image_filename"].apply(lambda x: f"{self.path}/imgs/{x}")
            self.tables.append(table_data)
    
    def get_sizes(self) -> Tuple[int, ...]:
        return tuple(len(table) for table in self.tables)
    
    def get_join_column(self):
        join_column_per_table = []
        for table in self.tables:
            if "img_path" in table.columns:
                join_column_per_table.append(table['img_path'])
            else:
                join_column_per_table.append(table['join_col'])
        return join_column_per_table

    def get_statistics(self, table_ids: List[int]) -> float:
        return 1

    def get_min_max_statistics(self):
        return 0, 1

def load_dataset(config: Config):
    modality = dataset2modality[config.dataset_name]
    if modality == "text":
        return TextDataset(config)
    elif modality == "images":
        return VideoDataset(config)
    elif modality == "multimodal":
        return MultiModalDataset(config)
    else:
        raise Exception(f"Modality {modality} not supported")