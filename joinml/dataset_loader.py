from joinml.config import Config
from joinml.utils import read_csv
import glob
from typing import List, Tuple, Any
import pandas as pd
from joinml.commons import dataset2modality
from joinml.oracle import Oracle
import os, json

class JoinDataset:
    def __init__(self) -> None:
        self.tables = []
        self.path = ""

    def get_sizes(self) -> Tuple[int, ...]:
        return tuple(len(table) for table in self.tables)

    def get_join_column(self) -> List[Any]:
        return []

    def get_statistics(self, table_ids: List[int]) -> float:
        return 1

    def get_min_max_statistics(self)-> Tuple[float, float]:
        return 1, 1

    def get_gt(self, oracle: Oracle):
        # load the cached groundtruth to save time
        if os.path.exists(f"{self.path}/gt.json"):
            with open(f"{self.path}/gt.json") as f:
                gts = json.load(f)
            return gts["count_gt"], gts["sum_gt"], gts["avg_gt"]

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
        table_files = sorted(table_files)
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
            length1 = self.tables[0]["length"][id1]
            length2 = self.tables[0]["length"][id2]
            return abs(length1 - length2)
        elif self.dataset == "twitter":
            length1 = self.tables[0]["length"][id1]
            length2 = self.tables[0]["length"][id2]
            return length1 / length2
        elif self.dataset == "webmasters":
            answer_count = self.tables[0]["answer_count"][id1]
            return answer_count
        else:
            return 1

    def get_min_max_statistics(self):
        if self.dataset == "quora":
            return self.tables[0]["length"].min() - self.tables[0]["length"].max(), \
                   self.tables[0]["length"].max() - self.tables[0]["length"].min()
        elif self.dataset == "twitter":
            return self.tables[0]["length"].min() / self.tables[0]["length"].max(), \
                   self.tables[0]["length"].max() / self.tables[0]["length"].min()
        elif self.dataset == "webmasters":
            return self.tables[0]["answer_count"].min(), self.tables[0]["answer_count"].max()
        else:
            return 0, 1
        

class VideoDataset(JoinDataset):
    def __init__(self, config: Config):
        self.tables = []
        table_files = glob.glob(f"{config.data_path}/{config.dataset_name}/data/table*.csv")
        table_files = sorted(table_files)
        self.dataset = config.dataset_name
        self.path = f"{config.data_path}/{config.dataset_name}"
        for i, table_file in enumerate(table_files):
            table_data = pd.read_csv(table_file)
            if self.dataset == "VeRi":
                table_data["img_path"] = table_data["imageName"].apply(lambda x: f"{self.path}/imgs/table{i}/{x}")
            else:
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
        if self.dataset == "city_human":
            x_pos1 = self.tables[0]["x"][id1]
            x_pos2 = self.tables[1]["x"][id2]
            return abs(x_pos1 - x_pos2)
        elif self.dataset == "VeRi":
            timestamp1 = self.tables[0]["timestamp"][id1]
            timestamp2 = self.tables[1]["timestamp"][id2]
            return abs(timestamp1 - timestamp2)
        else:
            return 1

    def get_min_max_statistics(self):
        if self.dataset == "city_human":
            return self.tables[0]["x"].min() - self.tables[1]["x"].max(), self.tables[0]["x"].max() - self.tables[1]["x"].min()
        elif self.dataset == "VeRi":
            return self.tables[0]["timestamp"].min() - self.tables[1]["timestamp"].min(), self.tables[0]["timestamp"].max() - self.tables[1]["timestamp"].min()
        else:
            return 0, 1

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
    elif modality == "image":
        return VideoDataset(config)
    elif modality == "multimodal":
        return MultiModalDataset(config)
    else:
        raise Exception(f"Modality {modality} not supported")