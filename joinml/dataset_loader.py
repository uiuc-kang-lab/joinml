from joinml.config import Config
from joinml.utils import read_csv
import glob
from typing import List, Tuple, Any
import pandas as pd
from joinml.commons import dataset2modality
from joinml.oracle import Oracle
import os, json
import numpy as np

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
        cache_path = f"{self.path}/gt.json"

        def _compute_extremes():
            mn, mx = float('inf'), float('-inf')
            vals = []
            for row in oracle.oracle_labels:
                idx = [int(v) for v in list(row)[:2]]
                s = self.get_statistics(idx)
                if s > mx: mx = s
                if s < mn: mn = s
                vals.append(s)
            return mn, mx, float(np.median(vals)) if vals else 0.0

        # load cached groundtruth when available
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                gts = json.load(f)
            count_gt = gts["count_gt"]
            sum_gt = gts["sum_gt"]
            avg_gt = gts["avg_gt"]
            if all(k in gts for k in ("min_gt", "max_gt", "median_gt")):
                return count_gt, sum_gt, avg_gt, gts["min_gt"], gts["max_gt"], gts["median_gt"]
            min_gt, max_gt, median_gt = _compute_extremes()
            return count_gt, sum_gt, avg_gt, min_gt, max_gt, median_gt

        dataset_sizes = self.get_sizes()
        count_gt = 0
        sum_gt = 0
        max_gt = float('-inf')
        min_gt = float('inf')
        all_stats = []
        if len(dataset_sizes) <= 2:
            for (i, j) in oracle.oracle_labels:
                count_gt += 1
                statistics = self.get_statistics([int(i), int(j)])
                sum_gt += statistics
                if statistics > max_gt:
                    max_gt = statistics
                if statistics < min_gt:
                    min_gt = statistics
                all_stats.append(statistics)
            avg_gt = sum_gt / count_gt if count_gt else 0.0
        else:
            for row in oracle.oracle_labels:
                count_gt += 1
                idx = [int(v) for v in list(row)[:2]]
                statistics = self.get_statistics(idx)
                sum_gt += statistics
                if statistics > max_gt:
                    max_gt = statistics
                if statistics < min_gt:
                    min_gt = statistics
                all_stats.append(statistics)
            avg_gt = sum_gt / count_gt if count_gt else 0.0
        median_gt = float(np.median(all_stats)) if all_stats else 0.0
        return count_gt, sum_gt, avg_gt, min_gt, max_gt, median_gt

class ImageDataset(JoinDataset):
    def __init__(self, config: Config):
        self.tables = []
        table_files = glob.glob(f"{config.data_path}/{config.dataset_name}/data/table*.csv")
        table_files = sorted(table_files)
        for table_file in table_files:
            table_data = pd.read_csv(table_file)
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
        if self.dataset == "ecomm-q9":
            return abs(self.tables[0]["size"][table_ids[0]] - self.tables[0]["size"][table_ids[1]])
        else:
            return self.tables[1]["size"][table_ids[0]]

class TextDataset(JoinDataset):
    def __init__(self, config: Config):
        self.tables = []
        table_files = glob.glob(f"{config.data_path}/{config.dataset_name}/data/table*.csv")
        table_files = sorted(table_files)
        for table_file in table_files:
            table_data = pd.read_csv(table_file)
            # cast join_col to string
            table_data["join_col"] = table_data["join_col"].astype(str)
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
        id1, id2 = table_ids[:2]
        if self.dataset in ["quora", "quora_three"]:
            length1 = self.tables[0]["length"][id1]
            length2 = self.tables[0]["length"][id2]
            return abs(length1 - length2)
        elif self.dataset == "company":
            length1 = self.tables[0]["length"][id1]
            length2 = self.tables[1]["length"][id2]
            return length1 / length2
        elif self.dataset == "webmasters":
            answer_count = self.tables[0]["answer_count"][id1]
            return answer_count
        elif self.dataset in ["movie-q5", "movie-q6"]:
            score1 = self.tables[0]['score'][id1]
            score2 = self.tables[0]['score'][id2]
            return score1 + score2
        else:
            return 1

    def get_min_max_statistics(self):
        if self.dataset in ["quora", "quora_three"]:
            return self.tables[0]["length"].min() - self.tables[0]["length"].max(), \
                   self.tables[0]["length"].max() - self.tables[0]["length"].min()
        elif self.dataset == "company":
            return self.tables[0]["length"].min() / self.tables[0]["length"].max(), \
                   self.tables[0]["length"].max() / self.tables[0]["length"].min()
        elif self.dataset == "webmasters":
            return self.tables[0]["answer_count"].min(), self.tables[0]["answer_count"].max()
        elif self.dataset in ["movie-q5", "movie-q6"]:
            return self.tables[0]['score'].max() + self.tables[0]['score'].min()
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
        id1, id2 = table_ids[:2]
        if self.dataset in ["city_human", "city_human_three"]:
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
        if self.dataset == "ecomm-q8":
            return self.tables[1]["size"][table_ids[1]]
        else:
            return self.tables[0]["size"][table_ids[0]]

    def get_min_max_statistics(self):
        return 0, 1

class JoinOpDataset(JoinDataset):
    def __init__(self, config: Config):
        self.tables = []
        table_files = [f"{config.data_path}/{config.dataset_name}/data/table{i}.csv" for i in config.table_ids]
        table_files = sorted(table_files)
        for table_file in table_files:
            table_data = pd.read_csv(table_file)
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
        return 1

def load_dataset(config: Config):
    modality = dataset2modality[config.dataset_name] if not config.dataset_name.startswith("synthetic") else "text"
    if modality == "text":
        return TextDataset(config)
    elif modality == "image":
        return VideoDataset(config)
    elif modality == "multimodal":
        return MultiModalDataset(config)
    elif modality == "image_1":
        return ImageDataset(config)
    elif modality == "joinop":
        return JoinOpDataset(config)
    else:
        raise Exception(f"Modality {modality} not supported")
