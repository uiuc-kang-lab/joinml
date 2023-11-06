from joinml.config import Config
from joinml.utils import read_csv
import glob
from typing import List

class JoinDataset:
    def __init__(self, config: Config):
        self.tables = []
        table_files = glob.glob(f"{config.data_path}/{config.dataset_name}/data/table*.csv")
        for table_file in table_files:
            table_data = read_csv(table_file)
            self.tables.append(table_data)
        id2join_col = []
        for table in self.tables:
            id2join_col_table = {}
            for row in table:
                id2join_col_table[int(row[0])] = row[-1]
            id2join_col.append(id2join_col_table)
        self.id2join_col = id2join_col
        self.dataset = config.dataset_name
        self.path = f"{config.data_path}/{config.dataset_name}"

        
    def get_sizes(self):
        return [len(table) for table in self.tables]
    
    def get_ids(self):
        return [list(id2_join_col_table.keys()) for id2_join_col_table in self.id2join_col]
    
    def get_join_column_per_table(self, ids: List[List[int]]):
        join_column_per_table = []
        if self.dataset in ["city_vehicle_2"]:
            for i, table_ids in enumerate(ids):
                join_column = []
                for Id in table_ids:
                    join_column.append(f"{self.path}/imgs/table{i}/{Id}.jpg")
                join_column_per_table.append(join_column)
        else:
            for id2join_col_table, table_ids in zip(self.id2join_col, ids):
                join_column = []
                for Id in table_ids:
                    join_column.append(id2join_col_table[Id])
                join_column_per_table.append(join_column)
        return join_column_per_table


# class Dataset(object):
#     def __init__(self, tables: List[Table], proxy: Proxy, name: str, limit: int=-1) -> None:
#         self.tables = tables
#         self.proxy = proxy
#         self.name = name
#         with open(f"datasets/{name}/positive_labels.csv") as f:
#             reader = csv.reader(f)
#             _ = next(reader)
#             labels = set()
#             for row in reader:
#                 satisfy_limit = True
#                 for i in row:
#                     if int(i) >= limit and limit != -1:
#                         satisfy_limit = False
#                 if satisfy_limit:
#                     label = "|".join(row)
#                     labels.add(label)
#         self.labels = labels
#         logging.info("groundtruth: {}".format(len(self.labels) / np.prod([len(table) for table in self.tables])))
    
#     def evaluate_conditions(self, conditions: List[List[int]]) -> np.array:
#         results = []
#         for ids in conditions:
#             ids = [str(Id) for Id in ids]
#             label = "|".join(ids)
#             if label in self.labels:
#                 results.append(1.)
#             else:
#                 results.append(0.)
#         return np.array(results)
    
#     def get_proxy_matrix(self):
#         return self.proxy.proxy_matrix

# def load_dataset(dataset: str, proxy: str, limit: int=-1):
#     if dataset == "qqp":
#         proxy_file = f"proxy/qqp/{proxy}.npy"
#         proxy_np = np.load(proxy_file)
#         proxy_scores = Proxy(proxy_np, limit=limit)
        
#         rows = []
#         with open("datasets/qqp/quora_questions.csv") as f:
#             reader = csv.reader(f)
#             header = next(reader)
#             qids = []
#             for row in reader:
#                 qid, _, _ = row
#                 if int(qid) < limit:
#                     rows.append(row)
        
#         table = Table(rows)

#         return Dataset([table, table], proxy_scores, dataset)
#     elif dataset == "company":
#         csv.field_size_limit(sys.maxsize)
#         proxy_file = f"proxy/company/{proxy}.npy"
#         proxy_np = np.load(proxy_file)
#         proxy_scores = Proxy(proxy_np, limit=limit)

#         rows = []
#         with open("datasets/company/companyA.csv") as f:
#             reader = csv.reader(f)
#             header = next(reader)
#             for row in reader:
#                 Id, _, _ = row
#                 Id = int(Id)
#                 if Id < limit or limit == -1:
#                     rows.append(row)
#         tableA = Table(rows)

#         rows = []
#         with open("datasets/company/companyB.csv") as f:
#             reader = csv.reader(f)
#             header = next(reader)
#             for row in reader:
#                 Id, _, _ = row
#                 Id = int(Id)
#                 if Id < limit or limit == -1:
#                     rows.append(row)
#         tableB = Table(rows)
#         return Dataset([tableA, tableB], proxy_scores, dataset, limit=limit)
#     elif dataset == "AICity_vehicle":
#         proxy_file = f"proxy/AICity_vehicle/{proxy}.npy"
#         proxy_np = np.load(proxy_file)
#         proxy_scores = Proxy(proxy_np, limit=limit)

#         cameras = ["c001", "c002", "c003"]
#         tables = []
#         for camera in cameras:
#             rows = []
#             with open(f"datasets/AICity_vehicle/S01/{camera}/det/det_yolo3_id.csv") as f:
#                 reader = csv.reader(f)
#                 header = next(reader)
#                 for row in reader:
#                     Id = int(row[0])
#                     if Id < limit or limit == -1:
#                         rows.append(row)
#             tables.append(Table(rows))
#         return Dataset(tables, proxy_scores, dataset, limit=limit)