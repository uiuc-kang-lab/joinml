import csv
from internal.table import Table
from internal.proxy import Proxy
import numpy as np
from typing import List
import logging
import sys

class Dataset(object):
    def __init__(self, tables: List[Table], proxy: Proxy, name: str, limit: int=-1) -> None:
        self.tables = tables
        self.proxy = proxy
        with open(f"datasets/{name}/positive_labels.csv") as f:
            reader = csv.reader(f)
            _ = next(reader)
            labels = set()
            for row in reader:
                satisfy_limit = True
                for i in row:
                    if int(i) >= limit and limit != -1:
                        satisfy_limit = False
                if satisfy_limit:
                    label = "|".join(row)
                    labels.add(label)
        self.labels = labels
        logging.info("groundtruth: {}".format(len(self.labels) / np.prod([len(table) for table in self.tables])))
    
    def evaluate_conditions(self, conditions: List[List[int]]) -> np.array:
        results = []
        for ids in conditions:
            ids = [str(Id) for Id in ids]
            label = "|".join(ids)
            if label in self.labels:
                results.append(1.)
            else:
                results.append(0.)
        return np.array(results)
    
    def get_proxy_matrix(self):
        return self.proxy.proxy_matrix

def load_dataset(dataset: str, proxy: str, limit: int=-1):
    if dataset == "qqp":
        proxy_file = f"proxy/qqp/{proxy}.npy"
        proxy_np = np.load(proxy_file)
        proxy_scores = Proxy(proxy_np, limit=limit)
        
        rows = []
        with open("datasets/qqp/quora_questions.csv") as f:
            reader = csv.reader(f)
            header = next(reader)
            qids = []
            for row in reader:
                qid, _, _ = row
                if int(qid) < limit:
                    rows.append(row)
        
        table = Table(rows)

        return Dataset([table, table], proxy_scores, dataset)
    elif dataset == "company":
        csv.field_size_limit(sys.maxsize)
        proxy_file = f"proxy/company/{proxy}.npy"
        proxy_np = np.load(proxy_file)
        proxy_scores = Proxy(proxy_np, limit=limit)

        rows = []
        with open("datasets/company/companyA.csv") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                Id, _, _ = row
                Id = int(Id)
                if Id < limit or limit == -1:
                    rows.append(row)
        tableA = Table(rows)

        rows = []
        with open("datasets/company/companyB.csv") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                Id, _, _ = row
                Id = int(Id)
                if Id < limit or limit == -1:
                    rows.append(row)
        tableB = Table(rows)
        return Dataset([tableA, tableB], proxy_scores, dataset, limit=limit)
    elif dataset == "AICity_vehicle":
        proxy_file = f"proxy/AICity_vehicle/{proxy}.npy"
        proxy_np = np.load(proxy_file)
        proxy_scores = Proxy(proxy_np, limit=limit)

        cameras = ["c001", "c002", "c003"]
        tables = []
        for camera in cameras:
            rows = []
            with open(f"datasets/AICity_vehicle/S01/{camera}/det/det_yolo3_id.csv") as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader:
                    Id = int(row[0])
                    if Id < limit or limit == -1:
                        rows.append(row)
            tables.append(Table(rows))
        return Dataset(tables, proxy_scores, dataset, limit=limit)