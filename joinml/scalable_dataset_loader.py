from joinml.config import Config
from joinml.utils import read_csv
import glob
from typing import List, Tuple, Any
import pandas as pd
from joinml.commons import dataset2modality
from joinml.oracle import Oracle
import os, json
import numpy as np
import math
import pandas as pd
import time
from tqdm import tqdm
from collections import defaultdict

class ScalableJoinDataset:
    def __init__(self, config: Config, K: int=10) -> None:
        # load tables
        self.tables = []
        table0 = pd.read_csv(f"{config.data_path}/large-join/data/table0.csv")
        table1 = pd.read_csv(f"{config.data_path}/large-join/data/table1.csv")
        self.tables.append(pd.DataFrame({"id": range(len(table0)*len(table1))}))
        table2 = pd.read_csv(f"{config.data_path}/large-join/data/table2.csv")
        self.tables.append(pd.DataFrame({"id": range(len(table2))}))
        table3 = pd.read_csv(f"{config.data_path}/large-join/data/table3.csv")
        self.tables.append(pd.DataFrame({"id": range(len(table3))}))
        table4 = pd.read_csv(f"{config.data_path}/large-join/data/table4.csv")
        table5 = pd.read_csv(f"{config.data_path}/large-join/data/table5.csv")
        self.tables.append(pd.DataFrame({"id": range(len(table4)*len(table5))}))
        self.table_sizes = [len(table0), len(table1), len(table2), len(table3), len(table4), len(table5)]
        # load scores
        self.scores = []
        score_base_path = f"{config.cache_path}/large-join_synthetic_scores"
        self.scores.append(np.load(f"{score_base_path}_01_2.npy"))
        self.scores.append(np.load(f"{score_base_path}_2_3.npy"))
        self.scores.append(np.load(f"{score_base_path}_3_45.npy"))
        for i in range(len(self.scores)):
            self.scores[i] = self.scores[i] / self.scores[i].sum(axis=1).reshape(-1, 1)
        # load normalized scores
        self.oracle_labels = pd.read_csv(f"{config.data_path}/large-join/oracle_labels/012345.csv", header=None).values
        self.oracle_012 = set([int((row[0]*self.table_sizes[1] + row[1]) * self.table_sizes[2] + row[2]) for row in self.oracle_labels])
        self.oracle_labels = set([tuple([str(int(t)) for t in row]) for row in self.oracle_labels])
        # other parameters
        self.K = K
        self.blocking_ids = set()
        self.strata = []
        self.total_blocking_scores = 0

    def get_sizes(self) -> Tuple[int, ...]:
        return tuple(len(table) for table in self.tables)

    def get_join_column(self) -> List[Any]:
        return []

    def get_statistics(self, table_ids: List[int]) -> float:
        return 1

    def get_min_max_statistics(self)-> Tuple[float, float]:
        return 1, 1

    def stratify(self, max_blocking_size: int):
        n_max = max([len(table) for table in self.tables])
        top_k = int(math.pow(max_blocking_size / n_max, 1/(len(self.tables)-1))) + 1
        top_k_tables = []
        # 01_2
        similarity_scores = self.scores[0]
        top_k_mapping = np.argsort(similarity_scores, axis=0)[-top_k:, :]
        l_table_ids = []
        r_table_ids = []
        for r_id in range(top_k_mapping.shape[1]):
            for l_id in top_k_mapping[:, r_id]:
                l_table_ids.append(l_id)
                r_table_ids.append(r_id)
        top_k_tables.append(pd.DataFrame({
            "l_table_id": l_table_ids,
            "r_table_id": r_table_ids,
            "score_01": similarity_scores[l_table_ids, r_table_ids]
        }))
        # 2_3
        similarity_scores = self.scores[1]
        top_k_mapping = np.argsort(similarity_scores, axis=1)[:, -top_k:]
        l_table_ids = []
        r_table_ids = []
        for l_id in range(len(top_k_mapping)):
            for r_id in top_k_mapping[l_id]:
                l_table_ids.append(l_id)
                r_table_ids.append(r_id)
        top_k_tables.append(pd.DataFrame({
            "l_table_id": l_table_ids,
            "r_table_id": r_table_ids,
            "score_12": similarity_scores[l_table_ids, r_table_ids]
        }))
        # 3_45
        similarity_scores = self.scores[2]
        top_k_mapping = np.argsort(similarity_scores, axis=1)[:, -top_k:]
        l_table_ids = []
        r_table_ids = []
        for l_id in range(len(top_k_mapping)):
            for r_id in top_k_mapping[l_id]:
                l_table_ids.append(l_id)
                r_table_ids.append(r_id)
        top_k_tables.append(pd.DataFrame({
            "l_table_id": l_table_ids,
            "r_table_id": r_table_ids,
            "score_23": similarity_scores[l_table_ids, r_table_ids]
        }))

        # join the top k tables
        final_table = top_k_tables[0]
        final_table = final_table.rename(columns={"l_table_id": f"table_0",
                                                    "r_table_id": "table_1",
                                                    "score_01": "score"})
        for i in range(1, len(top_k_tables)):
            assert isinstance(final_table, pd.DataFrame)
            final_table = final_table.merge(top_k_tables[i],
                                            left_on=f"table_{i}",
                                            right_on="l_table_id",
                                            suffixes=("_old", ""))
            final_table = final_table.drop(columns=[
                "l_table_id"]).rename(columns={"r_table_id": f"table_{i+1}"})
            final_table["score"] = final_table["score"] * final_table[f"score_{i}{i+1}"]
        final_table = final_table.sort_values(by="score", ascending=False)
        final_table = final_table.head(max_blocking_size)
        print(final_table.head(20))
        self.total_blocking_scores = final_table["score"].sum()
        print(f"Total blocking scores: {self.total_blocking_scores}")
        # divide the final table into config.K parts by order
        stratum_size = max_blocking_size // self.K
        for i in range(self.K):
            if i != self.K-1:
                stratum = final_table.iloc[i*stratum_size: (i+1)*stratum_size]
            else:
                stratum = final_table.iloc[i*stratum_size:]
            self.strata.append(stratum)
        final_table_ids = final_table[[f"table_{i}" for i in range(len(self.tables))]].values.tolist()
        self.blocking_ids = set([tuple(row) for row in final_table_ids])

        return self.blocking_ids, self.strata

    def get_gt(self):
        return len(self.oracle_labels)

    def run_oracle(self, data: List[int]) -> bool:
        t0, t1 = np.unravel_index(data[0], shape=(self.table_sizes[0], self.table_sizes[1]))
        t2 = data[1]
        t3 = data[2]
        t4, t5 = np.unravel_index(data[3], shape=(self.table_sizes[4], self.table_sizes[5]))
        test = tuple([str(int(t)) for t in [t0, t1, t2, t3, t4, t5]])

        return test in self.oracle_labels

    def sample(self, stratum_id: int, sample_size: int, replace: bool):
        sample_results = []
        if stratum_id > 0:
            stratum = self.strata[stratum_id-1]
            assert isinstance(stratum, pd.DataFrame)
            weights = stratum["score"].values / stratum["score"].sum()
            sample_ids = np.random.choice(len(stratum), size=sample_size, replace=replace, p=weights)
            samples = stratum[["table_0", "table_1", "table_2", "table_3"]].iloc[sample_ids].values
            sample_weights = weights[sample_ids]
            for i, sample in enumerate(samples):
                if self.run_oracle(sample):
                    sample_results.append(1 / len(stratum) / sample_weights[i])
                else:
                    sample_results.append(0)
        elif stratum_id == 0:
            samples, sample_weights, sample_size = self.weighted_wander_join(sample_size)
            population_size = self.get_stratum_size(0)
            for i, sample in enumerate(samples):
                if self.run_oracle(sample):
                    sample_results.append(1 / population_size / sample_weights[i])
                else:
                    sample_results.append(0)
            for _ in range(sample_size - len(sample_results)):
                sample_results.append(0)
        return sample_results


    def weighted_wander_join(self, sample_size: int):
        start = time.time()
        reweight_factor = 1 / (1 - self.total_blocking_scores)
        weights = self.scores[0].flatten() / len(self.tables[0])
        init_samples = np.random.choice(len(self.tables[0]) * len(self.tables[1]), size=sample_size, replace=True, p=weights)
        print(f"done init sampling: sample size={len(init_samples)}")
        # only take sample that is in self.oracle_012, calculate it using numpy
        samples = init_samples[np.isin(init_samples, list(self.oracle_012))]
        print("done filtering")
        sample_weights = weights[samples] * reweight_factor
        print(f"# useful samples: {len(samples)}")
        samples = np.unravel_index(samples, shape=(self.table_sizes[0] * self.table_sizes[1], self.table_sizes[2]))
        samples = (np.array(samples).T).tolist()
        for i in tqdm(range(len(samples))):
            for j in range(1, len(self.scores)):
                weights = self.scores[j][samples[i][-1]]
                table_entry = np.random.choice(len(self.tables[j+1]), size=1, replace=True, p=weights).item()
                sample_weights[i] *= weights[table_entry]
                samples[i].append(table_entry)
        output_sample = []
        output_sample_weights = []
        for i, sample in enumerate(samples):
             if tuple(sample) not in self.blocking_ids:
                output_sample.append(sample)
                output_sample_weights.append(sample_weights[i])
        print(f"Time taken: {time.time()-start}")
        effective_sample_size = len(output_sample) + len(init_samples) - len(samples)

        return output_sample, output_sample_weights, effective_sample_size

    def get_stratum_size(self, stratum_id) -> int:
        if stratum_id == 0:
            return np.prod([len(table) for table in self.tables]).item() - len(self.blocking_ids)
        else:
            return len(self.strata[stratum_id-1])

    def wander_join(self, sample_size: int):
        start = time.time()
        init_samples = np.random.choice(len(self.tables[0]) * len(self.tables[1]), size=sample_size, replace=True)
        print(f"done init sampling: sample size={len(init_samples)}")
        # only take sample that is in self.oracle_012, calculate it using numpy
        samples = init_samples[np.isin(init_samples, list(self.oracle_012))]
        print("done filtering")
        print(f"# useful samples: {len(samples)}")
        samples = np.unravel_index(samples, shape=(self.table_sizes[0] * self.table_sizes[1], self.table_sizes[2]))
        samples = (np.array(samples).T).tolist()
        for i in tqdm(range(len(samples))):
            for j in range(1, len(self.scores)):
                table_entry = np.random.choice(len(self.tables[j+1]), size=1, replace=True).item()
                samples[i].append(table_entry)
        output_sample = []
        for i, sample in enumerate(samples):
             if tuple(sample) not in self.blocking_ids:
                output_sample.append(sample)
        results = []
        for sample in output_sample:
            if self.run_oracle(sample):
                results.append(1)
            else:
                results.append(0)
        print(f"Time taken: {time.time()-start}")
        effective_sample_size = len(output_sample) + len(init_samples) - len(samples)
        results = results + [0 for _ in range(effective_sample_size - len(output_sample))]
        return results

class ScalableJoinDatasetN3:
    def __init__(self, config: Config, K: int=10, table_ids: List[int]=[0,1,2]) -> None:
        # load tables
        table0 = pd.read_csv(f"{config.data_path}/{config.dataset_name}/data/table{table_ids[0]}.csv")
        table1 = pd.read_csv(f"{config.data_path}/{config.dataset_name}/data/table{table_ids[1]}.csv")
        table2 = pd.read_csv(f"{config.data_path}/{config.dataset_name}/data/table{table_ids[2]}.csv")
        self.tables = [table0, table1, table2]
        self.table_sizes = [len(table0), len(table1), len(table2)]
    # load scores
        self.scores = []
        score_base_path = f"{config.cache_path}/{config.dataset_name}"
        self.original_scores_01 = np.load(f"{score_base_path}_{table_ids[0]}{table_ids[1]}.npy")
        self.original_scores_12 = np.load(f"{score_base_path}_{table_ids[1]}{table_ids[2]}.npy")
        self.scores.append(np.load(f"{score_base_path}_{table_ids[0]}{table_ids[1]}.npy"))
        self.scores.append(np.load(f"{score_base_path}_{table_ids[1]}{table_ids[2]}.npy"))
        for i in range(len(self.scores)):
            self.scores[i] = self.scores[i] / self.scores[i].sum(axis=1).reshape(-1, 1)
            print(f"shape of score {i}: {self.scores[i].shape}")
        self.oracle_labels = pd.read_csv(f"{config.data_path}/{config.dataset_name}/oracle_labels/{''.join([str(table_id) for table_id in table_ids])}.csv", header=None).values
        self.oracle_01 = [int((row[0]*self.table_sizes[1] + row[1])) for row in self.oracle_labels]
        self.oracle_01 = set(self.oracle_01)
        self.oracle_labels = set([tuple([str(int(t)) for t in row]) for row in self.oracle_labels])
        # other parameters
        self.K = K
        self.blocking_ids = set()
        self.strata = []
        self.total_blocking_scores = 0
        self.config = config

    def get_sizes(self) -> Tuple[int, ...]:
        return tuple(len(table) for table in self.tables)

    def get_join_column(self) -> List[Any]:
        return []

    def get_statistics(self, table_ids: List[int]) -> float:
        return 1

    def get_min_max_statistics(self)-> Tuple[float, float]:
        return 1, 1

    def stratify(self, max_blocking_size: int):
        n_max = max([len(table) for table in self.tables])
        top_k = 50 #int(math.pow(max_blocking_size / n_max, 1/(len(self.tables)-1))) + 1
        top_k_tables = []
        # 01
        similarity_scores = self.scores[0]
        top_k_mapping = np.argsort(similarity_scores, axis=0)[-top_k:, :]
        l_table_ids = []
        r_table_ids = []
        for r_id in range(top_k_mapping.shape[1]):
            for l_id in top_k_mapping[:, r_id]:
                l_table_ids.append(l_id)
                r_table_ids.append(r_id)
        top_k_tables.append(pd.DataFrame({
            "l_table_id": l_table_ids,
            "r_table_id": r_table_ids,
            "score_01": similarity_scores[l_table_ids, r_table_ids]
        }))

        # 12
        similarity_scores = self.scores[1]
        top_k_mapping = np.argsort(similarity_scores, axis=1)[:, -top_k:]
        l_table_ids = []
        r_table_ids = []
        for l_id in range(len(top_k_mapping)):
            for r_id in top_k_mapping[l_id]:
                l_table_ids.append(l_id)
                r_table_ids.append(r_id)
        top_k_tables.append(pd.DataFrame({
            "l_table_id": l_table_ids,
            "r_table_id": r_table_ids,
            "score_12": similarity_scores[l_table_ids, r_table_ids]
        }))

        # join the top k tables
        final_table = top_k_tables[0]
        final_table = final_table.rename(columns={"l_table_id": f"table_0",
                                                    "r_table_id": "table_1",
                                                    "score_01": "score"})
        for i in range(1, len(top_k_tables)):
            assert isinstance(final_table, pd.DataFrame)
            final_table = final_table.merge(top_k_tables[i],
                                            left_on=f"table_{i}",
                                            right_on="l_table_id",
                                            suffixes=("_old", ""))
            final_table = final_table.drop(columns=[
                "l_table_id"]).rename(columns={"r_table_id": f"table_{i+1}"})
            final_table["score"] = final_table["score"] * final_table[f"score_{i}{i+1}"]
        final_table = final_table.sort_values(by="score", ascending=False)
        final_table = final_table.head(max_blocking_size)
        print(len(final_table), max_blocking_size)
        print(final_table.head(20))
        self.total_blocking_scores = final_table["score"].sum()
        print(f"Total blocking scores: {self.total_blocking_scores}")

        # divide the final table into config.K parts by order
        stratum_size = max_blocking_size // self.K
        for i in range(self.K):
            if i != self.K-1:
                stratum = final_table.iloc[i*stratum_size: (i+1)*stratum_size]
            else:
                stratum = final_table.iloc[i*stratum_size:]
            self.strata.append(stratum)
        final_table_ids = final_table[[f"table_{i}" for i in range(len(self.tables))]].values.tolist()
        self.blocking_ids = set([tuple(row) for row in final_table_ids])

        return self.blocking_ids, self.strata

    def get_gt(self):
        return len(self.oracle_labels)

    def run_oracle(self, data: List[int]) -> bool:
        test = tuple([str(int(t)) for t in data])
        return tuple(test) in self.oracle_labels

    def sample(self, stratum_id: int, sample_size: int, replace: bool):
        sample_results = []
        if stratum_id > 0:
            stratum = self.strata[stratum_id-1]
            assert isinstance(stratum, pd.DataFrame)
            weights = stratum["score"].values / stratum["score"].sum()
            sample_ids = np.random.choice(len(stratum), size=sample_size, replace=replace, p=weights)
            samples = stratum[["table_0", "table_1", "table_2"]].iloc[sample_ids].values
            sample_weights = weights[sample_ids]
            for i, sample in enumerate(samples):
                if self.run_oracle(sample):
                    sample_results.append(1 / len(stratum) / sample_weights[i])
                else:
                    sample_results.append(0)
        elif stratum_id == 0:
            samples, sample_weights, sample_size = self.weighted_wander_join(sample_size)
            population_size = np.prod(self.get_sizes())
            for i, sample in enumerate(samples):
                if self.run_oracle(sample):
                    sample_results.append(1 / population_size / sample_weights[i])
                else:
                    sample_results.append(0)
            for _ in range(sample_size - len(sample_results)):
                sample_results.append(0)
        return sample_results


    def weighted_wander_join(self, sample_size: int):
        start = time.time()
        reweight_factor = 1 / (1 - self.total_blocking_scores)
        weights = self.scores[0].flatten() / len(self.tables[0])
        init_samples = np.random.choice(len(self.tables[0]) * len(self.tables[1]), size=sample_size, replace=True, p=weights)
        print(f"done init sampling: sample size={len(init_samples)}/{len(self.tables[0]) * len(self.tables[1])}")
        # only take sample that is in self.oracle_012, calculate it using numpy
        samples = init_samples[np.isin(init_samples, list(self.oracle_01))]
        print("done filtering")
        sample_weights = weights[samples] * reweight_factor
        print(f"# useful samples: {len(samples)}")
        samples = np.unravel_index(samples, shape=(self.table_sizes[0], self.table_sizes[1]))
        samples = (np.array(samples).T).tolist()
        for i in tqdm(range(len(samples))):
            for j in range(1, len(self.scores)):
                weights = self.scores[j][samples[i][-1]]
                table_entry = np.random.choice(len(self.tables[j+1]), size=1, replace=True, p=weights).item()
                sample_weights[i] *= weights[table_entry]
                samples[i].append(table_entry)
        output_sample = []
        output_sample_weights = []
        for i, sample in enumerate(samples):
             if tuple(sample) not in self.blocking_ids:
                output_sample.append(sample)
                output_sample_weights.append(sample_weights[i])
        print(f"Time taken: {time.time()-start}")
        effective_sample_size = len(output_sample) + len(init_samples) - len(samples)

        return output_sample, output_sample_weights, effective_sample_size

    def get_stratum_size(self, stratum_id) -> int:
        if stratum_id == 0:
            return np.prod([len(table) for table in self.tables]).item() - len(self.blocking_ids)
        else:
            return len(self.strata[stratum_id-1])

    def wander_join(self, sample_size: int):
        start = time.time()
        init_samples = np.random.choice(len(self.tables[0]) * len(self.tables[1]), size=sample_size, replace=True)
        print("done init sampling")
        samples = init_samples[np.isin(init_samples, list(self.oracle_01))]
        print("done filtering")
        print(f"# useful samples: {len(samples)}")
        samples = np.unravel_index(samples, shape=(self.table_sizes[0], self.table_sizes[1]))
        samples = (np.array(samples).T).tolist()
        for i in tqdm(range(len(samples))):
            for j in range(1, len(self.scores)):
                table_entry = np.random.choice(len(self.tables[j+1]), size=1, replace=True).item()
                samples[i].append(table_entry)
        output_sample = []
        for i, sample in enumerate(samples):
             if tuple(sample) not in self.blocking_ids:
                output_sample.append(sample)
        results = []
        for sample in output_sample:
            if self.run_oracle(sample):
                results.append(1)
            else:
                results.append(0)
        print(f"Time taken: {time.time()-start}")
        effective_sample_size = len(output_sample) + len(init_samples) - len(samples)
        results = results + [0 for _ in range(effective_sample_size - len(output_sample))]
        return results

    def wander_join_blocking(self, sample_size: int):
        start = time.time()
        init_samples = np.random.choice(len(self.tables[0]) * len(self.tables[1]), size=sample_size, replace=True)
        print("done init sampling")
        samples = init_samples[np.isin(init_samples, list(self.oracle_01))]
        print("done filtering")
        print(f"# useful samples: {len(samples)}")
        samples = np.unravel_index(samples, shape=(self.table_sizes[0], self.table_sizes[1]))
        samples = (np.array(samples).T).tolist()
        with open("block_noci/valid_threshold.json") as f:
            valid_threshold = json.load(f)
        valid_threshold_01 = valid_threshold[f"{self.config.dataset_name}_01" if self.config.dataset_name.count('-')<2 else f"{self.config.dataset_name.rsplit('-', 1)[0]}_01"]
        valid_threshold_12 = valid_threshold[f"{self.config.dataset_name}_12" if self.config.dataset_name.count('-')<2 else f"{self.config.dataset_name.rsplit('-', 1)[0]}_12"]
        for i in tqdm(range(len(samples))):
            for j in range(1, len(self.scores)):
                table_entry = np.random.choice(len(self.tables[j+1]), size=1, replace=True).item()
                samples[i].append(table_entry)
        output_sample = []
        for i, sample in enumerate(samples):
             if tuple(sample) not in self.blocking_ids and self.original_scores_01[sample[0], sample[1]] >= valid_threshold_01 and self.original_scores_12[sample[1], sample[2]] >= valid_threshold_12:
                output_sample.append(sample)
        results = []
        for sample in output_sample:
            if self.run_oracle(sample):
                results.append(1)
            else:
                results.append(0)
        print(f"Time taken: {time.time()-start}")
        effective_sample_size = len(output_sample) + len(init_samples) - len(samples)
        results = results + [0 for _ in range(effective_sample_size - len(output_sample))]
        return results

class ScalableJoinDatasetN4:
    def __init__(self, config: Config, K: int=10, table_ids: List[int]=[0,1,2,3]) -> None:
        # load tables
        table0 = pd.read_csv(f"{config.data_path}/{config.dataset_name}/data/table{table_ids[0]}.csv")
        table1 = pd.read_csv(f"{config.data_path}/{config.dataset_name}/data/table{table_ids[1]}.csv")
        table2 = pd.read_csv(f"{config.data_path}/{config.dataset_name}/data/table{table_ids[2]}.csv")
        table3 = pd.read_csv(f"{config.data_path}/{config.dataset_name}/data/table{table_ids[3]}.csv")
        self.tables = [table0, table1, table2, table3]
        self.table_sizes = [len(table0), len(table1), len(table2), len(table3)]
    # load scores
        self.scores = []
        score_base_path = f"{config.cache_path}/{config.dataset_name}"
        self.original_scores_01 = np.load(f"{score_base_path}_{table_ids[0]}{table_ids[1]}.npy")
        self.original_scores_12 = np.load(f"{score_base_path}_{table_ids[1]}{table_ids[2]}.npy")
        self.original_scores_23 = np.load(f"{score_base_path}_{table_ids[2]}{table_ids[3]}.npy")
        self.scores.append(np.load(f"{score_base_path}_01.npy"))
        self.scores.append(np.load(f"{score_base_path}_12.npy"))
        self.scores.append(np.load(f"{score_base_path}_23.npy"))
        for i in range(len(self.scores)):
            self.scores[i] = self.scores[i] / self.scores[i].sum(axis=1).reshape(-1, 1)
            print(f"shape of score {i}: {self.scores[i].shape}")
        self.oracle_labels = pd.read_csv(f"{config.data_path}/{config.dataset_name}/oracle_labels/{''.join([str(table_id) for table_id in table_ids])}.csv", header=None).values
        self.oracle_01 = [int((row[0]*self.table_sizes[1] + row[1])) for row in self.oracle_labels]
        self.oracle_01 = set(self.oracle_01)
        self.oracle_labels = set([tuple([str(int(t)) for t in row]) for row in self.oracle_labels])
        # other parameters
        self.K = K
        self.blocking_ids = set()
        self.strata = []
        self.total_blocking_scores = 0
        self.config = config

    def get_sizes(self) -> Tuple[int, ...]:
        return tuple(len(table) for table in self.tables)

    def get_join_column(self) -> List[Any]:
        return []

    def get_statistics(self, table_ids: List[int]) -> float:
        return 1

    def get_min_max_statistics(self)-> Tuple[float, float]:
        return 1, 1

    def stratify(self, max_blocking_size: int):
        n_max = max([len(table) for table in self.tables])
        top_k = 50 # int(math.pow(max_blocking_size / n_max, 1/(len(self.tables)-1))) + 1
        print("topK:", top_k)
        top_k_tables = []
        # 01
        similarity_scores = self.scores[0]
        top_k_mapping = np.argsort(similarity_scores, axis=0)[-top_k:, :]
        l_table_ids = []
        r_table_ids = []
        for r_id in range(top_k_mapping.shape[1]):
            for l_id in top_k_mapping[:, r_id]:
                l_table_ids.append(l_id)
                r_table_ids.append(r_id)
        top_k_tables.append(pd.DataFrame({
            "l_table_id": l_table_ids,
            "r_table_id": r_table_ids,
            "score_01": similarity_scores[l_table_ids, r_table_ids]
        }))

        # 12
        similarity_scores = self.scores[1]
        top_k_mapping = np.argsort(similarity_scores, axis=1)[:, -top_k:]
        l_table_ids = []
        r_table_ids = []
        for l_id in range(len(top_k_mapping)):
            for r_id in top_k_mapping[l_id]:
                l_table_ids.append(l_id)
                r_table_ids.append(r_id)
        top_k_tables.append(pd.DataFrame({
            "l_table_id": l_table_ids,
            "r_table_id": r_table_ids,
            "score_12": similarity_scores[l_table_ids, r_table_ids]
        }))

        # 23
        similarity_scores = self.scores[2]
        top_k_mapping = np.argsort(similarity_scores, axis=1)[:, -top_k:]
        l_table_ids = []
        r_table_ids = []
        for l_id in range(len(top_k_mapping)):
            for r_id in top_k_mapping[l_id]:
                l_table_ids.append(l_id)
                r_table_ids.append(r_id)
        top_k_tables.append(pd.DataFrame({
            "l_table_id": l_table_ids,
            "r_table_id": r_table_ids,
            "score_23": similarity_scores[l_table_ids, r_table_ids]
        }))

        # join the top k tables
        final_table = top_k_tables[0]
        final_table = final_table.rename(columns={"l_table_id": f"table_0",
                                                    "r_table_id": "table_1",
                                                    "score_01": "score"})
        for i in range(1, len(top_k_tables)):
            assert isinstance(final_table, pd.DataFrame)
            final_table = final_table.merge(top_k_tables[i],
                                            left_on=f"table_{i}",
                                            right_on="l_table_id",
                                            suffixes=("_old", ""))
            final_table = final_table.drop(columns=[
                "l_table_id"]).rename(columns={"r_table_id": f"table_{i+1}"})
            final_table["score"] = final_table["score"] * final_table[f"score_{i}{i+1}"]

        final_table = final_table.sort_values(by="score", ascending=False)
        final_table = final_table.head(max_blocking_size)

        self.total_blocking_scores = final_table["score"].sum()

        # divide the final table into config.K parts by order
        stratum_size = max_blocking_size // self.K
        for i in range(self.K):
            if i != self.K-1:
                stratum = final_table.iloc[i*stratum_size: (i+1)*stratum_size]
            else:
                stratum = final_table.iloc[i*stratum_size:]
            self.strata.append(stratum)
        final_table_ids = final_table[[f"table_{i}" for i in range(len(self.tables))]].values.tolist()
        self.blocking_ids = set([tuple(row) for row in final_table_ids])

        return self.blocking_ids, self.strata

    def get_gt(self):
        return len(self.oracle_labels)

    def run_oracle(self, data: List[int]) -> bool:
        test = tuple([str(int(t)) for t in data])
        return tuple(test) in self.oracle_labels

    def sample(self, stratum_id: int, sample_size: int, replace: bool):
        sample_results = []
        if stratum_id > 0:
            stratum = self.strata[stratum_id-1]
            assert isinstance(stratum, pd.DataFrame)
            weights = stratum["score"].values / stratum["score"].sum()
            sample_ids = np.random.choice(len(stratum), size=sample_size, replace=replace, p=weights)
            samples = stratum[["table_0", "table_1", "table_2", "table_3"]].iloc[sample_ids].values
            sample_weights = weights[sample_ids]
            for i, sample in enumerate(samples):
                if self.run_oracle(sample):
                    sample_results.append(1 / len(stratum) / sample_weights[i])
                else:
                    sample_results.append(0)
        elif stratum_id == 0:
            samples, sample_weights, sample_size = self.weighted_wander_join(sample_size)
            population_size = np.prod(self.get_sizes())
            for i, sample in enumerate(samples):
                if self.run_oracle(sample):
                    sample_results.append(1 / population_size / sample_weights[i])
                else:
                    sample_results.append(0)
            for _ in range(sample_size - len(sample_results)):
                sample_results.append(0)
        return sample_results


    def weighted_wander_join(self, sample_size: int):
        start = time.time()
        reweight_factor = 1 / (1 - self.total_blocking_scores)
        weights = self.scores[0].flatten() / len(self.tables[0])
        init_samples = np.random.choice(len(self.tables[0]) * len(self.tables[1]), size=sample_size, replace=True, p=weights)
        print(f"done init sampling: sample size={len(init_samples)}/{len(self.tables[0]) * len(self.tables[1])}")
        # only take sample that is in self.oracle_012, calculate it using numpy
        samples = init_samples[np.isin(init_samples, list(self.oracle_01))]
        print("done filtering")
        sample_weights = weights[samples] * reweight_factor
        print(f"# useful samples: {len(samples)}")
        samples = np.unravel_index(samples, shape=(self.table_sizes[0], self.table_sizes[1]))
        samples = (np.array(samples).T).tolist()
        for i in tqdm(range(len(samples))):
            for j in range(1, len(self.scores)):
                weights = self.scores[j][samples[i][-1]]
                table_entry = np.random.choice(len(self.tables[j+1]), size=1, replace=True, p=weights).item()
                sample_weights[i] *= weights[table_entry]
                samples[i].append(table_entry)
        output_sample = []
        output_sample_weights = []
        for i, sample in enumerate(samples):
             if tuple(sample) not in self.blocking_ids:
                output_sample.append(sample)
                output_sample_weights.append(sample_weights[i])
        print(f"Time taken: {time.time()-start}")
        effective_sample_size = len(output_sample) + len(init_samples) - len(samples)

        return output_sample, output_sample_weights, effective_sample_size

    def get_stratum_size(self, stratum_id) -> int:
        if stratum_id == 0:
            return np.prod([len(table) for table in self.tables]).item() - len(self.blocking_ids)
        else:
            return len(self.strata[stratum_id-1])

    def wander_join(self, sample_size: int):
        start = time.time()
        init_samples = np.random.choice(len(self.tables[0]) * len(self.tables[1]), size=sample_size, replace=True)
        print("done init sampling")
        samples = init_samples[np.isin(init_samples, list(self.oracle_01))]
        print("done filtering")
        print(f"# useful samples: {len(samples)}")
        samples = np.unravel_index(samples, shape=(self.table_sizes[0], self.table_sizes[1]))
        samples = (np.array(samples).T).tolist()
        for i in tqdm(range(len(samples))):
            for j in range(1, len(self.scores)):
                table_entry = np.random.choice(len(self.tables[j+1]), size=1, replace=True).item()
                samples[i].append(table_entry)
        output_sample = []
        for i, sample in enumerate(samples):
             if tuple(sample) not in self.blocking_ids:
                output_sample.append(sample)
        results = []
        for sample in output_sample:
            if self.run_oracle(sample):
                results.append(1)
            else:
                results.append(0)
        print(f"Time taken: {time.time()-start}")
        effective_sample_size = len(output_sample) + len(init_samples) - len(samples)
        results = results + [0 for _ in range(effective_sample_size - len(output_sample))]
        return results

    def wander_join_blocking(self, sample_size: int):
        start = time.time()
        init_samples = np.random.choice(len(self.tables[0]) * len(self.tables[1]), size=sample_size, replace=True)
        print("done init sampling")
        samples = init_samples[np.isin(init_samples, list(self.oracle_01))]
        print("done filtering")
        print(f"# useful samples: {len(samples)}")
        samples = np.unravel_index(samples, shape=(self.table_sizes[0], self.table_sizes[1]))
        samples = (np.array(samples).T).tolist()
        with open("block_noci/valid_threshold.json") as f:
            valid_threshold = json.load(f)
        valid_threshold_01 = valid_threshold[f"{self.config.dataset_name}_01"]
        valid_threshold_12 = valid_threshold[f"{self.config.dataset_name}_12"]
        valid_threshold_23 = valid_threshold[f"{self.config.dataset_name}_23"]
        for i in tqdm(range(len(samples))):
            for j in range(1, len(self.scores)):
                table_entry = np.random.choice(len(self.tables[j+1]), size=1, replace=True).item()
                samples[i].append(table_entry)
        output_sample = []
        for i, sample in enumerate(samples):
             if tuple(sample) not in self.blocking_ids and self.original_scores_01[sample[0], sample[1]] >= valid_threshold_01 and self.original_scores_12[sample[1], sample[2]] >= valid_threshold_12 and self.original_scores_23[sample[2], sample[3]] >= valid_threshold_23:
                output_sample.append(sample)
        results = []
        for sample in output_sample:
            if self.run_oracle(sample):
                results.append(1)
            else:
                results.append(0)
        print(f"Time taken: {time.time()-start}")
        effective_sample_size = len(output_sample) + len(init_samples) - len(samples)
        results = results + [0 for _ in range(effective_sample_size - len(output_sample))]
        return results

class ScalableJoinDatasetN4TopK:
    def __init__(self, config: Config, K: int=10, table_ids: List[int]=[0,1,2,3]) -> None:
        # load tables
        table0 = pd.read_csv(f"{config.data_path}/{config.dataset_name}/data/table{table_ids[0]}.csv")
        table1 = pd.read_csv(f"{config.data_path}/{config.dataset_name}/data/table{table_ids[1]}.csv")
        table2 = pd.read_csv(f"{config.data_path}/{config.dataset_name}/data/table{table_ids[2]}.csv")
        table3 = pd.read_csv(f"{config.data_path}/{config.dataset_name}/data/table{table_ids[3]}.csv")
        self.tables = [table0, table1, table2, table3]
        self.table_sizes = [len(table0), len(table1), len(table2), len(table3)]
    # load scores
        self.scores = []
        score_base_path = f"{config.cache_path}/{config.dataset_name.strip('-topk')}"
        self.original_scores_01 = np.load(f"{score_base_path}_{table_ids[0]}{table_ids[1]}.npy")
        self.original_scores_12 = np.load(f"{score_base_path}_{table_ids[1]}{table_ids[2]}.npy")
        self.original_scores_23 = np.load(f"{score_base_path}_{table_ids[2]}{table_ids[3]}.npy")
        self.scores.append(np.load(f"{score_base_path}_01.npy"))
        self.scores.append(np.load(f"{score_base_path}_12.npy"))
        self.scores.append(np.load(f"{score_base_path}_23.npy"))
        for i in range(len(self.scores)):
            self.scores[i] = self.scores[i] / self.scores[i].sum(axis=1).reshape(-1, 1)
            print(f"shape of score {i}: {self.scores[i].shape}")
        self.oracle_labels = pd.read_csv(f"{config.data_path}/{config.dataset_name}/oracle_labels/{''.join([str(table_id) for table_id in table_ids])}.csv", header=None).values
        self.oracle_01 = [int((row[0]*self.table_sizes[1] + row[1])) for row in self.oracle_labels]
        self.oracle_01 = set(self.oracle_01)
        self.oracle_labels = set([tuple([str(int(t)) for t in row]) for row in self.oracle_labels])
        # other parameters
        self.K = K
        self.blocking_ids = set()
        self.strata = []
        self.total_blocking_scores = 0
        self.config = config
        self.all_fabrics = ['Cotton', 'Unknown', 'Polyester', 'Blended', 'Synthetic', 'Viscose Rayon', 'Nylon']

    def get_sizes(self) -> Tuple[int, ...]:
        return tuple(len(table) for table in self.tables)

    def get_join_column(self) -> List[Any]:
        return []

    def get_statistics(self, table_ids: List[int]) -> float:
        return 1

    def get_min_max_statistics(self)-> Tuple[float, float]:
        return 1, 1

    def stratify(self, max_blocking_size: int):
        n_max = max([len(table) for table in self.tables])
        top_k = 50 # int(math.pow(max_blocking_size / n_max, 1/(len(self.tables)-1))) + 1
        print("topK:", top_k)
        top_k_tables = []
        # 01
        similarity_scores = self.scores[0]
        top_k_mapping = np.argsort(similarity_scores, axis=0)[-top_k:, :]
        l_table_ids = []
        r_table_ids = []
        for r_id in range(top_k_mapping.shape[1]):
            for l_id in top_k_mapping[:, r_id]:
                l_table_ids.append(l_id)
                r_table_ids.append(r_id)
        top_k_tables.append(pd.DataFrame({
            "l_table_id": l_table_ids,
            "r_table_id": r_table_ids,
            "score_01": similarity_scores[l_table_ids, r_table_ids]
        }))

        # 12
        similarity_scores = self.scores[1]
        top_k_mapping = np.argsort(similarity_scores, axis=1)[:, -top_k:]
        l_table_ids = []
        r_table_ids = []
        for l_id in range(len(top_k_mapping)):
            for r_id in top_k_mapping[l_id]:
                l_table_ids.append(l_id)
                r_table_ids.append(r_id)
        top_k_tables.append(pd.DataFrame({
            "l_table_id": l_table_ids,
            "r_table_id": r_table_ids,
            "score_12": similarity_scores[l_table_ids, r_table_ids]
        }))

        # 23
        similarity_scores = self.scores[2]
        top_k_mapping = np.argsort(similarity_scores, axis=1)[:, -top_k:]
        l_table_ids = []
        r_table_ids = []
        for l_id in range(len(top_k_mapping)):
            for r_id in top_k_mapping[l_id]:
                l_table_ids.append(l_id)
                r_table_ids.append(r_id)
        top_k_tables.append(pd.DataFrame({
            "l_table_id": l_table_ids,
            "r_table_id": r_table_ids,
            "score_23": similarity_scores[l_table_ids, r_table_ids]
        }))

        # join the top k tables
        final_table = top_k_tables[0]
        final_table = final_table.rename(columns={"l_table_id": f"table_0",
                                                    "r_table_id": "table_1",
                                                    "score_01": "score"})
        for i in range(1, len(top_k_tables)):
            assert isinstance(final_table, pd.DataFrame)
            final_table = final_table.merge(top_k_tables[i],
                                            left_on=f"table_{i}",
                                            right_on="l_table_id",
                                            suffixes=("_old", ""))
            final_table = final_table.drop(columns=[
                "l_table_id"]).rename(columns={"r_table_id": f"table_{i+1}"})
            final_table["score"] = final_table["score"] * final_table[f"score_{i}{i+1}"]

        final_table = final_table.sort_values(by="score", ascending=False)
        final_table = final_table.head(max_blocking_size)

        self.total_blocking_scores = final_table["score"].sum()

        # divide the final table into config.K parts by order
        stratum_size = max_blocking_size // self.K
        for i in range(self.K):
            if i != self.K-1:
                stratum = final_table.iloc[i*stratum_size: (i+1)*stratum_size]
            else:
                stratum = final_table.iloc[i*stratum_size:]
            self.strata.append(stratum)
        final_table_ids = final_table[[f"table_{i}" for i in range(len(self.tables))]].values.tolist()
        self.blocking_ids = set([tuple(row) for row in final_table_ids])

        return self.blocking_ids, self.strata

    def get_gt(self, top_k: int = None):
        return self.all_fabrics[:top_k] if top_k is not None else self.all_fabrics

    def run_oracle(self, data: List[int]) -> bool:
        test = tuple([str(int(t)) for t in data])
        if tuple(test) in self.oracle_labels:
            fabrics = [table.iloc[data[i]]['Fabric'] for i, table in enumerate(self.tables)]
            return set(fabrics)
        else:
            return set()

    def sample(self, stratum_id: int, sample_size: int, replace: bool):
        sample_results = defaultdict(list)
        if stratum_id > 0:
            stratum = self.strata[stratum_id-1]
            assert isinstance(stratum, pd.DataFrame)
            weights = stratum["score"].values / stratum["score"].sum()
            sample_ids = np.random.choice(len(stratum), size=sample_size, replace=replace, p=weights)
            samples = stratum[["table_0", "table_1", "table_2", "table_3"]].iloc[sample_ids].values
            sample_weights = weights[sample_ids]
            for i, sample in enumerate(samples):
                fabrics = self.run_oracle(sample)
                for fabric in self.all_fabrics:
                    if fabric in fabrics:
                        sample_results[fabric].append(1 / len(stratum) / sample_weights[i])
                    else:
                        sample_results[fabric].append(0)
        elif stratum_id == 0:
            samples, sample_weights, sample_size = self.weighted_wander_join(sample_size)
            population_size = np.prod(self.get_sizes())
            for i, sample in enumerate(samples):
                fabrics = self.run_oracle(sample)
                for fabric in self.all_fabrics:
                    if fabric in fabrics:
                        sample_results[fabric].append(1 / population_size / sample_weights[i])
                    else:
                        sample_results[fabric].append(0)
            for _ in range(sample_size - len(sample_results)):
                for fabric in self.all_fabrics:
                    sample_results[fabric].append(0)
        return sample_results


    def weighted_wander_join(self, sample_size: int):
        start = time.time()
        reweight_factor = 1 / (1 - self.total_blocking_scores)
        weights = self.scores[0].flatten() / len(self.tables[0])
        init_samples = np.random.choice(len(self.tables[0]) * len(self.tables[1]), size=sample_size, replace=True, p=weights)
        print(f"done init sampling: sample size={len(init_samples)}/{len(self.tables[0]) * len(self.tables[1])}")
        # only take sample that is in self.oracle_012, calculate it using numpy
        samples = init_samples[np.isin(init_samples, list(self.oracle_01))]
        print("done filtering")
        sample_weights = weights[samples] * reweight_factor
        print(f"# useful samples: {len(samples)}")
        samples = np.unravel_index(samples, shape=(self.table_sizes[0], self.table_sizes[1]))
        samples = (np.array(samples).T).tolist()
        for i in tqdm(range(len(samples))):
            for j in range(1, len(self.scores)):
                weights = self.scores[j][samples[i][-1]]
                table_entry = np.random.choice(len(self.tables[j+1]), size=1, replace=True, p=weights).item()
                sample_weights[i] *= weights[table_entry]
                samples[i].append(table_entry)
        output_sample = []
        output_sample_weights = []
        for i, sample in enumerate(samples):
             if tuple(sample) not in self.blocking_ids:
                output_sample.append(sample)
                output_sample_weights.append(sample_weights[i])
        print(f"Time taken: {time.time()-start}")
        effective_sample_size = len(output_sample) + len(init_samples) - len(samples)

        return output_sample, output_sample_weights, effective_sample_size

    def get_stratum_size(self, stratum_id) -> int:
        if stratum_id == 0:
            return np.prod([len(table) for table in self.tables]).item() - len(self.blocking_ids)
        else:
            return len(self.strata[stratum_id-1])

    def wander_join(self, sample_size: int):
        start = time.time()
        init_samples = np.random.choice(len(self.tables[0]) * len(self.tables[1]), size=sample_size, replace=True)
        print("done init sampling")
        samples = init_samples[np.isin(init_samples, list(self.oracle_01))]
        print("done filtering")
        print(f"# useful samples: {len(samples)}")
        samples = np.unravel_index(samples, shape=(self.table_sizes[0], self.table_sizes[1]))
        samples = (np.array(samples).T).tolist()
        for i in tqdm(range(len(samples))):
            for j in range(1, len(self.scores)):
                table_entry = np.random.choice(len(self.tables[j+1]), size=1, replace=True).item()
                samples[i].append(table_entry)
        output_sample = []
        for i, sample in enumerate(samples):
             if tuple(sample) not in self.blocking_ids:
                output_sample.append(sample)
        results = defaultdict(list)
        for sample in output_sample:
            fabrics = self.run_oracle(sample)
            for fabric in self.all_fabrics:
                if fabric in fabrics:
                    results[fabric].append(1)
                else:
                    results[fabric].append(0)
        print(f"Time taken: {time.time()-start}")
        effective_sample_size = len(output_sample) + len(init_samples) - len(samples)
        for _ in range(effective_sample_size - len(output_sample)):
            for fabric in self.all_fabrics:
                results[fabric].append(0)
        return results

    def wander_join_blocking(self, sample_size: int):
        start = time.time()
        init_samples = np.random.choice(len(self.tables[0]) * len(self.tables[1]), size=sample_size, replace=True)
        print("done init sampling")
        samples = init_samples[np.isin(init_samples, list(self.oracle_01))]
        print("done filtering")
        print(f"# useful samples: {len(samples)}")
        samples = np.unravel_index(samples, shape=(self.table_sizes[0], self.table_sizes[1]))
        samples = (np.array(samples).T).tolist()
        with open("block_noci/valid_threshold.json") as f:
            valid_threshold = json.load(f)
        valid_threshold_01 = valid_threshold[f"{self.config.dataset_name}_01"]
        valid_threshold_12 = valid_threshold[f"{self.config.dataset_name}_12"]
        valid_threshold_23 = valid_threshold[f"{self.config.dataset_name}_23"]
        for i in tqdm(range(len(samples))):
            for j in range(1, len(self.scores)):
                table_entry = np.random.choice(len(self.tables[j+1]), size=1, replace=True).item()
                samples[i].append(table_entry)
        output_sample = []
        for i, sample in enumerate(samples):
             if tuple(sample) not in self.blocking_ids and self.original_scores_01[sample[0], sample[1]] >= valid_threshold_01 and self.original_scores_12[sample[1], sample[2]] >= valid_threshold_12 and self.original_scores_23[sample[2], sample[3]] >= valid_threshold_23:
                output_sample.append(sample)
        results = []
        for sample in output_sample:
            if self.run_oracle(sample):
                results.append(1)
            else:
                results.append(0)
        print(f"Time taken: {time.time()-start}")
        effective_sample_size = len(output_sample) + len(init_samples) - len(samples)
        results = results + [0 for _ in range(effective_sample_size - len(output_sample))]
        return results

def load_dataset(config: Config):
    if config.dataset_name in ["ecomm-q10", "ecomm-q10-black", "ecomm-q10-white", "ecomm-q10-red"]:
        return ScalableJoinDatasetN3(config)
    elif config.dataset_name == "ecomm-q11" and not config.join_reorder:
        return ScalableJoinDatasetN4(config)
    elif config.dataset_name == "ecomm-q11" and config.join_reorder:
        return ScalableJoinDatasetN3(config, table_ids=config.table_ids, K=2)
    elif config.dataset_name == "ecomm-q11-topk":
        return ScalableJoinDatasetN4TopK(config, K=2)
    else:
        return ScalableJoinDataset(config)
