from joinml.config import Config
from joinml.dataset_loader import JoinDataset
from joinml.oracle import Oracle
from joinml.utils import divide_sample_rate
from joinml.sampling.sampler import RandomSampler

import networkx
from itertools import product
from torch.utils.data import Dataset
import logging
from tqdm import tqdm
import numpy as np

class ContrastiveTextDataset(Dataset):
    def __init__(self, data: dict, label):
        self.input_ids = data['input_ids']
        self.attention_mask = data['attention_mask']
        self.token_type_ids = data['token_type_ids']
        self.label = label
        assert len(self.attention_mask) == len(self.label), f"{len(self.attention_mask)}, {len(self.label)}"
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        return (self.input_ids[index], self.attention_mask[index]), self.label[index]

def get_train_data(config: Config, dataset: JoinDataset, oracle: Oracle, tokenizer=None) -> Dataset:
    """Get the training data for contrastive learning."""
    # get a sampler
    if config.train_data_sample_method == "random":
        sampler = RandomSampler()
    else:
        raise NotImplementedError(f"Sampler {config.train_data_sample_method} is not implemented yet.")
    # sample training data from the dataset
    table_sizes = dataset.get_sizes()
    # deal with self join
    if len(table_sizes) == 1:
        sample_size = int(np.sqrt(config.train_data_sample_rate) * table_sizes[0])
        logging.info(f"training data sample sizes: {sample_size}")
        train_data_ids = [sampler.sample(data=list(range(table_sizes[0])), size=sample_size, replace=False)] 
        train_data_per_table = dataset.get_join_column_per_table(ids=train_data_ids)
        train_data_ids *= 2
        train_data_per_table *= 2
    else:
        # TODO: implement for non-self-join situation
        raise NotImplementedError("Not implemented yet.")


    # flatten the training data
    train_data = []
    for train_data_table in train_data_per_table:
        train_data += train_data_table

    # tokenize if necessary
    if tokenizer is not None:
        train_data = tokenizer(train_data)
    
    # get labels for the training data by querying the oracle and get the connected components
    graph = networkx.Graph()
    for i in range(len(train_data)):
        graph.add_node(i)
    
    n_positive = 0
    for row in tqdm(product(*train_data_ids)):
        if oracle.query(row):
            n_positive += 1
            node_ids = [column_id + prev_table_size for column_id, prev_table_size in zip(row, [0] + table_sizes[:-1])]
            for i in range(len(node_ids)-1):
                graph.add_edge(node_ids[i], node_ids[i+1])
    
    logging.info(f"positive rate of the training data: {n_positive / np.prod([len(ids) for ids in train_data_ids])}")
    
    train_label = [None for _ in range(len(train_data["input_ids"]))]
    accmu_label_id = 0
    for label_id, component in enumerate(networkx.connected_components(graph)):
        for node in component:
            train_label[node] = label_id
            accmu_label_id = label_id
    accmu_label_id += 1
    for i in range(len(train_label)):
        if train_label[i] is None:
            train_label[i] = accmu_label_id
            accmu_label_id += 1

    logging.info(f"example of {len(train_data['input_ids'])} train data: {train_data['input_ids'][0]}")
    logging.info(f"example of {len(train_label)} train data label: {train_label[0]}")
    train_dataset = ContrastiveTextDataset(train_data, train_label)
    return train_dataset

    

