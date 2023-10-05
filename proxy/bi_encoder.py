from sentence_transformers import SentenceTransformer, util
import csv
import multiprocessing
import torch
import logging
import glob
from typing import Any, List
import sys
import logging

def get_embedding(table: List[int|str], output_folder: str, model_name: str="all-MiniLM-L6-v2", output_file_prefix: str=""):
    model = SentenceTransformer(model_name)
    embeddings = {}
    fid = 0
    for Id, content in table:
        try:
            embedding = model.encode(content, convert_to_tensor=True)
            embeddings[Id] = embedding
        except Exception as e:
            logging.info(f"table entry {Id} failed as {e}")
            return
        logging.info(f"generate embeddings for table entry {Id} successfully")

        if sys.getsizeof(embeddings) >= 1e9:
            torch.save(embeddings, f"{output_folder}/{output_file_prefix}{fid}.pt")
            embeddings.clear()
            fid += 1
    
    if len(embeddings) != 0:
        torch.save(embeddings, f"{output_folder}/{output_file_prefix}{fid}.pt")
        embeddings.clear()
        fid += 1

class Process(object):
    def __init__(self, embeddings) -> None:
        self.embeddings2 = embeddings

    def __call__(self, args) -> List:
        Id, embedding, output_folder = args
        results = []
        for Id2, embedding2 in self.embeddings2:
            score = util.cos_sim(embedding, embedding2).item()
            results.append([Id, Id2, score])
            # logging.info(f"done {Id}-{Id2}")
        with open(f"{output_folder}/bi_encoder.csv", "a+") as f:
                writer = csv.writer(f)
                writer.writerows(results)
        logging.info(f"finish left table entry with id {Id}")
        return results
        

def get_cosine_similarity(embedding_folder: str, output_folder: str, limit: int=-1, num_worker: int=4, is_selfjoin: bool=False):
    if is_selfjoin:
        embedding_filesl = glob.glob(embedding_folder)
        embedding_filesr = embedding_filesl
    else:
        embedding_filesl = glob.glob(f"{embedding_folder}/l*")
        embedding_filesr = glob.glob(f"{embedding_folder}/r*")

    for embedding_file1 in embedding_filesl:
        for embedding_file2 in embedding_filesr:
            embeddings1_dct = torch.load(embedding_file1)
            embeddings2_dct = torch.load(embedding_file2)
            embeddings1_args = [[Id, embeddings1_dct[Id], output_folder] for Id in embeddings1_dct]
            embeddings1_dct.clear()
            embeddings2 = [[Id, embeddings2_dct[Id]] for Id in embeddings2_dct]
            embeddings2_dct.clear()

            with multiprocessing.Pool(num_worker) as pool:
                pool.map(Process(embeddings2), embeddings1_args)
            
            num = len(embeddings1_args) * len(embeddings2)
            
            if len(num) >= limit and limit != -1:
                break
