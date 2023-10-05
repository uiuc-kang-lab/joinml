from sentence_transformers import SentenceTransformer, util
import csv
import multiprocessing
import torch
import logging
import glob
from typing import List
import sys

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

def initializer(_embeddings2):
    global embeddings2
    embeddings2 = _embeddings2

def run_job(args):
    Id, embedding = args
    results = []
    for Id2, embedding2 in embeddings2:
        score = util.cos_sim(embedding, embedding2).item()
        results.append([Id, Id2, score])
        

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
            embeddings1 = [[Id, embeddings1_dct[Id]] for Id in embeddings1_dct]
            embeddings1_dct.clear()
            embeddings2 = [[Id, embeddings2_dct[Id]] for Id in embeddings2_dct]
            embeddings2_dct.clear()
            scores = []

            with multiprocessing.Pool(processes=num_worker, initializer=initializer, initargs=(embeddings2,)) as pool:
                for i, results in enumerate(pool.imap_unordered(run_job, embeddings1)):
                    scores += results
                    sys.stderr.write('\rdone {0:%}'.format(i/len(embeddings1)))
            
            with open(f"{output_folder}/bi_encoder.csv", "a+") as f:
                writer = csv.writer(f)
                writer.writerow(scores)
            
            scores = []

            if len(scores) >= limit and limit != -1:
                break
