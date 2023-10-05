import torch
import glob
from tqdm import tqdm
from sentence_transformers import util
import os
import csv
import logging

logging.basicConfig(filename="qqp_embedding.log", level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

embedding_files = glob.glob(f"qqp_embeddings/*.csv")

embeddings = {}
logging.info("reading embeddings")
for embedding_file in tqdm(embedding_files):
    embedding = torch.load(embedding_file)
    qid = embedding_file.split(".")[0].split("/")[-1]
    if int(qid) > 3500:
        continue
    embeddings[qid] = embedding

logging.info("saving embeddings")
torch.save(embeddings, "qqp_embedding.pt")

# calculating proxy values
if not os.path.exists("bi_encoder_qqp_proxy.csv"):
    with open("bi_encoder_qqp_proxy.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["qid1", "qid2", "proxy_score"])

logging.info("calculating proxy scores")
for qid1 in tqdm(embeddings):
    for qid2 in embeddings:
        if int(qid1) < int(qid2):
            embedding1 = embeddings[qid1]
            embedding2 = embeddings[qid2]
            cosine_scores = util.cos_sim(embedding1, embedding2)
            with open("bi_encoder_qqp_proxy.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow([qid1, qid2, cosine_scores])

