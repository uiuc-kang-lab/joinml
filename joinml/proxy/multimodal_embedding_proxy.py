from joinml.proxy.proxy import Proxy
from joinml.config import Config
import clip
import pandas as pd
from typing import List
from PIL import Image
import numpy as np
from joinml.utils import calculate_score_for_tables
import torch
from tqdm import tqdm

class MultiModalEmbeddingProxy(Proxy):
    def __init__(self, config: Config) -> None:
        self.model, self.process = clip.load(config.proxy, device=config.device)
        self.device = config.device

    def _embed_text(self, text: List[str]):
        text_encodings = clip.tokenize(text, truncate=True).to(self.device)
        text_embeddings = self.model.encode_text(text_encodings).detach().cpu().numpy()
        return text_embeddings
    
    def _embed_image(self, image_path: List[str]):
        images = [Image.open(path) for path in image_path]
        images = torch.stack([self.process(image).to(self.device) for image in images])
        images_embeddings = self.model.encode_image(images).detach().cpu().numpy()
        return images_embeddings
    
    def get_proxy_score_for_tables(self, table1: pd.DataFrame, table2: pd.DataFrame, is_self_join: bool=False) -> np.ndarray:
        text1 = table1["join_col"].to_list()
        text1_embeddings = self._embed_text(text1)
        if is_self_join:
            text2_embeddings = text1_embeddings
        else:
            text2 = table2["join_col"].to_list()
            text2_embeddings = self._embed_text(text2)
        
        images1 = table1["img_path"].to_list()
        images1_embeddings = self._embed_image(images1)
        if is_self_join:
            images2_embeddings = images1_embeddings
        else:
            images2 = table2["img_path"].to_list()
            images2_embeddings = self._embed_image(images2)
        
        # calculate the similarity between text1 and text2
        text_scores = calculate_score_for_tables(text1_embeddings, text2_embeddings)
        # calculate the similarity between images1 and images2
        image_scores = calculate_score_for_tables(images1_embeddings, images2_embeddings)
        # calculate the similarity between text1 and images2
        text_image_scores = calculate_score_for_tables(text1_embeddings, images2_embeddings)
        # calculate the similarity between images1 and text2
        image_text_scores = calculate_score_for_tables(images1_embeddings, text2_embeddings)

        # return the max similarity
        scores = np.maximum(text_scores, image_scores)
        scores = np.maximum(scores, text_image_scores)
        scores = np.maximum(scores, image_text_scores)
        # scores = np.average([text_scores, image_scores, text_image_scores, image_text_scores], axis=0)
        return scores

if __name__ == "__main__":
    config = Config()
    config.proxy = "ViT-B/32"
    config.data_path = "../../data/wikidiverse"
    proxy = MultiModalEmbeddingProxy(config)
    table1 = pd.read_csv("../../data/wikidiverse/data/table0.csv")
    table2 = pd.read_csv("../../data/wikidiverse/data/table1.csv")
    table1["img_path"] = table1["id"].apply(lambda x: f"../../data/wikidiverse/imgs/table0/{x}.jpg")
    table2["img_path"] = table2["id"].apply(lambda x: f"../../data/wikidiverse/imgs/table1/{x}.jpg")
    # get the first 10 rows of each table
    table1 = table1[:10]
    table2 = table2[:10]
    print(table1)
    print(table2)
    scores = proxy.get_proxy_score_for_tables(table1, table2)
    print(scores)
    scores = proxy.get_proxy_score_for_tables(table1, table1, is_self_join=True)
    print(scores)


        
