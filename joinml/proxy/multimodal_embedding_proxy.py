from joinml.proxy.proxy import Proxy
from joinml.config import Config
import clip
import pandas as pd
from typing import List
from PIL import Image, ImageFile
import numpy as np
from joinml.utils import calculate_score_for_tables
import torch
from tqdm import tqdm
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

class MultiModalEmbeddingProxy(Proxy):
    def __init__(self, config: Config) -> None:
        self.model, self.process = clip.load(config.proxy, device=config.device)
        self.device = config.device
        self.config = config

    def _embed_text(self, text: List[str]):
        text_embeddings = []
        for i in tqdm(range(0, len(text), self.config.batch_size)):
            if i + self.config.batch_size > len(text):
                text_encodings = clip.tokenize(text[i:], truncate=True).to(self.device)
            else:
                text_encodings = clip.tokenize(text[i:i+self.config.batch_size], truncate=True).to(self.device)
            text_embeddings.append(self.model.encode_text(text_encodings).detach().cpu().numpy())
        return np.concatenate(text_embeddings, axis=0)
    
    def _embed_image(self, image_path: List[str]):
        images_embeddings = []
        for i in tqdm(range(0, len(image_path), self.config.batch_size)):
            if i + self.config.batch_size > len(image_path):
                images = []
                for path in image_path[i:]:
                    try:
                        images.append(Image.open(path))
                    except:
                        images.append(Image.new("RGB", (224, 224), (255, 255, 255)))
            else:
                images = []
                for path in image_path[i:i+self.config.batch_size]:
                    try:
                        images.append(Image.open(path))
                    except:
                        images.append(Image.new("RGB", (224, 224), (255, 255, 255)))
            images = torch.stack([self.process(image).to(self.device) for image in images])
            images_embeddings.append(self.model.encode_image(images).detach().cpu().numpy())
        return np.concatenate(images_embeddings, axis=0)
    
    def get_proxy_score_for_tables(self, table1: pd.DataFrame, table2: pd.DataFrame, is_self_join: bool=False) -> np.ndarray:
        # check if the embeddings exists
        text1_cache_path = f"{self.config.cache_path}/{self.config.dataset_name}_embeddings/text1.npy"
        text2_cache_path = f"{self.config.cache_path}/{self.config.dataset_name}_embeddings/text2.npy"
        images1_cache_path = f"{self.config.cache_path}/{self.config.dataset_name}_embeddings/images1.npy"
        images2_cache_path = f"{self.config.cache_path}/{self.config.dataset_name}_embeddings/images2.npy"
        if os.path.exists(text1_cache_path):
            text1_embeddings = np.load(text1_cache_path)
        else:
            text1 = table1["join_col"].to_list()
            text1_embeddings = self._embed_text(text1)
            np.save(text1_cache_path, text1_embeddings)
        
        if os.path.exists(text2_cache_path):
            text2_embeddings = np.load(text2_cache_path)
        else:
            if is_self_join:
                text2_embeddings = text1_embeddings
            else:
                text2 = table2["join_col"].to_list()
                text2_embeddings = self._embed_text(text2)
            np.save(text2_cache_path, text2_embeddings)
        
        if os.path.exists(images1_cache_path):
            images1_embeddings = np.load(images1_cache_path)
        else:
            images1 = table1["img_path"].to_list()
            images1_embeddings = self._embed_image(images1)
            np.save(images1_cache_path, images1_embeddings)
            
        if os.path.exists(images2_cache_path):
            images2_embeddings = np.load(images2_cache_path)
        else:
            if is_self_join:
                images2_embeddings = images1_embeddings
            else:
                images2 = table2["img_path"].to_list()
                images2_embeddings = self._embed_image(images2)
            np.save(images2_cache_path, images2_embeddings)
        
        text1_embeddings = np.array(text1_embeddings, dtype=np.float32)
        text2_embeddings = np.array(text2_embeddings, dtype=np.float32)
        images1_embeddings = np.array(images1_embeddings, dtype=np.float32)
        images2_embeddings = np.array(images2_embeddings, dtype=np.float32)

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


        
