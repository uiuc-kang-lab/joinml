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
from joinml.proxy.blip_retrieval import blip_retrieval
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F


def _clip_embed_text(model, text: List[str], device: str="cpu", batch_size: int=32):
    text_embeddings = []
    for i in tqdm(range(0, len(text), batch_size)):
        if i + batch_size > len(text):
            text_encodings = clip.tokenize(text[i:], truncate=True).to(device)
        else:
            text_encodings = clip.tokenize(text[i:i+config.batch_size], truncate=True).to(device)
        text_embeddings.append(model.encode_text(text_encodings).detach().cpu().numpy())
    return np.concatenate(text_embeddings, axis=0)

def _clip_embed_image(model, process, image_path: List[str], device: str="cpu", batch_size: int=32):
    images_embeddings = []
    for i in tqdm(range(0, len(image_path), batch_size)):
        if i + batch_size > len(image_path):
            images = []
            for path in image_path[i:]:
                images.append(Image.open(path))
        else:
            images = []
            for path in image_path[i:i+batch_size]:
                images.append(Image.open(path))
        images = torch.stack([process(image).to(device) for image in images])
        images_embeddings.append(model.encode_image(images).detach().cpu().numpy())
    return np.concatenate(images_embeddings, axis=0)


def _blip_preprocess(images: List) -> List:
    images = [image.convert("RGB") for image in images]
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    blip_transforms = transforms.Compose([
        transforms.Resize((384, 384),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    images = [blip_transforms(image) for image in images]
    return images

def _blip_embed_image(model, process, image_path: List[str], device: str="cpu", batch_size: int=32):
    images_embeddings = []
    for i in tqdm(range(0, len(image_path), batch_size)):
        if i + batch_size > len(image_path):
            images = []
            for path in image_path[i:]:
                images.append(Image.open(path))
        else:
            images = []
            for path in image_path[i:i+batch_size]:
                images.append(Image.open(path))
        images = process(images)
        images = torch.stack(images).to(device)
        image_feats = model.visual_encoder(images)
        image_embed = model.vision_proj(image_feats[:, 0, :])
        images_embeddings.append(image_embed.detach().cpu().numpy())
    return np.concatenate(images_embeddings, axis=0)


def _blip_embed_text(model, text: List[str], device: str="cpu", batch_size: int=32):
    text_embeddings = []
    for i in tqdm(range(0, len(text), batch_size)):
        if i + batch_size > len(text):
            text_encodings = model.tokenizer(text[i:], padding="max_length", truncation=True, max_length=35, return_tensors="pt").to(device)
        else:
            text_encodings = model.tokenizer(text[i:i+batch_size], padding="max_length", truncation=True, max_length=35, return_tensors="pt").to(device)
        text_encodings = model.text_encoder(text_encodings.input_ids, attention_mask=text_encodings.attention_mask, mode="text")
        text_embed = F.normalize(model.text_proj(text_encodings.last_hidden_state[:, 0, :]), dim=-1)
        text_embeddings.append(text_embed.detach().cpu().numpy())
    return np.concatenate(text_embeddings, axis=0)

class MultiModalEmbeddingProxy(Proxy):
    def __init__(self, config: Config) -> None:
        if config.proxy == "clip":
            self.model, self.process = clip.load("ViT-B/32", device=config.device)
            self.image_embed = _clip_embed_image
            self.text_embed = _clip_embed_text
        elif config.proxy == "blip":
            pretrained_link = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_flickr.pth'
            self.model = blip_retrieval(pretrained=pretrained_link, image_size=384, vit='base', 
                                        vit_grad_ckpt=True, vit_ckpt_layer=4, queue_size=57600,
                                        negative_all_rank=False)
            self.model = self.model.to(config.device)
            self.process = _blip_preprocess
            self.image_embed = _blip_embed_image
            self.text_embed = _blip_embed_text
        self.device = config.device
        self.config = config
    
    def get_proxy_score_for_tables(self, table1: List[str], table2: List[str], is_self_join: bool=False) -> np.ndarray:
        image_cache_path = f"{self.config.cache_path}/{self.config.dataset_name}_{self.config.proxy.split('/')[-1]}_image.npy"
        text_cache_path = f"{self.config.cache_path}/{self.config.dataset_name}_{self.config.proxy.split('/')[-1]}_text.npy"
        if not os.path.exists(image_cache_path):
            images_embeddings = self.image_embed(self.model, self.process, table1, self.device, self.config.batch_size)
            np.save(image_cache_path, images_embeddings)
        else:
            images_embeddings = np.load(image_cache_path)
        
        if not os.path.exists(text_cache_path):
            text_embeddings = self.text_embed(self.model, table2.to_list(), self.device, self.config.batch_size)
            np.save(text_cache_path, text_embeddings)
        else:
            text_embeddings = np.load(text_cache_path)

        text_embeddings = np.array(text_embeddings, dtype=np.float32)
        images_embeddings = np.array(images_embeddings, dtype=np.float32)

        import logging
        logging.info(f"image embedding has shape {images_embeddings.shape}")
        logging.info(f"text embedding has shape {text_embeddings.shape}")

        return calculate_score_for_tables(images_embeddings, text_embeddings)


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


        
