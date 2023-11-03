from joinml.proxy.proxy import Proxy
from joinml.config import Config
from joinml.proxy.resnet import resnet50
from joinml.utils import calculate_score_for_tuples, calculate_scre_for_tables

import os
import gdown
import torch
import numpy as np
import torch.nn as nn
from typing import List
import torch.nn.functional as F
from torchvision import transforms
from collections import OrderedDict
from PIL import Image

class Normalize(nn.Module):
    def __init__(self, p=2):
        super(Normalize, self).__init__()
        self.p = p

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=1)

class RGBSingleHead(nn.Module):
    """RGB model with a single linear/mlp projection head"""
    def __init__(self, name='resnet50', head='linear', feat_dim=128):
        super(RGBSingleHead, self).__init__()

        name, width = self._parse_width(name)
        dim_in = int(2048 * width)
        self.width = width

        self.encoder = resnet50(width=width)

        if head == 'linear':
            self.head = nn.Sequential(
                nn.Linear(dim_in, feat_dim),
                Normalize(2)
            )
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim),
                Normalize(2)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    @staticmethod
    def _parse_width(name):
        if name.endswith('x4'):
            return name[:-2], 4
        elif name.endswith('x2'):
            return name[:-2], 2
        else:
            return name, 1

    def forward(self, x, mode=0):
        # mode --
        # 0: normal encoder,
        # 1: momentum encoder,
        # 2: testing mode
        feat = self.encoder(x)
        if mode == 0 or mode == 1:
            feat = self.head(feat)
        return feat

def _build_info_min_model(model_cache_path: str):
    if not os.path.exists(f"{model_cache_path}/infomin.pth"):
        if not os.path.exists(model_cache_path):
            os.makedirs(model_cache_path)
        gdown.download(id="1cCuHA95Us7CejU5V63aZp5PaKVSSqXkK",
                       output=f"{model_cache_path}/infomin.pth", quiet=False)
    model = RGBSingleHead(name="resnet50", head="linear", feat_dim=128)
    ckpt = torch.load(f"{model_cache_path}/infomin.pth", map_location="cpu")
    state_dict = ckpt['model']
    encoder_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = k.replace('module.', '')
        if 'encoder' in k:
            k = k.replace('encoder.', '')
            encoder_state_dict[k] = v
    model.encoder.load_state_dict(encoder_state_dict)    
    return model

def _run_infomin(model: nn.modules, images_path: List[str], 
                 device: str="cpu", batch_size: int=8) -> np.ndarray:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    images = []
    for image_path in images_path:
        image = Image.open(image_path)
        image = transform(image)
        images.append(image)
    images = torch.stack(images)
    images = images.to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        features = []
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_features = model(batch_images)
            features.append(batch_features)
        features = torch.cat(features, dim=0)
        features = features.cpu().numpy()
    return features

class ImageEmbeddingProxy(Proxy):
    def __init__(self, config: Config):
        model_name = config.proxy
        if model_name == "infomin":
            self.model = _build_info_min_model(config.model_cache_path)
        else:
            raise NotImplementedError(f"Model {model_name} not implemented.")
        self.device = config.device
        self.batch_size = config.batch_size

    def get_proxy_score_for_tables(self, table1: List[str], table2: List[str]) -> np.ndarray:
        features1 = _run_infomin(self.model, table1, device=self.device, batch_size=self.batch_size)
        features2 = _run_infomin(self.model, table2, device=self.device, batch_size=self.batch_size)
        return calculate_scre_for_tables(features1, features2)

    def get_proxy_score_for_tuples(self, tuples: List[List[str]]) -> np.ndarray:
        # flatten tuples
        num_tuples = len(tuples)
        tuples = [item for sublist in tuples for item in sublist]
        features = _run_infomin(self.model, tuples, device=self.device, batch_size=self.batch_size)
        # convert back to tuples
        features = features.reshape(num_tuples, 2, -1)
        # run calculation
        return calculate_score_for_tuples(features)

if __name__ == "__main__":
    from itertools import product
    import time
    images = [f"../../data/city_vehicle/imgs/table0/{i}.jpg" for i in range(20)]
    table1 = images[:10]
    table2 = images[10:]
    tuples = list(product(images, images))
    config = Config()
    config.proxy = "infomin"
    proxy = ImageEmbeddingProxy(config)
    print(proxy.get_proxy_score_for_tables(table1, table2))
    print(proxy.get_proxy_score_for_tuples(tuples))

    start = time.time()
    for _ in range(10):
        proxy.get_proxy_score_for_tables(table1, table2)
        proxy.get_proxy_score_for_tuples(tuples)
    end = time.time()
    print(f"Time: {end-start}")

