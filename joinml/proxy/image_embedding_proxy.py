from joinml.proxy.proxy import Proxy
from joinml.config import Config
from joinml.proxy.resnet import resnet50
from joinml.utils import calculate_score_for_tuples, calculate_scre_for_tables
from joinml.proxy.reid_models import vit_base_patch16_224_TransReID

import os
import copy
import gdown
import torch
import numpy as np
import torch.nn as nn
from typing import List
import torch.nn.functional as F
from torchvision import transforms
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm


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


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x


class ReIDModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = vit_base_patch16_224_TransReID(img_size=[256, 256], sie_xishu=3.0, local_feature=True,
                                                    stride_size=[12, 12], drop_path_rate=0.1,
                                                    drop_rate= 0.0,
                                                    attn_drop_rate=0.0)
        self.in_planes = 768
        self.num_classes = 13164
        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
                    copy.deepcopy(block),
                    copy.deepcopy(layer_norm)
                )
        self.b2 = nn.Sequential(
                    copy.deepcopy(block),
                    copy.deepcopy(layer_norm)
                )
        
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_1.apply(weights_init_classifier)
        self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_2.apply(weights_init_classifier)
        self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_3.apply(weights_init_classifier)
        self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_4.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

    
    def forward(self, x):
        features = self.base(x)
        b1_feat = self.b1(features)
        global_feat = b1_feat[:, 0]

        feature_length = features.size(1) - 1
        patch_length = feature_length // 4
        token = features[:, 0:1]

        x = shuffle_unit(features, 8, 2)

        # lf_1
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2
        b2_local_feat = x[:, patch_length:patch_length*2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = x[:, patch_length*2:patch_length*3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
        b4_local_feat = x[:, patch_length*3:patch_length*4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]

        feat = self.bottleneck(global_feat)

        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        return torch.cat(
                    [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)


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

def _build_reid_model(model_cache_path: str):
    if not os.path.exists(f"{model_cache_path}/transreid.pth"):
        if not os.path.exists(model_cache_path):
            os.makedirs(model_cache_path)
        gdown.download(id="1CKH_Nyl7Aj4q6dh2v_kak0Khy44IsWQA",
                       output=f"{model_cache_path}/transreid.pth", quiet=False)
    model = ReIDModel()
    ckpt = torch.load(f"{model_cache_path}/transreid.pth", map_location="cpu")
    model.load_state_dict(ckpt)
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

def _run_reid(model: nn.modules, image_path: List[str],
              device: str="cpu", batch_size: int=8) -> np.ndarray:
    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    images = []
    for image_path in image_path:
        image = Image.open(image_path)
        image = transform(image)
        images.append(image)
    images = torch.stack(images)
    images = images.to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        features = []
        for i in tqdm(range(0, len(images), batch_size)):
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
            self.run  = _run_infomin
        elif model_name == "reid":
            self.model = _build_reid_model(config.model_cache_path)
            self.run = _run_reid
        else:
            raise NotImplementedError(f"Model {model_name} not implemented.")
        self.device = config.device
        self.batch_size = config.batch_size

    def get_proxy_score_for_tables(self, table1: List[str], table2: List[str], is_self_join: bool=False) -> np.ndarray:
        features1 = self.run(self.model, table1, device=self.device, batch_size=self.batch_size)
        if is_self_join:
            features2 = features1
        features2 = self.run(self.model, table2, device=self.device, batch_size=self.batch_size)
        return calculate_scre_for_tables(features1, features2)

    def get_proxy_score_for_tuples(self, tuples: List[List[str]]) -> np.ndarray:
        # flatten tuples
        num_tuples = len(tuples)
        tuples = [item for sublist in tuples for item in sublist]
        features = self.run(self.model, tuples, device=self.device, batch_size=self.batch_size)
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
    config.proxy = "reid"
    proxy = ImageEmbeddingProxy(config)
    print(proxy.get_proxy_score_for_tables(table1, table2))
    print(proxy.get_proxy_score_for_tuples(tuples))

    start = time.time()
    for _ in range(10):
        proxy.get_proxy_score_for_tables(table1, table2)
        proxy.get_proxy_score_for_tuples(tuples)
    end = time.time()
    print(f"Time: {end-start}")

