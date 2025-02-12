import json
from collections import OrderedDict
from typing import Union, List

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseClip(nn.Module):
    _AVAILABLE_MODELS = ['RN50', 'RN50x16']  # refer to Table 3. in the paper

    def __init__(self,
                 name: str,
                 classnames: List[str] = None,
                 templates: List[str] = None,
                 device: Union[str, torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu',
                 jit: bool = False, download_root: str = None):
        super(DenseClip, self).__init__()
        self.clip_model, self.preprocess = clip.load(name, device, jit, download_root)

        self._init_visual(device)
        self._init_zeroshot_classifier(classnames, templates, device)

    def _init_visual(self, device):
        self.visual = self.clip_model.visual
        self.conv1 = nn.Conv2d(self.visual.attnpool.v_proj.in_features,
                               self.visual.attnpool.v_proj.out_features,
                               kernel_size=(1, 1)).to(device).to(self.dtype)
        self.conv2 = nn.Conv2d(self.visual.attnpool.c_proj.in_features,
                               self.visual.attnpool.c_proj.out_features,
                               kernel_size=(1, 1)).to(device).to(self.dtype)
        conv1_weight_shape = (*self.visual.attnpool.v_proj.weight.shape, 1, 1)
        conv2_weight_shape = (*self.visual.attnpool.c_proj.weight.shape, 1, 1)
        self.conv1.load_state_dict(
            OrderedDict(weight=self.visual.attnpool.v_proj.weight.reshape(conv1_weight_shape),
                        bias=self.visual.attnpool.v_proj.bias))
        self.conv2.load_state_dict(
            OrderedDict(weight=self.visual.attnpool.c_proj.weight.reshape(conv2_weight_shape),
                        bias=self.visual.attnpool.c_proj.bias))

    @torch.no_grad()
    def _init_zeroshot_classifier(self, classnames, templates, device):
        # refer to: https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]  # format with class
            texts = clip.tokenize(texts).to(device)  # tokenize
            class_embeddings = self.clip_model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()

            zeroshot_weights.append(class_embedding)

        # shape: [E, C]
        # where E is the dimension of an embedding and C is the number of classes.
        self.zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def _stem(self, x):
        for conv, bn in [(self.visual.conv1, self.visual.bn1),
                         (self.visual.conv2, self.visual.bn2),
                         (self.visual.conv3, self.visual.bn3)]:
            x = self.visual.relu(bn(conv(x)))

        x = self.visual.avgpool(x)
        return x

    def encode_image(self, image):
        image = image.type(self.dtype)
        feature = self._stem(image)
        feature = self.visual.layer1(feature)
        feature = self.visual.layer2(feature)
        feature = self.visual.layer3(feature)
        feature = self.visual.layer4(feature)

        # removed attnpool
        feature = self.conv1(feature)
        feature = self.conv2(feature)

        return feature

    def forward(self, images):
        # [B, E, h, w]
        features = self.encode_image(images)
        # [B, w, h, E]
        features_t = features.transpose(1, 3)
        # [B, w, h, C]
        output_t = features_t @ self.zeroshot_weights
        # [B, C, h, w]
        output = output_t.transpose(1, 3)
        output = F.interpolate(output, size=images.shape[-2:], mode='bilinear')
        return output

    @staticmethod
    def available_models():
        return DenseClip._AVAILABLE_MODELS


class Clip(nn.Module):
    _AVAILABLE_MODELS = ['RN50', 'RN50x16']  # refer to Table 3. in the paper

    def __init__(self,
                 name: str,
                 classnames: List[str] = None,
                 templates: List[str] = None,
                 device: Union[str, torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu',
                 jit: bool = False, download_root: str = None):
        super(Clip, self).__init__()
        self.clip_model, self.preprocess = clip.load(name, device, jit, download_root)

        if classnames is None:
            classnames = _DEFAULT_CLASSNAMES

        if templates is None:
            templates = _DEFAULT_TEMPLATES

        self._init_zeroshot_classifier(classnames, templates, device)

    @torch.no_grad()
    def _init_zeroshot_classifier(self, classnames, templates, device):
        # refer to: https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]  # format with class
            texts = clip.tokenize(texts).to(device)  # tokenize
            class_embeddings = self.clip_model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()

            zeroshot_weights.append(class_embedding)

        # shape: [E, C]
        # where E is the dimension of an embedding and C is the number of classes.
        self.zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)

    def encode_image(self, image):
        feature = self.clip_model.encode_image(image)
        feature /= feature.norm(dim=-1, keepdim=True)
        return feature

    def forward(self, images):
        features = self.encode_image(images)
        output = features @ self.zeroshot_weights
        return F.softmax(output, dim=-1)

    @staticmethod
    def available_models():
        return Clip._AVAILABLE_MODELS
