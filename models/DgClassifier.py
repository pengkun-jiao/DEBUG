import os
import torch
from torch import nn
from torch.utils import model_zoo
from torchvision.models.resnet import BasicBlock, model_urls, Bottleneck
import torch.nn.functional as F
import numpy as np
import torchvision.models as models



class DgClassifier(nn.Module):
    def __init__(self, encoder, masker, classifier):
        super(DgClassifier, self).__init__()
        self.encoder = encoder
        self.masker = masker
        self.classifier = classifier
    

    def forward(self, x):
        features = self.encoder(x)
        if self.masker != None:
            features = self.masker(features)
        scores = self.classifier(features)

        return scores