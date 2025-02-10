import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# class Classifier(nn.Module):
#     def __init__(self, in_dim, num_classes):
#         super(Classifier, self).__init__()
#         self.in_dim = in_dim
#         self.num_classes = num_classes

#         self.layers = nn.Linear(in_dim, num_classes)

#     def forward(self, features):
#         scores = self.layers(features)
#         return scores

class Classifier(nn.Module):
    def __init__(self, num_classes=12, in_dim=2048):
        super(Classifier, self).__init__()
        
        # self.fc1 = nn.Linear(in_dim, 1024, bias=False)
        # self.fc2 = nn.Linear(1024, num_classes, bias=False)

        self.fc = nn.Linear(in_dim, num_classes, bias=False)

    def forward(self, x, dropout=False, return_feat=False):
        if return_feat:
            return x
        x = self.fc(x)
        # x = self.fc1(x)
        # x = self.fc2(x)
        return x

    def weight_norm(self):
        # for fc in [self.fc1, self.fc2]:
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))
            
    def weights_init(self):
        self.fc.weight.data.normal_(0.0, 0.1)


class Masker(nn.Module):
    def __init__(self, in_dim=2048, num_classes=2048, middle =8192, k = 1024):
        super(Masker, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.k = k

        self.layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_dim, middle),
            nn.BatchNorm1d(middle, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(middle, middle),
            nn.BatchNorm1d(middle, affine=True),
            nn.ReLU(inplace=True),
            nn.Linear(middle, num_classes))

        self.bn = nn.BatchNorm1d(num_classes, affine=False)

    def forward(self, f):
       mask = self.bn(self.layers(f))
       z = torch.zeros_like(mask)
       for _ in range(self.k):
           mask = F.gumbel_softmax(mask, dim=1, tau=0.5, hard=False)
           z = torch.maximum(mask,z)
       return z