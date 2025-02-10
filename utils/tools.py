import sys 
sys.setrecursionlimit(3000) 

import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
import torch.nn.functional as F
import random

def entropy(p):
    return -1 * torch.sum(p * torch.log2(p), dim=1)

def inference(score):
    return torch.argmax(score, dim=-1)

def openset_inference(score, score_ova=None, threshold=0.8):
    p_m = F.softmax(score, dim=1)
    pred = torch.argmax(p_m, dim=-1)
    miu = entropy(p_m)
    for i in range(len(miu)):
        if miu[i] > threshold:
            pred[i] = -1
    return pred

# def openset_inference(score, score_ova, threshold=0.8):
#     pred = torch.argmax(score, dim=-1)
#     for i in range(len(pred)):
#         if score_ova[i][1][pred[i]] < score_ova[i][0][pred[i]]:
#             pred[i] = -1
#     return pred


def batch_mask(batch, clip_mask, label, background=None):
    mask_batch = torch.empty_like(batch)
    bgmix_batch = torch.empty_like(batch)

    bs = batch.shape[0]
    for i in range(bs):
        img = batch[i]
        mask = clip_mask[i]

        if background == None:
            # bg = batch[random.randint(0, bs-1)]
            size = len(label)
            bg_idx = i
            step = 0
            while(label[bg_idx] == label[i]):
                bg_idx += 1
                if bg_idx == size: bg_idx = 0
                step += 1
                if step == size: break
            
            bg = batch[bg_idx]
            
        else:
            bg = background
    
        mask_batch[i][0] = mask * img[0]
        mask_batch[i][1] = mask * img[1]
        mask_batch[i][2] = mask * img[2]

        _mask = 1 - mask
        bgmix_batch[i][0] = mask * img[0] + _mask * bg[0]
        bgmix_batch[i][1] = mask * img[1] + _mask * bg[1]
        bgmix_batch[i][2] = mask * img[2] + _mask * bg[2]

    return mask_batch, bgmix_batch

def mask_image(img, mask, bg):
    mask_img = torch.clone(img)
    mix_img = torch.clone(img)

    for j in range(mask.shape[0]):
        for k in range(mask.shape[1]):
            if not mask[j][k]:
                mix_img[0][j][k] = bg[0][j][k]
                mix_img[1][j][k] = bg[1][j][k]
                mix_img[2][j][k] = bg[2][j][k]

                mask_img[0][j][k] = 0
                mask_img[1][j][k] = 0
                mask_img[2][j][k] = 0


    return mask_img, mix_img

def denorm(tensor, device, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    std = torch.Tensor(std).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor(mean).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res


def save_image_from_tensor_batch(batch, column, path, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], device='cpu'):
    batch = denorm(batch, device, mean, std)
    save_image(batch, path, nrow=column)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def step_rampup(current, rampup_length):
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return 0.0


def get_current_consistency_weight(epoch, weight, rampup_length, rampup_type='step'):
    if rampup_type == 'step':
        rampup_func = step_rampup
    elif rampup_type == 'linear':
        rampup_func = linear_rampup
    elif rampup_type == 'sigmoid':
        rampup_func = sigmoid_rampup
    else:
        raise ValueError("Rampup schedule not implemented")

    return weight * rampup_func(epoch, rampup_length)

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

