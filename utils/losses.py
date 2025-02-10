import torch
import torch.nn as nn
from utils.tools import *

def ce_loss(scores, labels):
    return nn.CrossEntropyLoss()(scores, labels)

def kl_loss(ture, pred):
    ture = nn.Softmax(dim=1)(ture)
    pred = nn.LogSoftmax(dim=1)(pred)
    loss = nn.KLDivLoss(reduction = 'batchmean')(pred, ture)
    return loss

def kd_loss(outputs, labels, teacher_outputs, alpha, t):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/t, dim=1),
                             F.softmax(teacher_outputs/t, dim=1)) * (alpha * t * t) + \
                             F.cross_entropy(outputs, labels) * (1. - alpha)
 
    return KD_loss

def l2_loss(f_a, f_b):
    return nn.MSELoss()(f_a, f_b) 

def factorization_loss(f_a, f_b):
    # empirical cross-correlation matrix
    f_a_norm = (f_a - f_a.mean(0)) / (f_a.std(0)+1e-6)
    f_b_norm = (f_b - f_b.mean(0)) / (f_b.std(0)+1e-6)
    c = torch.mm(f_a_norm.T, f_b_norm) / f_a_norm.size(0)

    on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
    off_diag = off_diagonal(c).pow_(2).mean()
    loss = on_diag + 0.005 * off_diag

    return loss



def ova_loss(scores, label):
    """
    inputs --> scores: score from multi bi-classifier, shape = [N, 2, num_class] 
           --> label : known source label

    output <-- loss for positive and negative
    """
    assert len(scores.size()) == 3
    assert scores.size(1) == 2

    scores = F.softmax(scores, 1)

    label_p = torch.zeros((scores.size(0), scores.size(2))).long().cuda() 
    label_range = torch.arange(0, scores.size(0)).long()
    label_p[label_range, label] = 1
    label_n = 1 - label_p

    open_loss_pos = torch.mean(torch.sum(-torch.log(scores[:, 1, :] + 1e-8) * label_p, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(scores[:, 0, :] +  1e-8) * label_n, 1)[0])

 
    return open_loss_pos, open_loss_neg


# def ova_loss(scores, label):
#     assert len(scores.size()) == 3
#     assert scores.size(1) == 2
    
#     scores = F.softmax(scores, 1)
#     score_p = scores[:, 1, :]
#     pre_p = F.softmax(score_p, 1)

#     label_p = torch.zeros((scores.size(0), scores.size(2))).long().cuda() 
#     label_range = torch.arange(0, scores.size(0)).long()
#     label_p[label_range, label] = 1
#     label_n = 1 - label_p
#     open_loss_pos = torch.mean(torch.sum(-torch.log(scores[:, 1, :] + 1e-8) * label_p, 1))
    
#     tmp = pre_p.detach()
#     tmp[label_range, label] = 0
#     second = torch.max(tmp, dim=1)[1]
#     open_loss_neg = torch.mean(-torch.log(scores[label_range, 0, second] +  1e-8))

#     return open_loss_pos, open_loss_neg





