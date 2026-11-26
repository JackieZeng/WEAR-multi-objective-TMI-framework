# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 20:58:27 2022

@author: Jackie
"""
import torch
import torch.nn.functional as F

#%%

def compute_point_lengths(target, masktw):
    # length of the tripleg that contains each point
    # target: [B,T], masktw: [B,T]
    B, T = target.shape
    device = target.device

    target = target.clone()
    target[~masktw] = 0
    
    # change points of y, computed with a matrix
    changes = torch.cat([torch.zeros((B, 1), device=device, dtype=torch.bool), target[:, 1:] != target[:, :-1]], dim=1) # [B,T]
    tleg_ids = changes.cumsum(dim=1)  # [B,T]
    tleg_ids[~masktw] = -1  # [B,T]

    # ids of valid triplegs
    unique_ids = torch.unique(tleg_ids)
    unique_ids = unique_ids[unique_ids != -1]  # drop invalid ids

    # length of each tripleg
    mask_tleg = (tleg_ids.unsqueeze(-1) == unique_ids).float() # [B,T,U], U is len(unique_ids)
    tleg_lens = mask_tleg.sum(1, keepdim=True) # [B,1,U]

    # length of the tripleg that contains each point
    point_lens = tleg_lens.repeat(1, T, 1) * mask_tleg # [B,T,U]
    point_lens = point_lens.amax(2) # [B,T]

    return point_lens


def compute_point_weights(point_lens, alpha=0.):
    # length of the tripleg that contains each point
    # point_lens: [B,T]
    point_weights = 1 / point_lens**alpha  # 1/|Tl|^α
    point_weights = torch.where(point_lens > 0, point_weights, point_lens)  # [B,T]
    return point_weights


def criterion_smoothed_weighted_ce(logits, target, masktw, sm=0., alpha=0.):
    # logits: [B,T,M], target: [B,T], masktw: [B,T]
    _, _, M = logits.shape
 
    target = target.clone()
    target[~masktw] = 0
    
    # smoothed labels: 
    target_onehot = F.one_hot(target, num_classes=M).float() # [B,T] -> [B,T,M]
    target_smoothed = target_onehot * (1 - sm) + (1 - target_onehot) * (sm / (M - 1))  # [B,T,M]

    # KL divergence from the softmax probabilities
    log_probs = F.log_softmax(logits, dim=-1)  # [B,T,M]
    loss_point = - (target_smoothed * log_probs).sum(dim=-1)  # [B,T]
    
    # weight of each point
    point_lens = compute_point_lengths(target, masktw) # [B,T]
    point_weights = compute_point_weights(point_lens, alpha) # [B,T]

    loss = loss_point * point_weights # [B,T]

    count = point_weights.sum()
    return loss, count



def criterion_adjacent_contrast(logits, y, masktw, alpha=0.):
    # logits: [B,T,M], y: [B,Tu], masktw: [B,T]
    contrast_true = (y[:, :-1] != y[:, 1:]).float()  # [B,Tu-1]
    mask_valid = masktw[:, 1:] # [B,Tu-1]
    
    # weight of each point
    point_lens = compute_point_lengths(y, masktw) # [B,T]
    point_weights = compute_point_weights(point_lens, alpha) # [B,T]
    point_weights = torch.minimum(point_weights[:, :-1], point_weights[:, 1:]) # [B,Tu-1]

    # ---- dot product distance
    probs = logits.softmax(-1) # [B,T,M]
    dot_dist = 1 - (probs[:, :-1] * probs[:, 1:]).sum(-1) # [B,Tu-1]

    #
    dot_dist = dot_dist[mask_valid]
    contrast_true = contrast_true[mask_valid]
    bce_weights = point_weights[mask_valid]

    # BCE loss without reduction
    loss = F.binary_cross_entropy(dot_dist, contrast_true, reduction='none')
    loss = loss * bce_weights

    count = (mask_valid * point_weights).sum()
    return loss, count

