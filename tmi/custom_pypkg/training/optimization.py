# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 20:58:27 2022

@author: Jackie
"""
import torch.optim as optim


def get_optimizer(model, args):
    params = [{'params': model.parameters(), "lr": args.lr}]
    optimizer = optim.Adam(params)
    return optimizer

def get_scheduler(optimizer, **kwargs):
    step_peak = kwargs.get('step_peak')
    step_half = kwargs.get('step_half')
    peak = kwargs.get('peak', 1)
    lr_lambda = lambda step: min(peak * (step + 1) / step_peak, peak * 2**(- ((step + 1) - step_peak) / step_half))
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda = lr_lambda
    )
    return scheduler
        
