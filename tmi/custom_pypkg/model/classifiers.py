# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:44:03 2022

@author: Jackie
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time

from .crf import CRF
from .encoders import ConvEncoder, AggrEncoder, RecuEncoder

from ..utils.dataset import Xdf_Set, collate_Xdf, BatchCutter


#%%
class TimeIt:
    def __init__(self, elapsed_dict):
        self.elapsed_dict = elapsed_dict
        
    def __repr__(self):
        return str(self.elapsed_dict)
        
    def __call__(self, kw, func):
        '''
        e.g.
            timer = TimeIt(elapsed_dict)
            y = timer(kw, func)(x, a, b)
        '''
        class TimedFunc:
            def __init__(self, elapsed_dict, func):
                self.elapsed_dict = elapsed_dict
                self.func = func
            
            def __call__(self, *args, **kwargs):
                ts = time.time()
                outputs = self.func(*args, **kwargs)
                self.elapsed_dict[kw] = self.elapsed_dict.get(kw, 0) + time.time() - ts
                return outputs
            
        return TimedFunc(self.elapsed_dict, func)
        


#%% ==== Full Model ====
class FullModel(nn.Module):
    def __init__(self, 
                 C, M, 
                 num_conv, dim_conv, kernel_size, dim_out_cnn, pdrop_cnn,
                 dim_hidden, pdrop_agg,
                 n_rnn, pdrop_rnn, 
                 ):
        super().__init__()
        self.C = C
        self.M = M
        self.kernel_size = kernel_size
        self.dim_out_cnn = dim_out_cnn
        self.pdrop_cnn = pdrop_cnn
        self.dim_hidden = dim_hidden
        self.n_rnn = n_rnn
        self.pdrop_rnn = pdrop_rnn 
        self.batch_cutter = BatchCutter(nsmp_max=4*256, max_empty_ratio=0.3, B_min=1, sort=False)
        # layers
        self.cnn = ConvEncoder(C, num_conv, dim_conv, kernel_size, dim_out_cnn, pdrop_cnn)
        self.agg = AggrEncoder(dim_out_cnn, dim_hidden, pdrop_agg)
        self.rnn = RecuEncoder(dim_hidden, n_rnn, M, pdrop_rnn)
        self.crf = CRF(M, batch_first=True)
        
        self._elapsed_dict = dict(cnn=0, agg=0, rnn=0, crf=0)
        self.timer = TimeIt(self._elapsed_dict)
    
    def get_device(self):
        return next(self.parameters()).device
    
    def timed_feature_extraction(self, module_name, *inputs):
        fe_timer = self.timer(module_name, getattr(self, module_name).feature_extraction)
        return fe_timer(*inputs)
        
    
    def feature_extraction(self, x, mask=None):
        # shape of x: [B,T,1+C]([itw,feat]), mask: [B,T]
        
        # CNN
        x = self.timed_feature_extraction('cnn', x) # -> [B,T,1+dim_out_cnn]
        
        # AGG
        assert mask is not None, 'Aggregation Module requires a mask as input'
        x = self.timed_feature_extraction('agg', x, mask) # -> [B,Tu,dim_hidden]
        
        # RNN
        x = self.timed_feature_extraction('rnn', x) # -> [B,Tu,dim_hidden]
        return x
    
        
    def get_emissions(self, x, mask=None):
        # shape of x: [B,T,1+C], mask: [B,T]
        x = self.feature_extraction(x, mask) # -> [B,Tu,dim_hidden]
        output = self.rnn.fc_out(x) # -> [B,Tu,M]
        return output
        
        
    def criterion_crf_loss(self, e, y, weight, masktw, reduction='token_mean'):
        # shape of e: [B,Tu,M], y: [B,Tu], mask: [B,Tu]
        e = e.detach()
        count = masktw.sum()
        return self.timer('crf', self.crf)(e, y, weight, masktw, reduction), count


    def forward(self, x, weight=None, mask=None, masktw=None, use_crf=True):
        # shape of x: [B,T,1+C], mask: [B,T]
        x = self.get_emissions(x, mask) # -> [B,Tu,M]
        if use_crf:
            return self.timer('crf', self.crf.decode)(x, weight, masktw) # list, [[y11, y12, ...], [y21, y22, ...], ...]
        else:
            return x # [B,Tu,M] 
        
        
    def build_dataloader(self, X_df, batch_size):
        dataset = Xdf_Set( X_df )
        dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, collate_fn=collate_Xdf)
        return dataloader
    
    
    @ torch.no_grad()
    def predict(self, X_df, batch_size=32, **kwargs):
        # kwargs
        use_crf = kwargs.get('use_crf', True)
        onehot = kwargs.get('onehot', False)
        softmax = kwargs.get('softmax', False)

        # predict
        predloader = self.build_dataloader(X_df, batch_size)
        res_pred = []
        for ibat, batch in enumerate(predloader):
            for batch_cut in self.batch_cutter(batch):
                # get data
                x_bat = batch_cut['X'].to(self.get_device()) # [B,T,C]
                mask_bat = batch_cut['mask'].to(self.get_device()) # [B,T]
                masktw_bat = batch_cut['masktw'].to(self.get_device()) # [B,Tu]
                # predict
                if use_crf:
                    w_bat = batch_cut['weight'].to(self.get_device()) # [B,Tu]
                    y_bat = self.forward(x_bat, weight=w_bat, mask=mask_bat, masktw=masktw_bat, use_crf=True) # list, [[y11, y12, ...], [y21, y22, ...], ...]
                    y_bat = torch.LongTensor([yij for yi in y_bat for yij in yi]) # -> [npnt,]
                    res_pred.append( y_bat ) # [npnt,]
                else:
                    oh_bat = self.forward(x_bat, mask=mask_bat, masktw=masktw_bat, use_crf=False) # -> [B,Tu,M]
                    res_pred.append( oh_bat[masktw_bat] ) # -> [ntw,M]
        res_pred = torch.cat( res_pred ).cpu()
        
        # return
        if use_crf: # y_perd
            return res_pred
        elif not onehot: # oh_pred -> y_perd
            return res_pred.argmax(-1)
        else: # oh_pred
            return F.softmax(res_pred, dim=-1) if softmax else res_pred
