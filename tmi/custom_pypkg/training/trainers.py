# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 20:58:27 2022

@author: Jackie
"""
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import time

from ..utils.dataset import Xydf_Set, collate_Xydf, BatchCutter
from . import optimization
from . import loss_funcs

from ..preprocessing import features


#%%
class Trainer:
    def __init__(self, 
            args, 
            model,
            optimizer=None, 
            scalar=None,
            start_epoch=0, 
            device='cpu', 
            verbose=False, 
            n_iter_print=9999, 
        ):
        ### initialize the encoding model ###
        self.model = model.to(device)
        ### initialize the optimizer ###
        self.optimizer = (
            optimizer if optimizer is not None else 
            optim.Adam(self.model.parameters(), lr=args.lr)
        )
        
        ### Other Parameters ###
        # arguments
        self.args = args
        # data 
        self.scalar = scalar
        # training hyper-parameter
        self.start_epoch = start_epoch
        self.tol = args.tol
        self.lr = self.optimizer.param_groups[0]['lr'] if optimizer is not None else args.lr 
        # trainer setting
        self.logger = args.logger
        self.device = device
        self.verbose = verbose
        self.n_iter_print = n_iter_print
        self._elapsed_dict = dict()
        
        # optimizer
        self.schedule_kwargs = args.schedule_kwargs
        self.scheduler = optimization.get_scheduler(self.optimizer, **args.schedule_kwargs)
        # data loader
        self.batch_cutter = BatchCutter(**args.cutter_kwargs)
        # training hyper-parameter
        self.B = args.B
        self.max_epoch = args.max_epoch
        self.epoch = None
        
        
    def printf(self,*args,**kwargs):
        if self.verbose:
            if self.logger is not None:
                self.logger.log_string( *args,**kwargs )
            else:
                print(*args,**kwargs)
                
        
    ### ==== data preparation ====
    def calculate_xy(self, seq):
        _t0 = time.time()
        # calculate x & y
        X_df, y_df = features.compute_xy(seq, self.args, sort=False)
        self.printf('done', end='')
        if 'dt' in X_df.columns:
            self.printf('dt_mean={:.1f} s, '.format(X_df.dt.mean()), end='')
        if 'dl' in X_df.columns:
            self.printf('dl_mean={:.1f} m, '.format(X_df.dl.mean()), end='')
        self.printf('finish!')
        self._elapsed_dict['calculate_xy'] = self._elapsed_dict.get('calculate_xy', 0) + time.time() - _t0
        return X_df, y_df
    
    
    def get_last_model(self):
        return self.model
    
            
    ### ==== build dataloader ====
    def build_trainloader(self, X_df, y_df, batch_size):
        _t0 = time.time()
        self.printf('building training dataset (time window)...', end='')
        self.trainset = Xydf_Set(X_df, y_df)
        self.trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=collate_Xydf)
        self.printf('finish!')
        self._elapsed_dict['build_trainloader'] = self._elapsed_dict.get('build_trainloader', 0) + time.time() - _t0
        
        
    ### ==== start and finish training ====
    def training_initialize(self, seq_train):
        #
        self.tstart = time.time()
        self.best_loss = np.inf
        self.n_epoch_unchange = 0
        self.lr_history = []
        #
        self.X_df, self.y_df = self.calculate_xy(seq_train)
        self.X_df = self.scalar.fit_transform(self.X_df)
        

    def early_stop(self, loss):
        if loss < self.best_loss - self.tol:
            self.n_epoch_unchange = 0
        else:
            self.n_epoch_unchange += 1
        self.best_loss = min(loss, self.best_loss)
        return True if self.n_epoch_unchange > 100 else False
    
    
    def lossbackpropa(self, batch):
        self.model.train()
        # --prediction and back-propa--
        loss_dict = {}
        count_dict = {}
        for batch_cut in self.batch_cutter(batch):
            # X:[B,T,C+1], y:[B,Tu], mask:[B,T], masktw:[B,tw]
            _t20 = time.time()
            x_cut = batch_cut['X'].to(self.device)
            y_cut = batch_cut['y'].to(self.device)
            mask_cut = batch_cut['mask'].to(self.device)
            masktw_cut = batch_cut['masktw'].to(self.device)
            w_cut = batch_cut['weight'].to(self.device)
            if len(masktw_cut) == 1: continue
            _t21 = time.time()
            # --prediction & loss calculation--
            emission = self.model.get_emissions(x_cut, mask_cut) # [B,T,C] => [B,Tu,M]

            for loss_type in self.args.loss_params:
                loss_kwargs = self.args.loss_params[loss_type]
                if loss_type == 'crf':
                    loss_cut, count_cut = self.model.criterion_crf_loss(emission, y_cut, w_cut, masktw_cut, reduction='none')
                elif loss_type == 'ce':
                    loss_cut, count_cut = loss_funcs.criterion_smoothed_weighted_ce(emission, y_cut, masktw_cut, **loss_kwargs)
                elif loss_type == 'adj':
                    loss_cut, count_cut = loss_funcs.criterion_adjacent_contrast(emission, y_cut, masktw_cut, **loss_kwargs)

                if loss_type not in loss_dict:
                    loss_dict[loss_type] = loss_cut.sum()
                    count_dict[loss_type] = count_cut
                else:
                    loss_dict[loss_type] += loss_cut.sum()
                    count_dict[loss_type] += count_cut

        # --back propagation--
        for loss_type in self.args.loss_params:
            loss_dict[loss_type] = loss_dict[loss_type] / count_dict[loss_type]
            
        loss_total = sum(loss_dict.values())
        loss_total.backward()
        
        # --optimization--
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss_dict
    
    
    def run_epoch(self, seq_train):
        # initial 
        loss_total_dict = {}
        nbat_total = 0
        
        # start
        if self.epoch == 0:
            self.build_trainloader(self.X_df, self.y_df, self.B)
            
        # start
        for ibat,batch in enumerate(self.trainloader):
            # --prediction, loss calculation, and back propagation--
            loss_pred_bat = self.lossbackpropa(batch)
            self.lr_history.append(self.optimizer.param_groups[0]['lr'])
            # --record loss--
            for loss_type in self.args.loss_params:
                if loss_type not in loss_total_dict:
                    loss_total_dict[loss_type] = loss_pred_bat[loss_type]
                else:
                    loss_total_dict[loss_type] += loss_pred_bat[loss_type]
                nbat_total += 1
                
        # -- finish --
        loss_mean_dict = {loss_type: loss_total_dict[loss_type] / nbat_total for loss_type in self.args.loss_params}
        return loss_mean_dict


    def fit(self, seq_train):
        # initialize
        self.training_initialize(seq_train)
        
        # training
        for epoch in range(self.start_epoch, self.max_epoch):
            # --run one epoch--
            self.epoch = epoch
            loss_epoch_dict = self.run_epoch(seq_train)
            
            # print the result
            self.elapsed = time.time() - self.tstart
            if (epoch + 1) % self.n_iter_print == 0:
                lr = self.optimizer.param_groups[0]['lr']
                loss_epoch = sum(loss_epoch_dict.values())
                sentence = '[Epoch {}]: [Loss = {:.8f}], undesc = {}, lr = {:.6f}, Elapsed: {:.2f} s'.format(epoch, loss_epoch, self.n_epoch_unchange, lr, self.elapsed)
                self.printf(sentence)
                loss_epoch_detail = '\tLosses: ' + ', '.join([f'{loss_type}: {loss_epoch_dict[loss_type]:.5f}' for loss_type in self.args.loss_params])
                self.printf(loss_epoch_detail)
                
            # decide whether to stop early
            if self.early_stop(loss_epoch): 
                break
            
            # update learning rate
            self.scheduler.step()


