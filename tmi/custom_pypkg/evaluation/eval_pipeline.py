# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 11:11:27 2022

@author: Jackie
"""
import numpy as np
import pandas as pd

from ..preprocessing import features
from . import metrics
from . import visual

class EvaluationPipeline:
    # model test for each fold
    def __init__(self, args, scalar, model, validate_with):
        self.args = args
        self.scalar = scalar
        self.model = model
        self.validate_with = validate_with

    
    def sequence_identification(self, seq, **kwargs):
        # kwargs
        batch_size = kwargs.get('batch_size', 32)
        use_crf = kwargs.get('use_crf', True)

        # -- sequentialize --
        self.args.logger.log_string('calculating x and y...', end='')
        X_test_df, y_test_df = features.compute_xy(seq, self.args, sort=False)
        
    
        if 'dt' in X_test_df.columns:
            self.args.logger.log_string('dt_mean={:.1f} s, '.format(X_test_df.dt.mean()), end='')
        
        # -- feature scalar -- 
        X_test_sc = self.scalar.transform(X_test_df)
        self.args.logger.log_string('done.')
        
        self.model.eval()
        X_test_sc = X_test_sc.sort_index()
        y_test_df = y_test_df.set_index('itw', append=True)
        y_test_df = y_test_df.groupby(['uid','tid','itw'], sort=True).mode.max()
        
        # -- predict y --
        y_pred = self.model.predict(X_test_sc, batch_size=batch_size, use_crf=use_crf).cpu() # if not has_crf, it won't use crf
        y_pred_df = pd.DataFrame(y_pred, columns=['mode_pred'], index=y_test_df.index)
        
        # -- merge to the sequence --
        mode_of_points = seq[['uid','tid','ipnt','mode','dl','dt','timestamp','lng','lat']]
        is_tlid_change = (mode_of_points.tid!=mode_of_points.tid.shift(1)) | (mode_of_points['mode']!=mode_of_points['mode'].shift(1))
        mode_of_points['tlid'] = is_tlid_change.cumsum()-1
        
        X_test_df = self.scalar.inverse_transform(X_test_sc)
        seq_infos = X_test_df.reset_index()[['uid','tid','itw','ipnt']]
        seq_infos = seq_infos.merge(y_pred_df, on=['uid','tid','itw'])
        seq_infos = seq_infos.merge(mode_of_points , on=['uid','tid','ipnt'])
        seq_infos = seq_infos.set_index(['uid','tid','itw','ipnt'])
            
        return seq_infos



    def process(self, seq, metric=True, **kwargs):
        self.args.logger.log_string('*'*12 + ' test '+ '*'*12)
        # -- TMI one point-level --
        self.seq_infos = self.sequence_identification(seq, **kwargs)
        
        # -- evaluate from sequence --
        if metric:
            correctness, transfer, tripleg, mode_chain = metrics.evaluate_from_sequence(self.seq_infos, self.args.t_thr, self.args.d_thr)
            visual.print_evaluation(correctness, transfer, tripleg, mode_chain, validate_with=self.validate_with, logger=self.args.logger, show_all=False)
        return self.seq_infos
    
    
    

