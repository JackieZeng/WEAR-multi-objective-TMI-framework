# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 11:11:27 2022

@author: Jackie
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_recall_fscore_support as prf
)
from ..preprocessing.features import sphericalDistance

# ==== Evaluation ====
class Evaluator:
    def __init__(self, col_true='mode', col_pred='mode_pred', t_thr=120, d_thr=100):
        self.col_true = col_true
        self.col_pred = col_pred
        self.t_thr = t_thr
        self.d_thr = d_thr

    # -- accuracy, f1score, and confusion matrix --
    def eval_correctness(self, seq_infos):
        '''
        Compute accuracy, F1-score, and the confusion matrix.

        params:
            seq_infos: trajectory table with columns [self.col_true, self.col_pred]
        '''
        label_true = seq_infos[self.col_true].values
        label_pred = seq_infos[self.col_pred].values
        
        # # point-based
        acc = accuracy_score(label_true, label_pred)
        f1score = prf(label_true, label_pred, average='macro')[2]
        conf = confusion_matrix(label_true, label_pred)
        
        # --summarize--
        correctness = dict(
            acc=acc, f1score=f1score, conf=conf
        )
        return correctness


    # -- trasfer --
    def eval_transfer(self, seq_infos):
        '''
        Count recall and precision of detected transfer points. A true transfer is counted as correct when a detected transfer lies within the time or distance threshold.

        params:
            seq_infos: trajectory table with columns ['uid', 'tid', 'timestamp', 'lng', 'lat'] + [col_true, col_pred]
            t_thr: time threshold for a transfer match, in seconds
            d_thr: distance threshold for a transfer match, in meters
        '''
        count_tp_list = [] # true-positive count
        count_fp_list = [] # false-positive count
        count_fn_list = [] # false-negative count
        
        for tid, traj in seq_infos.groupby('tid'):
            label_true = traj[self.col_true].values
            label_pred = traj[self.col_pred].values
            timestamp = traj['timestamp'].values
            lng = traj['lng'].values
            lat = traj['lat'].values

            # true and predicted transfer points
            true_trans = []
            pred_trans = []
            for i in range(len(traj) - 1):
                if label_true[i] != label_true[i+1]:
                    true_trans.append(i)
                if label_pred[i] != label_pred[i+1]:
                    pred_trans.append(i)
                    
            # counts
            tp = 0
            for idx_true in true_trans.copy():
                for idx_pred in pred_trans.copy():
                    dt = abs(timestamp[idx_true] - timestamp[idx_pred])
                    dl = sphericalDistance(lng[idx_true], lat[idx_true], lng[idx_pred], lat[idx_pred])
                    if dt <= self.t_thr or dl <= self.d_thr:
                        tp += 1
                        true_trans.remove(idx_true)
                        pred_trans.remove(idx_pred)
                        break
            
            fn = len(true_trans)
            fp = len(pred_trans)

            count_tp_list.append(tp)
            count_fp_list.append(fp)
            count_fn_list.append(fn)

        transfer_metrics = dict(
            count_tp=np.sum(count_tp_list),
            count_fp=np.sum(count_fp_list),
            count_fn=np.sum(count_fn_list)
        )
        return transfer_metrics



    # -- tripleg --
    def _get_triplegs(self, label_seq):
        """List of single-mode segments (triplegs)."""
        triplegs = []
        start = 0
        for i in range(len(label_seq) - 1):
            if label_seq[i] != label_seq[i+1]:
                triplegs.append({'start': start, 'end': i, 'label': label_seq[i], 'length': i - start})
                start = i
        triplegs.append({'start': start, 'end': len(label_seq)-1, 'label': label_seq[ len(label_seq)-1], 'length': len(label_seq) - start - 1})
        return triplegs

    def _is_tripleg_match(self, tripleg1, tripleg2, timestamp, lng, lat):
        """Whether two single-mode segments match."""
        # same mode required
        if tripleg1['label'] != tripleg2['label']:
            return False
            
        # endpoints must fall inside the time or distance threshold
        t1_start = timestamp[tripleg1['start']]
        t1_end = timestamp[tripleg1['end']] 
        t2_start = timestamp[tripleg2['start']]
        t2_end = timestamp[tripleg2['end']]
        
        dt_start = abs(t1_start - t2_start)
        dt_end = abs(t1_end - t2_end)
        
        dl_start = sphericalDistance(
            lng[tripleg1['start']], lat[tripleg1['start']], 
            lng[tripleg2['start']], lat[tripleg2['start']]
        )
        dl_end = sphericalDistance(
            lng[tripleg1['end']], lat[tripleg1['end']], 
            lng[tripleg2['end']], lat[tripleg2['end']]
        )
        
        return (dt_start <= self.t_thr or dl_start <= self.d_thr) and (dt_end <= self.t_thr or dl_end <= self.d_thr)

    def eval_tripleg(self, seq_infos):
        '''
        Count recall and precision of detected triplegs. A tripleg is correct when its endpoints meet the time or distance threshold and the mode label matches.

        params:
            seq_infos: trajectory table with columns ['uid', 'tid', 'timestamp', 'lng', 'lat', 'mode', 'mode_pred']
            t_thr: time threshold for a tripleg match, in seconds
            d_thr: distance threshold for a tripleg match, in meters
        '''
        count_tp_list = []
        count_fp_list = []
        count_fn_list = []

        for tid, traj in seq_infos.groupby('tid'):
            label_true = traj[self.col_true].values
            label_pred = traj[self.col_pred].values
            timestamp = traj['timestamp'].values
            lng = traj['lng'].values
            lat = traj['lat'].values

            true_triplegs = self._get_triplegs(label_true)
            pred_triplegs = self._get_triplegs(label_pred)

            tp = 0
            for tripleg1 in true_triplegs.copy():
                for tripleg2 in pred_triplegs.copy():
                    if self._is_tripleg_match(tripleg1, tripleg2, timestamp, lng, lat):
                        tp += 1
                        true_triplegs.remove(tripleg1)
                        pred_triplegs.remove(tripleg2)
                        break
            
            # counts
            fn = len(true_triplegs)
            fp = len(pred_triplegs)

            count_tp_list.append(tp)
            count_fp_list.append(fp)
            count_fn_list.append(fn)

        tripleg_metrics = dict(
            count_tp=np.sum(count_tp_list),
            count_fp=np.sum(count_fp_list),
            count_fn=np.sum(count_fn_list)
        )
        return tripleg_metrics
    
    
    # -- mode chain --
    def eval_mode_chain(self, seq_infos):
        M = np.unique(seq_infos[self.col_true]).shape[0]
        ntraj = seq_infos.groupby('tid').size().shape[0]
        
        # for original mode chain metrics
        ntleg_true = 0
        ntleg_pred = 0
        ntrans_true = 0
        ntrans_pred = 0
        abserr_trans_total = 0
        ntrans_overlap = 0

        # for instance-based 2-gram metrics
        count_tp_2gram = 0
        count_true_2gram = 0
        count_pred_2gram = 0

        # # --traverse trajectories--
        for tid,traj in seq_infos.groupby('tid'):
            # # --initialization--
            y_trip = traj[self.col_true].values
            y_pred = traj[self.col_pred].values
            timestamp = traj['timestamp'].values
            lng = traj['lng'].values
            lat = traj['lat'].values
            
            # # -- instance-based 2-gram evaluation --
            # 1. Find true 2-grams
            true_2grams = []
            for i in range(len(y_trip) - 1):
                if y_trip[i] != y_trip[i+1]:
                    true_2grams.append({'index': i, 'mode_before': y_trip[i], 'mode_after': y_trip[i+1]})
            count_true_2gram += len(true_2grams)

            # 2. Find predicted 2-grams
            pred_2grams = []
            for i in range(len(y_pred) - 1):
                if y_pred[i] != y_pred[i+1]:
                    pred_2grams.append({'index': i, 'mode_before': y_pred[i], 'mode_after': y_pred[i+1]})
            count_pred_2gram += len(pred_2grams)
            
            # 3. Match true and predicted 2-grams
            tp_2gram = 0
            for true_2g in true_2grams.copy():
                for pred_2g in pred_2grams.copy():
                    # Condition 1: transfer points are close
                    dt = abs(timestamp[true_2g['index']+1] - timestamp[pred_2g['index']+1])
                    dl = sphericalDistance(lng[true_2g['index']+1], lat[true_2g['index']+1], lng[pred_2g['index']+1], lat[pred_2g['index']+1])
                    is_close = dt <= self.t_thr or dl <= self.d_thr

                    # Condition 2: modes before and after are correct
                    modes_match = (true_2g['mode_before'] == pred_2g['mode_before']) and \
                                  (true_2g['mode_after'] == pred_2g['mode_after'])

                    if is_close and modes_match:
                        tp_2gram += 1
                        true_2grams.remove(true_2g)
                        pred_2grams.remove(pred_2g)
                        break # Move to the next true 2-gram

            count_tp_2gram += tp_2gram

            # # -- original mode chain logic --
            trans_mat_true = np.zeros((M+1,M),dtype=int) # [start_trans|middle_trans]
            trans_mat_pred = np.zeros((M+1,M),dtype=int)
            # # --start transition--
            trans_mat_true[0,int(y_trip[0])] += 1
            trans_mat_pred[0,int(y_pred[0])] += 1
            # # --middle transition
            for i,j in zip(y_trip[:-1],y_trip[1:]):
                if i!=j:
                    trans_mat_true[int(i)+1,int(j)] += 1
            for i,j in zip(y_pred[:-1],y_pred[1:]):
                if i!=j:
                    trans_mat_pred[int(i)+1,int(j)] += 1
            # # --total absolute error, 2-gram overlap--
            ntleg_true += np.sum(trans_mat_true)
            ntleg_pred += np.sum(trans_mat_pred)
            ntrans_true += np.sum(trans_mat_true)
            ntrans_pred += np.sum(trans_mat_pred)
            abserr_trans_total += np.sum(np.abs(trans_mat_true - trans_mat_pred))
            ntrans_overlap += np.sum(np.minimum(trans_mat_true, trans_mat_pred))

        # # -- original metrics calculation --
        mae_transfer = abserr_trans_total / ntrans_true
        recall_2gram_mat = ntrans_overlap / ntrans_true if ntrans_true > 0 else 0
        precision_2gram_mat = ntrans_overlap / ntrans_pred if ntrans_pred > 0 else 0
        f1score_2gram_mat = 2 * recall_2gram_mat * precision_2gram_mat / (recall_2gram_mat + precision_2gram_mat) if (recall_2gram_mat + precision_2gram_mat) > 0 else 0

        # # -- instance-based 2-gram metrics calculation --
        recall_2gram = count_tp_2gram / count_true_2gram if count_true_2gram > 0 else 0
        precision_2gram = count_tp_2gram / count_pred_2gram if count_pred_2gram > 0 else 0
        f1score_2gram = 2 * recall_2gram * precision_2gram / (recall_2gram + precision_2gram) if (recall_2gram + precision_2gram) > 0 else 0

        # --summarize--
        mode_chain = dict(
            ntraj=ntraj, ntleg_true=ntleg_true, ntleg_pred=ntleg_pred, mae_transfer=mae_transfer,
            ntrans_overlap=ntrans_overlap, ntrans_true=ntrans_true, ntrans_pred=ntrans_pred,
            recall_2gram_mat=recall_2gram_mat, precision_2gram_mat=precision_2gram_mat, f1score_2gram_mat=f1score_2gram_mat,
            recall_2gram=recall_2gram, precision_2gram=precision_2gram, f1score_2gram=f1score_2gram
        )
        return mode_chain
    

    def metric_calculation_pipeline(self, seq_infos):
        '''
        Compute every evaluation metric.

        params:
            seq_infos: trajectory table
        '''
        correctness = self.eval_correctness(seq_infos)
        transfer = self.eval_transfer(seq_infos)
        tripleg = self.eval_tripleg(seq_infos)
        mode_chain = self.eval_mode_chain(seq_infos)
        return correctness, transfer, tripleg, mode_chain


def evaluate_from_sequence(seq_infos, t_thr=120, d_thr=100):
    evaluator = Evaluator(col_true='mode', col_pred='mode_pred', t_thr=t_thr, d_thr=d_thr)
    correctness, transfer, tripleg, mode_chain = evaluator.metric_calculation_pipeline(seq_infos)
    return correctness, transfer, tripleg, mode_chain
        
