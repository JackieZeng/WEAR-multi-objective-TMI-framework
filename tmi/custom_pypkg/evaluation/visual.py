# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 11:11:27 2022

@author: Jackie
"""
import numpy as np
import pandas as pd

    
def print_evaluation(correctness=None, transfer=None, tripleg=None, mode_chain=None, validate_with=None, logger=None, show_all=False):
    printf = logger.log_string if logger is not None else print
    if validate_with is not None:
        printf('Validate with {}'.format(validate_with))
    # --correctness--
    if correctness is not None:
        # # point-based
        acc = correctness.get('acc')
        f1score = correctness.get('f1score')
        conf = correctness.get('conf')
        printf('---- Correctness ----')
        printf('acc:\t**{:.2f}%**   f1-score:\t**{:.2f}%**'.format(100*acc,100*f1score), '\n', conf) 
        if show_all:
            printf('Confusion Matrix - point')
            printf(np.around(conf / conf.sum(axis=1,keepdims=True) * 100,1))
        
    if transfer is not None:
        cnt_tp_tsf = transfer.get('count_tp')
        cnt_fn_tsf = transfer.get('count_fn')
        cnt_fp_tsf = transfer.get('count_fp')
        recall = cnt_tp_tsf / (cnt_tp_tsf + cnt_fn_tsf)
        precision = cnt_tp_tsf / (cnt_tp_tsf + cnt_fp_tsf)
        f1score = 2 * recall * precision / (recall + precision)
        printf('---- Transfer ----')
        printf(f'Recall: {100*recall:.2f}%, Precision: {100*precision:.2f}%, F1-score: {100*f1score:.2f}%')
        printf('Confusion Matrix - transfer')
        printf(np.array([[cnt_tp_tsf, cnt_fn_tsf], [cnt_fp_tsf, 0]]))

    if tripleg is not None:
        cnt_tp_tlg = tripleg.get('count_tp')
        cnt_fn_tlg = tripleg.get('count_fn')
        cnt_fp_tlg = tripleg.get('count_fp')
        recall = cnt_tp_tlg / (cnt_tp_tlg + cnt_fn_tlg)
        precision = cnt_tp_tlg / (cnt_tp_tlg + cnt_fp_tlg)
        f1score = 2 * recall * precision / (recall + precision)
        printf('---- Tripleg CLF ----')
        printf(f'Recall: {100*recall:.2f}%, Precision: {100*precision:.2f}%, F1-score: {100*f1score:.2f}%')
        printf('Confusion Matrix - tripleg')
        printf(np.array([[cnt_tp_tlg, cnt_fn_tlg], [cnt_fp_tlg, 0]]))

    if mode_chain is not None:
        ntraj = mode_chain.get('ntraj')
        ntleg_true = mode_chain.get('ntleg_true')
        ntleg_pred = mode_chain.get('ntleg_pred')
        mae_transfer = mode_chain.get('mae_transfer')
        recall_2gram = mode_chain.get('recall_2gram')
        precision_2gram = mode_chain.get('precision_2gram')
        f1score_2gram = mode_chain.get('f1score_2gram')
        printf('-- Mode Chain --')
        printf(f'Number of trajectories: {ntraj:.0f}')
        printf(f'Number of true leg: {ntleg_true:.0f}')
        printf(f'Number of pred leg: {ntleg_pred:.0f}')
        printf(f'MAE of transfer: {mae_transfer:.6f}')
        printf(f'Recall of 2-gram: {100*recall_2gram:.2f}%, Precision of 2-gram: {100*precision_2gram:.2f}%, F1-score of 2-gram: {100*f1score_2gram:.2f}%')

# ==== Generate DataFrame for Save ====
def generate_evaluation_dataframe(correctness, transfer=None, tripleg=None, mode_chain=None):
    # --read metrics--
    acc = correctness.get('acc')
    f1score = correctness.get('f1score')
    conf = correctness.get('conf')
    
    if transfer is not None:
        cnt_tp_tsf = transfer.get('count_tp')
        cnt_fn_tsf = transfer.get('count_fn')
        cnt_fp_tsf = transfer.get('count_fp')

    if tripleg is not None:
        cnt_tp_tlg = tripleg.get('count_tp')
        cnt_fn_tlg = tripleg.get('count_fn')
        cnt_fp_tlg = tripleg.get('count_fp')

    if mode_chain is not None:
        ntraj = mode_chain.get('ntraj')
        ntleg_true = mode_chain.get('ntleg_true')
        ntleg_pred = mode_chain.get('ntleg_pred')
        mae_transfer = mode_chain.get('mae_transfer')
        ntrans_overlap = mode_chain.get('ntrans_overlap')
        ntrans_true = mode_chain.get('ntrans_true')
        ntrans_pred = mode_chain.get('ntrans_pred')
        recall_2gram_mat = mode_chain.get('recall_2gram_mat')
        precision_2gram_mat = mode_chain.get('precision_2gram_mat')
        f1score_2gram_mat = mode_chain.get('f1score_2gram_mat')
        recall_2gram = mode_chain.get('recall_2gram')
        precision_2gram = mode_chain.get('precision_2gram')
        f1score_2gram = mode_chain.get('f1score_2gram')

    travel_modes = ['m'+str(i) for i in range(conf.shape[1])]
    # # --point--
    conf_df = pd.DataFrame(conf, index=travel_modes, columns=travel_modes)
    recall_df = pd.DataFrame(
        np.array([[conf[i,i]] for i in range(len(conf))]) / conf.sum(axis=1, keepdims=True),
        index=travel_modes, columns=['recall']
    )
    precis_df = pd.DataFrame(
        np.array([conf[i,i] for i in range(len(conf))]) / conf.sum(axis=0, keepdims=True),
        index=['precision'],columns=travel_modes
    )
    metrics = pd.DataFrame(
        [[acc, f1score]], index=['precision'], columns=['accuracy','f1-score'],
    )
    
    if transfer is not None:
        transfer_df = pd.DataFrame(
            [[cnt_tp_tsf, cnt_fn_tsf, cnt_fp_tsf]], index=['precision'],
            columns=['cnt_tp_tsf', 'cnt_fn_tsf', 'cnt_fp_tsf'],
        )
        metrics = pd.concat([metrics, transfer_df], axis=1)
    
    if tripleg is not None:
        tripleg_df = pd.DataFrame(
            [[cnt_tp_tlg, cnt_fn_tlg, cnt_fp_tlg]], index=['precision'],
            columns=['cnt_tp_tlg', 'cnt_fn_tlg', 'cnt_fp_tlg'],
        )
        metrics = pd.concat([metrics, tripleg_df], axis=1)
        
    if mode_chain is not None:
        mode_chain_df = pd.DataFrame(
            [[ntraj, ntleg_true, ntleg_pred, mae_transfer, ntrans_overlap, ntrans_true, ntrans_pred, recall_2gram_mat, precision_2gram_mat, f1score_2gram_mat, recall_2gram, precision_2gram, f1score_2gram]], index=['precision'],
            columns=['ntraj', 'ntleg_true', 'ntleg_pred', 'mae_transfer', 'ntrans_overlap', 'ntrans_true', 'ntrans_pred', 'recall_2gram_mat', 'precision_2gram_mat', 'f1score_2gram_mat', 'recall_2gram', 'precision_2gram', 'f1score_2gram'],
        )
        metrics = pd.concat([metrics, mode_chain_df], axis=1)

    conf_df = pd.concat([conf_df, precis_df], axis=0)
    conf_df = pd.concat([conf_df, recall_df, metrics], axis=1)
    conf_df['metric'] = 'point'

    # # --concatenate--
    result_df = (
        conf_df
        .reset_index()
        .rename(columns={'index':'actual'})
    )
        
    return result_df



# ==== Save Evaluations ====
def save_evaluations(path, correctness, transfer=None, tripleg=None, mode_chain=None):
    try: 
        # only one result
        result_df = generate_evaluation_dataframe(correctness, transfer, tripleg, mode_chain)
        cols_index = ['metric','actual']
    except: 
        # multiple results
        result_df = []
        for key in correctness.keys():
            corre_cut = correctness[key]
            trans_cut = transfer[key] if transfer is not None else None
            tpleg_cut = tripleg[key] if tripleg is not None else None
            modec_cut = mode_chain[key] if mode_chain is not None else None
            res_cut = generate_evaluation_dataframe(corre_cut, trans_cut, tpleg_cut, modec_cut)
            res_cut['index'] = key
            cols_index = ['index', 'metric', 'actual']
            result_df.append(res_cut)
        result_df = pd.concat(result_df)
    
    result_df = result_df.set_index(cols_index)
    result_df.to_csv(path)
