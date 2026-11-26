# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 16:41:50 2021

@author: jackie
"""
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from adjustText import adjust_text

def create_path(path):
    import os
    if type(path) != str:
        return False
    # whether the path already exists
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
        return True
    else:
        return False
    

def plot_adjust_limit(ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]

    xlim = (xlim[0] - x_range * 0.1, xlim[1] + x_range * 0.1)
    ylim = (ylim[0] - y_range * 0.1, ylim[1] + y_range * 0.2)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)



def plot_loss_relation_lines(ax, loss_relations, score_x, score_y, score_overall):

    for loss_rela in loss_relations:
        loss_curr = loss_rela[0]
        loss_next = loss_rela[1]
        if loss_curr not in loss_desc_list or loss_next not in loss_desc_list:
            continue
        idx_curr = loss_desc_list.index(loss_curr)
        idx_next = loss_desc_list.index(loss_next)

        if 'W' not in loss_curr and 'W' in loss_next: 
            key = 'W'
        elif 'A' not in loss_curr and 'A' in loss_next:
            key = 'A'
        else:
            key = 'R'
        arrowprops = loss_relation_arrowprops[key]
        ax.annotate('', xy=(score_x[idx_next], score_y[idx_next]), 
                   xytext=(score_x[idx_curr], score_y[idx_curr]),
                   arrowprops=arrowprops)

    return


def plot_add_text(ax, score_x, score_y, all_losses):
    texts = []
    for x, y ,txt in zip(score_x, score_y, all_losses):
        text = ax.text(x, y, txt, ha='center', va='bottom', fontsize=7, 
                        bbox=dict(facecolor='white', edgecolor='lightgray', boxstyle='round', alpha=0.85, pad=0.2, linewidth=0.2))
        texts.append(text)

    return texts



def plot_contour(ax, better='higher'):
    cmap = 'coolwarm_r' if better == 'higher' else 'coolwarm'
    # keep the current xlim and ylim
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # contour field over the current axis limits
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100) 
    X, Y = np.meshgrid(x, y)
    
    # F1 score
    F1 = 2 * X * Y / (X + Y)
    F1[np.isnan(F1)] = 0  # replace NaN from division by zero
    
    # draw the contour field
    ax.contourf(X, Y, F1, levels=5, cmap=cmap, alpha=0.2)
    contour = ax.contour(X, Y, F1, levels=5, colors='grey', alpha=0.7, linewidths=0.4)
    ax.clabel(contour, inline=True, fontsize=8)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


# == Other parameters for iteration ==
seed_list = [0,1,2]

split_list = ['usersplit', 'trajsplit']


loss_param_list = [
    # E + ER
    {'alpha': 0, 'use_adj': False},

    # WE + WER
    {'alpha': 0.5, 'use_adj': False},

    # EA + EAR
    {'alpha': 0, 'use_adj': True},

    # WEA + WERA
    {'alpha': 0.5, 'use_adj': True},
]

loss_dict = {}
loss_relations = []
for loss_param in loss_param_list:
    w_alpha = loss_param['alpha']
    use_adj = loss_param['use_adj']

    suffix_param = f'W({w_alpha})E'
    if use_adj:
        suffix_param += 'A'
    
    desc_loss_param = ''
    if loss_param['alpha'] != 0:
        desc_loss_param += 'W'
    desc_loss_param += 'E'
    if loss_param['use_adj']:
        desc_loss_param += 'A'

    loss_dict[desc_loss_param] = (suffix_param, loss_param)
    loss_dict[desc_loss_param + 'R'] = (suffix_param, loss_param)
    
    loss_relations.append([desc_loss_param, desc_loss_param + 'R'])


loss_relations += [
    # no R
    ['E', 'WE'],
    ['E', 'EA'],
    ['WE', 'WEA'],
    ['EA', 'WEA'],
    # to R
    ['E', 'ER'],
    ['WE', 'WER'],
    ['EA', 'EAR'],
    ['WEA', 'WEAR'],
    # with R
    ['ER', 'WER'],
    ['ER', 'EAR'],
    ['WER', 'WEAR'],
    ['EAR', 'WEAR'],
]

loss_relation_arrowprops = {
    'W': dict(arrowstyle='->,head_length=0.3,head_width=0.15', color=np.array([158,49,80])/256, alpha=0.8, lw=0.8, linestyle=':'),
    'A': dict(arrowstyle='->,head_length=0.3,head_width=0.15', color=np.array([250,134,0])/256, alpha=0.8, lw=0.8, linestyle=':'),
    'R': dict(arrowstyle='->,head_length=0.3,head_width=0.15', color=np.array([78,171,144])/256, alpha=0.8, lw=0.8, linestyle=':'),
}

cols_metric = [
    'accuracy', 'f1-score', 
    'cnt_tp_tsf', 'cnt_fn_tsf', 'cnt_fp_tsf', 
    'cnt_tp_tlg', 'cnt_fn_tlg', 'cnt_fp_tlg', 
    'recall_2gram_mat', 'precision_2gram_mat', 'f1score_2gram_mat',
    'recall_2gram', 'precision_2gram', 'f1score_2gram']


mname = 'c6-r3-crf'
model_path = SCRIPT_DIR / 'result records (auto-save)'

use_acc = False

for isp, (split) in enumerate(split_list):

    result_all = {}
    text_dict = {i: [] for i in range(4)}
    for desc_loss_param, (suffix_param, loss_param) in loss_dict.items():
        # desc: short loss name; suffix: filename suffix; param: loss settings
        rec_path = model_path / split

        result_all[desc_loss_param] = {}
        result_seeds = dict(
            acc = [],
            f1 = [],
            recall_clf = [],
            precision_clf = [],
            recall_tsf = [],
            precision_tsf = [],
            recall_tlg = [],
            precision_tlg = [],
            recall_2gram = [],
            precision_2gram = [],
            f1score_2gram = [],
        )
        for seed in seed_list:
            result_df = pd.read_csv(rec_path / f'({mname})-{suffix_param}-s{seed}.csv')

            # 
            if 'R' not in desc_loss_param:
                index = 'not_crf'
            else:
                index = 'use_crf'

            res = result_df[result_df['index'] == index][cols_metric].dropna().iloc[0]
            recall = result_df[result_df['index'] == index]['recall'].dropna().values.flatten()
            precision = result_df[(result_df['index'] == index) & (result_df['actual'] == 'precision')][[f'm{i}' for i in range(len(recall))]].values.flatten()
            
            #
            acc = res['accuracy'] * 100
            f1 = res['f1-score'] * 100
            recall_clf = recall.mean() * 100
            precision_clf = precision.mean() * 100
            recall_tsf = res['cnt_tp_tsf'] / (res['cnt_tp_tsf'] + res['cnt_fn_tsf']) * 100
            precision_tsf = res['cnt_tp_tsf'] / (res['cnt_tp_tsf'] + res['cnt_fp_tsf']) * 100
            recall_tlg = res['cnt_tp_tlg'] / (res['cnt_tp_tlg'] + res['cnt_fn_tlg']) * 100
            precision_tlg = res['cnt_tp_tlg'] / (res['cnt_tp_tlg'] + res['cnt_fp_tlg']) * 100
            recall_2gram = res['recall_2gram_mat'] * 100
            precision_2gram = res['precision_2gram_mat'] * 100
            f1score_2gram = res['f1score_2gram_mat'] * 100

            result_seeds['acc'].append(acc)
            result_seeds['f1'].append(f1)
            result_seeds['recall_clf'].append(recall_clf)
            result_seeds['precision_clf'].append(precision_clf)
            result_seeds['recall_tsf'].append(recall_tsf)
            result_seeds['precision_tsf'].append(precision_tsf)
            result_seeds['recall_tlg'].append(recall_tlg)
            result_seeds['precision_tlg'].append(precision_tlg)
            result_seeds['recall_2gram'].append(recall_2gram)
            result_seeds['precision_2gram'].append(precision_2gram)
            result_seeds['f1score_2gram'].append(f1score_2gram)

        result_all[desc_loss_param] = result_seeds

    use_two_rows = True
    if use_two_rows:
        fig, axes = plt.subplots(2, 2, figsize=(12,6.5), gridspec_kw={'wspace':0.12, 'hspace':0.24})
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 4, figsize=(12,2.7), gridspec_kw={'wspace':0.18})

    output_dict = {}

    #%% Accuracy & F1-score
    ax = axes[0]
    ax.set_title("(a) Point label")

    if use_acc:
        # aggregate scores
        loss_desc_list = list(loss_dict.keys())
        acc_arr = np.array([result_all[desc_loss_param]['acc'] for desc_loss_param in loss_desc_list])
        f1_arr = np.array([result_all[desc_loss_param]['f1'] for desc_loss_param in loss_desc_list])
        harmonic_arr = np.array([2 * acc * f1 / (acc + f1) for acc, f1 in zip(acc_arr, f1_arr)])

        acc_avg_arr = np.mean(acc_arr, axis=1)
        f1_avg_arr = np.mean(f1_arr, axis=1)
        harmonic_avg_arr = 2 * acc_avg_arr * f1_avg_arr / (acc_avg_arr + f1_avg_arr)
        harmonic_std_arr = np.std(harmonic_arr, axis=1)

        output_dict['acc'] = {desc: val for desc, val in zip(loss_desc_list, acc_avg_arr)}
        output_dict['f1'] = {desc: val for desc, val in zip(loss_desc_list, f1_avg_arr)}
        output_dict['harmonic'] = {desc: val for desc, val in zip(loss_desc_list, harmonic_avg_arr)}
        output_dict['harmonic_std'] = {desc: val for desc, val in zip(loss_desc_list, harmonic_std_arr)}

        # scatter plot
        ax.plot(acc_avg_arr, f1_avg_arr, 'o', c='black', ms=1.5)

        # loss-relation arrows
        plot_loss_relation_lines(ax, loss_relations, acc_avg_arr, f1_avg_arr, harmonic_avg_arr)

        # text labels
        texts = plot_add_text(ax, acc_avg_arr, f1_avg_arr, loss_desc_list)
        text_dict[0] += texts

        if use_two_rows:
            ax.set_ylabel('F1-score (%)')
        if not use_two_rows:
            ax.set_xlabel('Accuracy (%)')
            ax.set_ylabel('F1-score (%)')

    else:
        loss_desc_list = list(loss_dict.keys())
        acc_arr = np.array([result_all[desc_loss_param]['acc'] for desc_loss_param in loss_desc_list])
        recall_clf_arr = np.array([result_all[desc_loss_param]['recall_clf'] for desc_loss_param in loss_desc_list])
        precision_clf_arr = np.array([result_all[desc_loss_param]['precision_clf'] for desc_loss_param in loss_desc_list])
        my_f1_arr = np.array([2 * recall_clf * precision_clf / (recall_clf + precision_clf) for recall_clf, precision_clf in zip(recall_clf_arr, precision_clf_arr)])
        macro_f1_arr = np.array([result_all[desc_loss_param]['f1'] for desc_loss_param in loss_desc_list])
        pseudo_recall_clf_arr = recall_clf_arr * macro_f1_arr / my_f1_arr
        pseudo_precision_clf_arr = precision_clf_arr * macro_f1_arr / my_f1_arr

        acc_avg_arr = np.mean(acc_arr, axis=1)
        acc_std_arr = np.std(acc_arr, axis=1)
        pseudo_recall_avg_arr = np.mean(pseudo_recall_clf_arr, axis=1)
        pseudo_precision_avg_arr = np.mean(pseudo_precision_clf_arr, axis=1)
        pseudo_harmonic_avg_arr = np.mean(macro_f1_arr, axis=1)
        pseudo_harmonic_std_arr = np.std(macro_f1_arr, axis=1)

        output_dict['acc'] = {desc: val for desc, val in zip(loss_desc_list, acc_avg_arr)}
        output_dict['acc_std'] = {desc: val for desc, val in zip(loss_desc_list, acc_std_arr)}
        output_dict['pseudo_recall'] = {desc: val for desc, val in zip(loss_desc_list, pseudo_recall_avg_arr)}
        output_dict['pseudo_precision'] = {desc: val for desc, val in zip(loss_desc_list, pseudo_precision_avg_arr)}
        output_dict['macro_f1'] = {desc: val for desc, val in zip(loss_desc_list, pseudo_harmonic_avg_arr)}
        output_dict['pseudo_harmonic_std'] = {desc: val for desc, val in zip(loss_desc_list, pseudo_harmonic_std_arr)}

        # scatter plot
        ax.plot(pseudo_recall_avg_arr, pseudo_precision_avg_arr, 'o', c='black', ms=1.5)

        # loss-relation arrows
        plot_loss_relation_lines(ax, loss_relations, pseudo_recall_avg_arr, pseudo_precision_avg_arr, pseudo_harmonic_avg_arr)

        # text labels
        texts = plot_add_text(ax, pseudo_recall_avg_arr, pseudo_precision_avg_arr, loss_desc_list)
        text_dict[0] += texts

        if use_two_rows:
            ax.set_ylabel('Precision (%)')
        if not use_two_rows:
            ax.set_xlabel('Recall (%)')
            ax.set_ylabel('Precision (%)')

    
    #%% Transfer

    ax = axes[1]
    ax.set_title("(b) Transfer")

    # aggregate scores
    recall_tsf_arr = np.array([result_all[desc_loss_param]['recall_tsf'] for desc_loss_param in loss_desc_list])
    precision_tsf_arr = np.array([result_all[desc_loss_param]['precision_tsf'] for desc_loss_param in loss_desc_list])
    f1_tsf_arr = 2 * recall_tsf_arr * precision_tsf_arr / (recall_tsf_arr + precision_tsf_arr)

    recall_tsf_avg_arr = np.mean(recall_tsf_arr, axis=1)
    precision_tsf_avg_arr = np.mean(precision_tsf_arr, axis=1)
    f1_tsf_avg_arr = np.mean(f1_tsf_arr, axis=1)
    f1_tsf_std_arr = np.std(f1_tsf_arr, axis=1)

    output_dict['recall_tsf'] = {desc: val for desc, val in zip(loss_desc_list, recall_tsf_avg_arr)}
    output_dict['precision_tsf'] = {desc: val for desc, val in zip(loss_desc_list, precision_tsf_avg_arr)}
    output_dict['f1_tsf'] = {desc: val for desc, val in zip(loss_desc_list, f1_tsf_avg_arr)}
    output_dict['f1_tsf_std'] = {desc: val for desc, val in zip(loss_desc_list, f1_tsf_std_arr)}

    # scatter plot
    ax.plot(recall_tsf_avg_arr, precision_tsf_avg_arr, 'o', c='black', ms=1.5)

    # loss-relation arrows
    plot_loss_relation_lines(ax, loss_relations, recall_tsf_avg_arr, precision_tsf_avg_arr, f1_tsf_avg_arr)
    
    # best loss-combination path

    # text labels
    texts = plot_add_text(ax, recall_tsf_avg_arr, precision_tsf_avg_arr, loss_desc_list)
    text_dict[1] += texts

    # axis labels
    if use_two_rows:
        pass
    else:
        ax.set_xlabel('Recall (%)')

    #%% Tripleg

    ax = axes[2]
    ax.set_title("(c) Tripleg")

    # aggregate scores
    recall_tlg_arr = np.array([result_all[desc_loss_param]['recall_tlg'] for desc_loss_param in loss_desc_list])
    precision_tlg_arr = np.array([result_all[desc_loss_param]['precision_tlg'] for desc_loss_param in loss_desc_list])
    f1_tlg_arr = 2 * recall_tlg_arr * precision_tlg_arr / (recall_tlg_arr + precision_tlg_arr)

    recall_tlg_avg_arr = np.mean(recall_tlg_arr, axis=1)
    precision_tlg_avg_arr = np.mean(precision_tlg_arr, axis=1)
    f1_tlg_avg_arr = np.mean(f1_tlg_arr, axis=1)
    f1_tlg_std_arr = np.std(f1_tlg_arr, axis=1)

    output_dict['recall_tlg'] = {desc: val for desc, val in zip(loss_desc_list, recall_tlg_avg_arr)}
    output_dict['precision_tlg'] = {desc: val for desc, val in zip(loss_desc_list, precision_tlg_avg_arr)}
    output_dict['f1_tlg'] = {desc: val for desc, val in zip(loss_desc_list, f1_tlg_avg_arr)}
    output_dict['f1_tlg_std'] = {desc: val for desc, val in zip(loss_desc_list, f1_tlg_std_arr)}


    # scatter plot
    ax.plot(recall_tlg_avg_arr, precision_tlg_avg_arr, 'o', c='black', ms=1.5)

    # loss-relation arrows
    plot_loss_relation_lines(ax, loss_relations, recall_tlg_avg_arr, precision_tlg_avg_arr, f1_tlg_avg_arr)

    # best loss-combination path

    # text labels
    texts = plot_add_text(ax, recall_tlg_avg_arr, precision_tlg_avg_arr, loss_desc_list)
    text_dict[2] += texts
    
    # axis labels
    if use_two_rows:
        ax.set_xlabel('Recall (%)')
        ax.set_ylabel('Precision (%)')
    else:
        ax.set_xlabel('Recall (%)')

    #%% Mode chain

    ax = axes[3]
    ax.set_title("(d) Mode chain")

    # aggregate scores
    recall_2gram_arr = np.array([result_all[desc_loss_param]['recall_2gram'] for desc_loss_param in loss_desc_list])
    precision_2gram_arr = np.array([result_all[desc_loss_param]['precision_2gram'] for desc_loss_param in loss_desc_list])
    harmonic_arr = 2 * recall_2gram_arr * precision_2gram_arr / (recall_2gram_arr + precision_2gram_arr)    

    recall_2gram_avg_arr = np.mean(recall_2gram_arr, axis=1)
    precision_2gram_avg_arr = np.mean(precision_2gram_arr, axis=1)
    harmonic_avg_arr = np.mean(harmonic_arr, axis=1)
    harmonic_std_arr = np.std(harmonic_arr, axis=1)

    output_dict['recall_2gram'] = {desc: val for desc, val in zip(loss_desc_list, recall_2gram_avg_arr)}
    output_dict['precision_2gram'] = {desc: val for desc, val in zip(loss_desc_list, precision_2gram_avg_arr)}
    output_dict['harmonic'] = {desc: val for desc, val in zip(loss_desc_list, harmonic_avg_arr)}
    output_dict['harmonic_std'] = {desc: val for desc, val in zip(loss_desc_list, harmonic_std_arr)}

    # scatter plot
    ax.plot(recall_2gram_avg_arr, precision_2gram_avg_arr, 'o', c='black', ms=1.5)

    # loss-relation arrows
    plot_loss_relation_lines(ax, loss_relations, recall_2gram_avg_arr, precision_2gram_avg_arr, harmonic_avg_arr)

    # best loss-combination path

    # text labels
    texts = plot_add_text(ax, recall_2gram_avg_arr, precision_2gram_avg_arr, loss_desc_list)    
    text_dict[3] += texts
    
    # axis labels
    if use_two_rows:
        ax.set_xlabel('Recall (%)')
    else:
        ax.set_xlabel('Recall (%)')

    #%% Save
    for i, ax in enumerate(axes): 
        texts_i = text_dict[i]
        adjust_text(texts_i, ax=ax, expand=(1.4, 2.0),
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, shrinkA=0))
        plot_adjust_limit(ax)
        
        # integer y-axis ticks
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # draw the contour field
        plot_contour(ax)
        
    create_path(str(SCRIPT_DIR / 'figures') + os.sep)
    plt.savefig(SCRIPT_DIR / 'figures' / f'{split}-LC-({mname}).png', bbox_inches='tight', dpi=600)
    plt.close()


    #%% Save output_dict
    output_df = pd.DataFrame(output_dict).round(2).reset_index()
    output_df.to_csv(SCRIPT_DIR / 'result records (auto-save)' / f'{split}-LC-({mname}).csv', index=False)

