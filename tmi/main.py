# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 16:41:50 2021

@author: jackie
"""
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from custom_pypkg.utils import loading
from custom_pypkg.utils import initial
from custom_pypkg.utils import config

from custom_pypkg.preprocessing import scalars
from custom_pypkg.model import classifiers
from custom_pypkg.training import optimization, trainers
from custom_pypkg.evaluation import metrics, visual, eval_pipeline


import torch

import pandas as pd
import datetime
from itertools import product

import warnings
warnings.filterwarnings('ignore')

# == Device of training ==
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

M = 5
num_folds = 3

cols = ['dt','dl','v','a','w','distmin','nstops','distrd','distrl']
test_dict = dict(
    use_crf = {'batch_size': 32, 'use_crf': True},
    not_crf = {'batch_size': 32, 'use_crf': False},
) 

# == Other parameters for iteration ==
seed_list = [0,1,2]

split_list = ['trajsplit', 'usersplit']


w_alpha_default = 0.5

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

caseiter = product(seed_list, split_list, loss_param_list)
total_cases = sum([1 for _ in caseiter])

#%% ==== Start hyper-parameter testing ==== 
caseiter = product(seed_list, split_list, loss_param_list)

for icase, (seed, split, loss_param) in enumerate(caseiter):
    w_alpha = loss_param.get('alpha', w_alpha_default)
    use_adj = loss_param.get('use_adj', False)

    # define loss_params
    if use_adj:
        loss_params = {
            'ce': {'alpha': w_alpha},
            'adj': {'alpha': w_alpha},
            'crf': {},
        }
    else:
        loss_params = {
            'ce': {'alpha': w_alpha},
            'crf': {},
        }

    # define description of loss_params
    suffix_param = f'W({w_alpha})E'
    if use_adj:
        suffix_param += 'A'

    #%% -- initial parameters --

    # # set logging info
    log_path = str(SCRIPT_DIR / 'log') + os.sep
    time_now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_name = f'[Log-{time_now}]-{suffix_param}-{split}-s{seed}.txt'
    logger = config.Logger(log_path, log_name)
    
    # # config hyperparameters
    args = config.parse_args(
        logger = logger, seed=seed, 
        
        # -- sequence argumetns --
        feat_cols = cols, 
        
        # -- data arguments --
        M = M, C = len(cols), 
        
        # -- model arguments --
        num_conv=6, dim_conv=128, kernel_size=3, dim_out_cnn=128, pdrop_cnn=0.5,
        dim_hidden = 128, pdrop_agg=0.5,
        n_rnn = 3, pdrop_rnn=0.3, 
        
        # -- dataloader arguments --
        cutter_kwargs = dict(nsmp_max = 8*256, max_empty_ratio = 0.6, B_min=3, sort=True),

        # -- training arguments --
        schedule_kwargs = dict(policy = 'warmup', step_peak = 2, step_half = 4, peak=1),
        max_epoch = 15,
        B = 64, 
        lr = 1e-3, tol = 1e-6, 
        loss_params = loss_params,
        
        # -- trainer arguments --
        verbose = 1, n_iter_print = 1,
        
        # -- evaluation arguments --
        t_thr = 120, d_thr = 100,
        
        # -- file arguments --
        root_data = str(REPO_ROOT / 'data') + os.sep, 
    )
    # # logging config
    args_str = ['Namespace(\n'] + ['\t' + ' = '.join(map(str, kv)) + ',\n' for kv in args._get_kwargs()] + [')']
    args_str = ''.join(args_str)
    args.logger.log_string(args_str)
    
    # # suffix of files
    suffix_model = f'{suffix_param}-s{seed}'
    args.logger.log_string(f'======== [Case {icase+1}/{total_cases}]: {split}/{suffix_model} seed{args.seed} ========')

    # # record the predicted frames
    seq_infos_dict = {'use_crf':[], 'not_crf':[]}
    
    #%% ==== Three-fold cross-validation ====
    for ifold in range(num_folds):
        args.logger.log_string('------- [{} / Fold {}] -------'.format(split, ifold))
        
        #%% == loading trajectory-sequence, fit a `scalar` ==
        seq_train, seq_test = loading.load_data(args.root_data, split, ifold, args)
        
        scalar = scalars.StandardScalarDataFrame(cols=cols)
        
        #%% == Model initialization ==
        
        mname = 'c{}-r{}-crf'.format(args.num_conv, args.n_rnn)
        model_name = '({})-{}_{}'.format(mname, suffix_model, ifold)
        model_save_path = str(SCRIPT_DIR / 'well-trained-model' / split) + os.sep
        
        # -- set random seed --
        initial.set_random_state(args.seed)
        
        # -- model initialization --
        model = classifiers.FullModel(
            C=args.C, M=args.M, 
            num_conv=args.num_conv, dim_conv=args.dim_conv, kernel_size=args.kernel_size, 
            dim_out_cnn=args.dim_out_cnn, pdrop_cnn=args.pdrop_cnn,
            dim_hidden=args.dim_hidden, pdrop_agg=args.pdrop_agg,
            n_rnn=args.n_rnn, pdrop_rnn=args.pdrop_rnn, 
        )
        
        if ifold == 0:
            args.logger.log_string(model)
            desc_num_para = 'Number of parameters:\n' + '\n'.join(['{}: {}'.format(n, sum(c.numel() for c in child.parameters())) for n,child in model.named_children()])
            args.logger.log_string(desc_num_para, end='\n\n')
        
        #%% == Training ==
        # -- initialize trainer --
        optimizer = optimization.get_optimizer(model, args)
        for _ in range(3): torch.cuda.empty_cache()
        
        # -- model training --
        trainer = trainers.Trainer(
            args, 
            model, 
            optimizer, 
            scalar = scalar,
            start_epoch = 0, 
            device = device, 
            verbose = args.verbose, 
            n_iter_print = args.n_iter_print, 
        )

        # -- fit data --
        trainer.fit(seq_train)
        model = trainer.get_last_model()
        scalar = trainer.scalar
        
        # -- save model --
        args.logger.log_string('CRF transition:\n{}'.format(model.crf.transitions))
        initial.create_path(model_save_path)
        torch.save(model.state_dict(), model_save_path + '{}_weights.pth'.format(model_name))

        #%% == Evaluating model ==
        
        for key in seq_infos_dict.keys():
            # -- kwargs --
            kwargs = test_dict[key]
            # -- evaluation pipeline --
            pipe = eval_pipeline.EvaluationPipeline(
                args, scalar, model, validate_with=f'{split}-{key}'
            )
            seq_infos = pipe.process(seq_test, **kwargs)
            seq_infos_dict[key].append(seq_infos)
            
            
        
    #%% ==== Evaluating three-fold results ====
    seq_infos_all = {key:pd.concat(seq_infos_list) for key, seq_infos_list in seq_infos_dict.items()}
    
    correctness_all = {}
    transfer_all = {}
    tripleg_all = {}
    mode_chain_all = {}
    
    for key, seq_infos in seq_infos_all.items():
        ident_dir = str(SCRIPT_DIR / 'identified_trajectories' / split) + os.sep
        initial.create_path(ident_dir)
        seq_infos.to_parquet(ident_dir + f'seq_infos-model=({mname})-{suffix_model}-{key}.parquet', index=True)

        # evaluate identification quality
        correctness, transfer, tripleg, mode_chain = metrics.evaluate_from_sequence(seq_infos)
        args.logger.log_string('#'*100)
        args.logger.log_string('**OVERALL** Validate with {}-{}-{}'.format(mname, suffix_model, key))
        visual.print_evaluation(correctness, transfer, tripleg, mode_chain, logger=args.logger)
        args.logger.log_string('#'*100)
        
        correctness_all[key] = correctness
        transfer_all[key] = transfer
        tripleg_all[key] = tripleg
        mode_chain_all[key] = mode_chain

    # == save results ==
    rec_path = str(SCRIPT_DIR / 'result records (auto-save)' / split) + os.sep
    initial.create_path(rec_path)
    visual.save_evaluations(
        path=rec_path+'({})-{}.csv'.format(mname, suffix_model), 
        correctness=correctness_all, transfer=transfer_all, tripleg=tripleg_all, mode_chain=mode_chain_all
    )
        
    #%%
    args.logger.close()

