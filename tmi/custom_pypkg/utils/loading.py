# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 20:55:16 2022

@author: Jackie
"""
import pandas as pd 

# --training and testing data loading--
def load_data(root_data, split, ifold, args):
    # reading data
    traj_seq = pd.read_parquet(root_data + 'trajseq.parquet')
    
    # train-test-split
    assert split in ['trajsplit','usersplit','randsplit']
    loc_path = root_data + 'train_test_split/{}/'.format(split)
    if split == 'trajsplit':
        file_test = 'tid_test_{}.csv'.format(ifold)
        tid_test = pd.read_csv(loc_path + file_test).values.flatten()
        mask_test = traj_seq.tid.isin(tid_test)
    elif split == 'usersplit':
        file_test = 'uid_test_{}.csv'.format(ifold)
        uid_test = pd.read_csv(loc_path + file_test).values.flatten()
        mask_test = traj_seq.uid.isin(uid_test)
    else:
        file_test = 'tid_test_{}.csv'.format(ifold)
        tid_test = pd.read_csv(loc_path + file_test).values.flatten()
        mask_test = traj_seq.tid.isin(tid_test)
    
    seq_train = traj_seq[~mask_test]
    seq_test = traj_seq[mask_test]
    
    return seq_train, seq_test
