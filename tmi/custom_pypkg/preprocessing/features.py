# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 21:46:25 2022

@author: Jackie
"""

import numpy as np


def sphericalDistance(lng1, lat1, lng2, lat2): 
    '''lng/lat may be int or float, or a Series, ndarray, or array'''
    lat1,lat2,lng1,lng2 = map(np.radians, [lat1,lat2,lng1,lng2])
    # spherical distance AB: {arccos[sin(a)*sin(x)+cos(b)*cos(x)*cos(b-y)]}*R  (a,b,x,y)
    R = 6378.137 # Earth radius, km
    temp = np.sin(lat1)*np.sin(lat2)+\
            np.cos(lat1)*np.cos(lat2)*np.cos(lng2-lng1)  
    try:
        temp[temp>1.0] = 1.0
        d = np.arccos(temp)*R* 1000
        d[np.isnan(d)] = 0
        return np.around(d,3)
    except:
        temp = min(temp, 1.0)
        d = np.arccos(temp)*R* 1000
        return round(d, 3)
    
    
def calDistFrom(data, by):
    lng_nxt = data.groupby(by).lng.shift(-1)
    lat_nxt = data.groupby(by).lat.shift(-1)
    dl = sphericalDistance(data.lng, data.lat, lng_nxt, lat_nxt)
    return dl


def calTimeFrom(data, by):
    dt = data.groupby(by).timestamp.shift(-1) - data.timestamp
    return dt


def calDistxyFrom(data, by):
    lng_nxt = data.groupby(by).lng.shift(-1)
    lat_nxt = data.groupby(by).lat.shift(-1)
    
    dx_abs = sphericalDistance(data.lng, data.lat, lng_nxt, data.lat)
    dir_x = (data.lng <= lng_nxt).astype(int) * 2 - 1
    dy_abs = sphericalDistance(data.lng, data.lat, data.lng, lat_nxt)
    dir_y = (data.lat <= lat_nxt).astype(int) * 2 - 1
    
    return dir_x * dx_abs, dir_y * dy_abs


def calTrigonometric(data, by):
    dx, dy = calDistxyFrom(data, by)
    
    data['dx'], data['dy'] = dx, dy
    dx_nxt = data.groupby(by).dx.shift(-1)
    dy_nxt = data.groupby(by).dy.shift(-1)
    data.drop(columns=['dx', 'dy'], inplace=True)
    
    cos = (dx * dx_nxt + dy * dy_nxt) / ((dx**2 + dy**2) * (dx_nxt**2 + dy_nxt**2) + 1e-2)**0.5
    cos = cos.fillna(1)
    sin = (dx * dy_nxt - dx_nxt * dy) / ((dx**2 + dy**2) * (dx_nxt**2 + dy_nxt**2) + 1e-2)**0.5
    sin = sin.fillna(0)
    return cos, sin


def seqence_sorting(seq_df, index, columns):
    def sort_tid_by_npnt(X_df):
        info_df = X_df.iloc[:,[0]].dropna().reset_index()
        npnt_sort = info_df.groupby(['tid']).size().rename('npnt').reset_index().sort_values(['npnt','tid']).set_index(['tid'])
        return npnt_sort.index
    # data sorting by number of tw
    index_sort = sort_tid_by_npnt(seq_df)
    seq_df = (
        seq_df.reset_index().set_index('tid')
        .loc[index_sort].reset_index()
        .set_index(index)[columns]
    )
    # finish
    return seq_df

def compute_xy(trips, args, sort=False):
    
    # compute features
    trips['dl'] = calDistFrom(trips, by='tid')
    trips['dt'] = calTimeFrom(trips, by='tid').fillna(1)
    trips['v'] = (trips.dl / trips.dt).fillna(0)
    dv = (trips.groupby('tid').v.shift(-1) - trips.v).fillna(0)
    trips['a'] = (dv / trips.dt).fillna(0)
    cos, sin = calTrigonometric(trips, by='tid')
    trips['w'] = (np.arccos(cos) / trips.dt).fillna(0)
    
    #
    trips['dtw'] = (trips.groupby('tid').twid.shift(-1) - trips.twid).fillna(1)
    
    #
    is_twid_change = ((trips.tid!=trips.tid.shift(1)) | (trips.twid!=trips.twid.shift(1)))
    trips['itw'] = is_twid_change.cumsum() - 1
    itw_min = trips.groupby(['tid']).itw.min().rename('itw_min')
    trips = trips.merge(itw_min, on=['tid'], how='left')
    trips.loc[:,'itw'] = (trips.itw - trips.itw_min).astype(int) # for index
    trips.drop(columns=['itw_min'], inplace=True)
    
    # calculate X and y
    idx_xy = (args.idx_cols + args.ptord_cols)
    cols_x = (args.tw_cols + args.feat_cols + args.weight_cols)
    cols_y = (args.tw_cols + args.label_cols + args.trans_cols)
    X_df = trips[idx_xy + cols_x].set_index(idx_xy)
    y_df = trips[idx_xy + cols_y].set_index(idx_xy)
    if sort:
        X_df = seqence_sorting(X_df, index=idx_xy, columns=cols_x)
        y_df = seqence_sorting(y_df, index=idx_xy, columns=cols_y)
        
    # finish
    return X_df, y_df

