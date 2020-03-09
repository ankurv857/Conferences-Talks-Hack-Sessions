#@author - Ankur

import sys
import pandas as pd
import torch
import numpy as np
import copy
from bisect import bisect
from datetime import datetime, timedelta
from os import listdir
from os.path import join, isdir
from collections import namedtuple
from torch.utils.data import Dataset
import random
import glob, os
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from numpy import split
from numpy import array
from scipy import optimize
from scipy.stats import norm
import math

class data_load(Dataset):

    def __init__(self, data, areas,seq_len=14, cv=None, train=True, target='quantity'):
        self.seq_len = seq_len
        self.seq_delta = timedelta(days=seq_len)
        self.cv = cv
        self.train = train
        self.target = target
        self.to_embed = ['drop_area' ,'dt_week',  'dt_day']
        self.to_skip = ['dt_date']
        self.dynamic=['data']
        self.static=['areas']
        self._init_emb_dict_([data,areas])
        self.data , self.areas = self._init_emb_insert_([data,areas])
        self._init_data_sample_() 
        self._init_count_dim_() 
        #print('check the dimensions for all 4 types' , 'self.num_seq_int' , self.num_seq_int , 'self.num_seq_real' , self.num_seq_real ,'self.num_stat_int ' , self.num_stat_int  , 'self.num_stat_real' , self.num_stat_real)

    
    #Create the embedding dictionaries for the embed features
    def _init_emb_dict_(self,dataframes):
        emb_dict = {} ; emb_len = {}
        for df in dataframes:
            for key in df.keys():
                if key in np.unique(self.to_embed):
                    emb_dict[key ] = np.unique(df[key])
                    emb_len[key ] = len(np.unique(df[key]))
        self.emb_dict = emb_dict ; self.emb_len = emb_len 

    #Insert the dictionary mapping into respective dataframes
    def _init_emb_insert_(self, dataframes):
        df_list = [] ;  self.emb_int = []
        for df in dataframes:
            for key in self.emb_dict.keys():
                if key in df.keys():
                    df[key] = df[key].replace(list(self.emb_dict.get(key)) , list(range(len(self.emb_dict.get(key)))))
                    self.emb_int += [key]
            df_list.append(df)
        return df_list[0] , df_list[1]
        
    #create sample index from data with sequence length
    def _init_data_sample_(self):
        """Construct the mapping of sampleable data's to index in data df.
        Each sample loads temporal data between (day - seq_len) and  (day + seq_len), so it's important that we
        ensure no overlap in days observed between the train and test set.
        """

        data_index = self.data.index
        # Check that data indices are complete and consecutive
        assert sorted(data_index) == list(range(min(data_index), max(data_index) + 1))

        min_day = (self.data['dt'].min()) + self.seq_delta
        max_day = self.data['dt'].max()
        print('min day and max day' , min_day , max_day ,  max_day - min_day)

        #Sample the data for min and max day removing the sequence length from head and tail for proper warmup
        samples = copy.deepcopy(self.data)
        samples['idx'] = samples.index
        samples = samples.loc[(samples['dt'] >= min_day) & (samples['dt'] <= max_day)]
        #print('num samples',len(samples) , 'total data' , len(self.data) , 'Sampled %:', len(samples)/len(self.data))

        if self.cv is not None:
            cv_start = self.cv['start'] 
            cv_end = self.cv['end']
        else:
            # Defaults to a test set 2*seq_delta long
            cv_start = max_day - 2*self.seq_delta
            cv_end = max_day

        assert cv_start <= cv_end
        assert cv_start >= min_day
        assert cv_end <= max_day
        print('cv_start',cv_start)
        print('cv_end',cv_end)

        if self.train:
            # Each sample will be predicted seq_len out from sample day, so ensure there's no overlap between
            # train predictions and test warmup (test day - seq_len).
            #print('in train')
            samples = samples.loc[(samples['dt'] < cv_start)| (samples['dt'] > cv_end)] ; print('in train' , samples.shape)
        else:
            samples = samples.loc[(samples['dt'] >= cv_start) & (samples['dt'] <= cv_end)] ;  print('in test' , samples.shape)

        samples = samples.sort_values(['drop_area','dt'] , ascending = True)
        self.data_samples = samples['idx'].tolist()

        self.data_index = self.data.index
        assert len(self.data_samples) <= len(self.data_index)

    def _init_sample_index_(self, data_index):
        """Returns the other data vectors given the data index."""
        data = self.data.iloc[data_index]
        areas = self.areas[self.areas.drop_area.isin(np.unique(data.drop_area))]
        return data , areas

    def _init_count_dim_(self):
        self.num_seq_real, self.num_seq_int, self.num_stat_real, self.num_stat_int  = 0, 0, 0, 0
        # Dynamic features
        for data_id in self.dynamic:
            data = getattr(self,data_id)
            for key in data.keys():
                if key not in ['drop_area', 'dt'] + self.to_skip:
                    if key in self.to_embed:
                        self.num_seq_int += 1
                    if key not in self.to_embed + [self.target]:
                        self.num_seq_real += 1
        # Static features
        for data_id in self.static:
            data = getattr(self,data_id)
            for key in data.keys():
                if key not in self.to_skip:
                    if key in self.to_embed:
                        self.num_stat_int += 1
                    if key not in self.to_embed + [self.target]:
                        self.num_stat_real += 1

    def _preprocess(self):
        pass

    def __len__(self):
        #print('enter length' , len(self.data_samples))
        return len(self.data_samples)

    def __getitem__(self , j):
        #print('entered getitem')

        """get an index j as input coming from len(self.data_samples) use it to get the other data (areas) indices
        returns 4 data structures of sizes:
        1) 2xseq_len,self.num_seq_real
        2) 2xseq_len,self.num_seq_int
        3) self.num_stat_real
        4) self.num_stat_int
        """
        #print('Entered the __getitem__(self,j)')
        data_idx = self.data_samples[j]  #; print('dataidx', data_idx)  # Returns index of data row in self.data that can be constructed into a sample
        data , areas = self._init_sample_index_(data_idx) 

        data_seq = self.data.iloc[data_idx - self.seq_len:data_idx + self.seq_len] #; print('data_seq'  , data_seq['dt'].min() , data_seq['dt'].max())

        #Ensure that the sequence is for a single time series
        assert data_seq['drop_area'].nunique() == 1


        seq_real = np.zeros((2 * self.seq_len, self.num_seq_real  ), float)
        seq_int = np.zeros((2 * self.seq_len, self.num_seq_int ), int)
        y_type = self.data.dtypes[self.target]
        targets = np.zeros(2 * self.seq_len, y_type)
        real_id, int_id = 0, 0

        seq_int_dict = {}
        seq_real_dict = {}
        stat_int_dict = {}
        stat_real_dict = {}

        ### Dynamic Features ###

        # loops through data
        #print('data_seq.keys()' , data_seq.keys())
        for key in data_seq.keys():
            if key not in ['drop_area', 'dt'] + self.to_skip:
                if key == self.target:
                    targets[:] = data_seq[key]
                if key in self.to_embed:
                    seq_int_dict[int_id] = key
                    seq_int[:, int_id] = data_seq[key]
                    int_id += 1
                if key not in self.to_embed + [self.target]:
                    seq_real_dict[real_id] = key
                    seq_real[:, real_id] = data_seq[key]
                    real_id += 1

        ### Static Features ###

        stat_real = np.zeros(self.num_stat_real, float)
        stat_int = np.zeros(self.num_stat_int, int)
        real_id, int_id = 0, 0

        # loops through areas
        areas_seq = areas
        for key in areas_seq.keys():
            if key not in self.to_skip:
                if key in self.to_embed:
                    stat_int_dict[int_id] = key
                    stat_int[int_id] = areas_seq[key]
                    int_id += 1
                if key not in self.to_embed + [self.target]:
                    stat_real_dict[real_id] = key
                    stat_real[real_id] = areas_seq[key]
                    real_id += 1

        if hasattr(self , 'seq_int_dict') is False:
            self.seq_int_dict =  seq_int_dict
            self.seq_real_dict = seq_real_dict
            self.stat_int_dict = stat_int_dict
            self.stat_real_dict = stat_real_dict
        assert self.seq_int_dict == seq_int_dict
        assert self.seq_real_dict == seq_real_dict
        assert self.stat_int_dict == stat_int_dict
        assert self.stat_real_dict == stat_real_dict

        #print('dicts' , self.seq_real_dict,self.seq_int_dict,self.stat_real_dict,self.stat_int_dict)
        #print('tensor_shape' , seq_real.shape , seq_int.shape , stat_real.shape ,stat_int.shape ,  targets.shape)
        #tensor_shape (28, 3) (28, 2) (1,) (1,) (28,)
        #print('tensors' , seq_real , seq_int , stat_real ,stat_int ,  targets)
        return torch.tensor(seq_real, dtype=torch.float32), torch.LongTensor(seq_int), torch.tensor(stat_real, dtype=torch.float32), torch.LongTensor(stat_int), torch.tensor(targets, dtype=torch.float32)
    
 
 