#@author - Ankur

import pandas as pd
import numpy as np
import torch
import os
import datetime
from dask import dataframe as dd 
from arguments import get_args

class data_read():
    def __init__(self,dir ,dataframe_list  ,date_list , target_list ,idx, multiclass_discontinuous , text, remove_list):
        self.dir = dir ; self.date_list = date_list ; self.target_list = target_list ;self.idx = idx
        self.multiclass_discontinuous = multiclass_discontinuous ; self.text = text ;self.remove_list = remove_list
        self.df_list = self._init_read_(dataframe_list) 
        self.df_list = self._init_dtype_(self.df_list) 
        self._init_number_dtype_split_(self.df_list)
        self._init_impute_vars_(self.df_list)
        self._init_number_dtype_split_(self.df_list)
        self.df_list[0] = self.df_list[0].sort_values(['drop_area','dt'] , ascending = True)
        torch.save(self.df_list[0] , os.path.join(self.dir,'data.pt')) ; torch.save(self.df_list[1] , os.path.join(self.dir,'areas.pt'))
        self.data = torch.load(os.path.join(self.dir,'data.pt')) ; self.areas = torch.load(os.path.join(self.dir,'areas.pt'))
        self.dtype_list = [self.date_list,self.strings,self.number,self.number_binary, self.number_multi ,self.number_continuous, self.multiclass_discontinuous ,self.text ,self.target_list]
        self.dtype_list_name = ['date_list', 'strings', 'number','number_binary', 'number_multi' ,'number_continuous','multiclass_discontinuous' ,'text' ,'target_list']
        self._init_data_check_(self.dtype_list , self.dtype_list_name)


    def _init_read_(self,dataframes):
        df_list = []
        for df in dataframes:
            data = dd.read_csv(os.path.join(self.dir ,df) , na_values = ' ', low_memory=False , assume_missing = True) ; data = data.compute()
            print(data.columns , data.head(5) ,'the shape is here' , data.shape) 
            df_list.append(data)
        return df_list

    def _init_dtype_(self,dataframes):
        df_list = []
        strings = [] ; number = []
        for df in dataframes:
            strs = list(df.select_dtypes(include = [np.object])) ; strings += strs
            nbr = list(df.select_dtypes(exclude = [np.object])) ; number += nbr
            for key in df.keys():
                if key in self.date_list:
                    df[key] = pd.to_datetime(df[key])
                    df[key + '_date'] = df[key].dt.date
                    df[key + '_year'] = df[key].dt.year
                    df[key + '_month'] = df[key].dt.month
                    df[key + '_week'] = df[key].dt.week
                    df[key + '_day'] = df[key].dt.day
                    df[key + '_weekday'] = df[key].dt.weekday
                    df = df.sort_values([key],ascending=True)
                    number += [key + '_year' , key + '_month', key + '_week', key + '_day', key + '_weekday' ]
            df_list.append(df)
            self.strings = list(set(np.unique(strings)) - set((self.date_list + self.idx + self.multiclass_discontinuous + self.text + self.target_list +self.remove_list)))
            self.number = list(set(np.unique(number)) - set((self.idx + self.remove_list + self.multiclass_discontinuous +self.target_list)))
        return df_list

    def _init_number_dtype_split_(self,dataframes):
        number_binary = [] ; number_multi = [] ; number_continuous = []
        for df in dataframes:
            for key in df.keys():
                if (key in self.number and len(np.unique(df[key])) <=2):
                    number_binary += [key]
                if (key in self.number and (len(np.unique(df[key])) > 2 and len(np.unique(df[key])) <= 12)):
                    number_multi += [key]
                if (key in self.number and len(np.unique(df[key])) > 12 ):
                    number_continuous += [key]
            self.number_binary = np.unique(number_binary) ; self.number_multi = np.unique(number_multi) ;self.number_continuous = np.unique(number_continuous)

    def _init_impute_vars_(self,dataframes):
        for df in dataframes:
            for key in df.keys():
                if key in self.strings:
                    df[key].fillna('Impute',inplace = True)
                if key in self.number_binary:
                    df[key].fillna(0,inplace = True)
                if key in self.number_multi:
                    df[key].fillna(0,inplace = True)
                if key in self.number_continuous:
                    df[key].fillna(0 , inplace = True)

    def _init_data_check_(self , dtype_list , dtype_list_name):
        for i in range(0,len(self.dtype_list)):
            print('Data types' , i , self.dtype_list_name[i], self.dtype_list[i])