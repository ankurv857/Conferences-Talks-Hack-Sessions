#@author - Ankur

import os
from arguments import get_args

args = get_args()
config = {}

config['directory'] = args.data_dir
config['dataframe_list'] = ['data_jan_july.csv' , 'areas.csv']
config['date_list'] = ['dt']
config['target_list'] = ['quantity']
config['idx'] = []
config['multiclass_discontinuous'] = []
config['text'] = []
config['remove_list'] = []