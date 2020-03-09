#@author - Ankur

import argparse
import numpy as np
import os
from os.path import join, exists
import sys
import time
import torch
import torch.utils.data
from torch.utils.data import Dataset , DataLoader
from torch.nn import functional as F
from torchvision import transforms
import torch.nn as nn
import pandas as pd
from config import config
from reader import data_read
from loader import data_load
from model import RNN_MLP
from arguments import get_args

def masked_loss(pred_g, g, warm_up=14):
    """should take prediction of size 2xseq_len-1,1 and  g of size 2xseq_len,1
        and compute the mean squared error note that we are only doing next_step predictions"""
    #extracts the predictions of warm_up+1 till end and flattens them, note that predictions are done at the previous step
    #so :,warm_up,: stores the predicted value for warm_up+1
    y_hat = pred_g[:, warm_up-1:-1].contiguous().view(-1)
    #print( 'y_hat',y_hat , y_hat.size() , pred_g.size())
    #extracts the ground truth targets from warm_up+1 till the end and flattens them
    y = g[:, warm_up:].contiguous().view(-1)
    #print( 'y', y ,  y.size(), g.size())
    #extracts the ground truth  from warm_up till the end-1 and flattens them
    y_1 = g[:, warm_up-1:-1].contiguous().view(-1)
    #print( 'y_1', y_1 , y_1.size())
    #computes loss for predictions from network
    loss = F.mse_loss(y_hat, y, reduction='mean')
    #computes loss for y_t+1=y_t
    loss_vanilla = F.mse_loss(y_hat, y, reduction='mean')
    return loss, loss_vanilla

def rnn_epoch(epoch, train):
    if train:
        #sets the network in train mode so parameters accepts gradient, if there is batch_norm the batch_mean/std are used
        #if there is dropout a mask is sampled
        net.train()
        loader = train_loader
    else:
        #sets the network in test mode parameters do not have gradients, batch norm uses running moments
        # dropout outputs are rescaled since they are not sampled
        net.eval()
        loader = test_loader

    # Cycle through the data loader for num_batches equal to __len of the dataset
    cum_loss, cum_loss_van = 0, 0
    e_start = time.time()
    for i, data in enumerate(loader):
        #extract batch and gets source and targets for predictions
        #t_real is expected to be a torch float tensor batch_size,seq_lenx2,num_t_real_features
        #t_int is expected to be a torch long tensor batch_size,seq_lenx2,num_t_int_features
        #s_real is expected to be a torch float tensor batch_size,num_s_real_features
        #s_int is expected to be a torch long tensor batch_size,num_s_int_features
        start = time.time()
        t_real, t_int, s_real, s_int, g = [arr for arr in data]

        #forward pass
        optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr, alpha=.9)
        optimizer.zero_grad() #removes gradients from the optimizer, default is to accumulate
        pred_g = net(t_real, t_int, s_real, s_int) #forward pass, it outputs a 2xseq_len predictions
        #loss computation for real network and pred_g=g_(t-1)
        loss, loss_vanilla = masked_loss(pred_g, g, warm_up=args.seq_len)
        #backward
        if train:
            loss.backward() #loss is backpropagated
            optimizer.step() #parameters updates is applied
        #accumulate losses
        cum_loss += loss.item()
        cum_loss_van += loss_vanilla.item()

        print('cum_loss' , cum_loss , 'cum_loss_van' , cum_loss_van)

        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d] LossRNN: %.6f, LossVAN: %.6f, Time: %.6f'
                % (epoch, args.epochs, i+1,
                    (len(loader.dataset) // args.batch_size) + 1,
                    cum_loss/(i + 1),
                    cum_loss_van/(i + 1),
                    time.time() - start))
        sys.stdout.flush()
            

if __name__ == '__main__':
    print('Yo! LSTM Models for Multi Time series')
    args = get_args()

    data = data_read(config["directory"], config['dataframe_list'] ,config['date_list'] , config['target_list'] ,
    config['idx'], config['multiclass_discontinuous'],config['text'] , config['remove_list'])
    
    print("Here goes the Neural Network data loader" , data.df_list[0].head(2) , data.df_list[1].head(2))
    dataset_train = data_load(data.data , data.areas , cv=None , train=True) ; print('done with dataset train' , dataset_train)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size= args.batch_size , shuffle=True, num_workers=48) ; print('done with train loader')
    dataset_test = data_load(data.data , data.areas , cv=None , train=False) ; print('done with dataset test' , dataset_train)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size= args.batch_size , shuffle=True, num_workers=16) ; print('done with test loader')

    _ = dataset_train[0] ; _ = dataset_test[0] 

# ################# EMBEDDINGS ################

    # Dynamic embeddings
    in_seq_list = []   # each item is num of unique input values for embedding
    out_seq_list = []  # each item is num of output values for embedding

    # Static embeddings
    in_stat_list = []  # each item is num of unique input values for embedding
    out_stat_list = []  # each item is num of output values for embedding


    # FIX: seq_int_dict - only contains [0, 1] instead of [0, 1,...,10]
    for key in dataset_train.seq_int_dict.keys():
        in_seq_list.append(len(dataset_train.emb_dict[dataset_train.seq_int_dict[key]]))
    out_seq_list = [args.embs_size] * len(in_seq_list)

    for key in dataset_train.stat_int_dict.keys():
        in_stat_list.append(len(dataset_train.emb_dict[dataset_train.stat_int_dict[key]]))
    out_stat_list = [args.embs_size] * len(in_stat_list)

    in_seq_real = dataset_train.num_seq_real  # num of dynamic non-embedded features
    in_stat_real = dataset_train.num_stat_real  # num of static non-embedded features

    print('inputs to net', in_seq_list, out_seq_list, in_stat_list, out_stat_list, in_seq_real, in_stat_real, args.in_rnn, args.out_rnn, args.out_mlp)

# #################################

    # create the network the network currently lacks a method to pass in inputs but it will need to know:
    # sizes of input and outputs for each of the embeddings,
    # size of outputs and number of layers for encoders combining real and int features for rnn and mlp
    # size of output for LSTM
    # number of layers for decoder predicting the target

    net = RNN_MLP(in_seq_list, out_seq_list, in_stat_list, out_stat_list, in_seq_real, in_stat_real, args.in_rnn, args.out_rnn, args.out_mlp)

    for e in range(args.epochs):
        #for each epoch runs one loop on train and one validation
        rnn_epoch(e, train=True)

