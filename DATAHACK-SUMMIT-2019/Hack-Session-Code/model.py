#@author - Ankur

from collections import OrderedDict, defaultdict
import os
from os.path import join
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

def Perceptron(in_dim, out_dim, nlayers, mid_dim=None, disc_input=False):
    # with 0 layers it is either embedding or a embedding layer
    if nlayers == 0:
        if disc_input is False:
            return nn.Linear(in_dim,out_dim)
        else:
            return nn.Embedding(in_dim + 1, out_dim, padding_idx=in_dim) #the embedding layer is create with an extra (padded) dimension
    else:
        if mid_dim is None: #if mid_dim is not specified it uses two times the in_dim size
            mid_dim = in_dim * 2
        if disc_input is False: #creates the first layer as linear or embedding with output size mid_dim
            layers = [('linear1', nn.Linear(in_dim, mid_dim))]
        else:
            layers = [('embedding1', nn.Embedding(in_dim + 1, mid_dim, padding_idx=in_dim))]
        for i in range(nlayers - 1): #left-over layers are mid_dim --> mid_dim with a ReLU
            layers.append(('relu{}'.format(i + 1), nn.ReLU()))
            layers.append(('linear{}'.format(i + 2), nn.Linear(mid_dim, mid_dim)))
        layers.append(('relu{}'.format(nlayers), nn.ReLU()))
        #last layer with output of size out_dim
        layers.append(('linear{}'.format(nlayers + 1), nn.Linear(mid_dim, out_dim)))
        return nn.Sequential(OrderedDict(layers))


class MultiEmbedding(nn.Module):
    def __init__(self, in_sizes, out_sizes, sequenced=False):
        super().__init__()
        self.sequenced = sequenced
        self.emb =  nn.ModuleList([])
        #loops through the list of input/output sizes and create a new embedding layer of the respective size
        for in_s, out_s in zip(in_sizes, out_sizes):
            self.emb.append(nn.Embedding(in_s, out_s))
            #print('self.emb',self.emb) 

    def forward(self, x):
        if self.sequenced:
            assert len(x.size()) == 3 #if the size of the embedding is 3
        else:
            assert len(x.size()) == 2 #if the size of the embedding is 2

        emb_list=[]
        # Loops through the feature dimmension of x and applies the ith embedding to the ith feature
        for i in range(x.size()[-1]):
            if len(x.size()) == 2:
                emb_list.append(self.emb[i](x[:, i]))
                #print('emb_list_stat',np.shape(emb_list))
            elif len(x.size()) == 3:
                emb_list.append(self.emb[i](x[:, :, i]))
                #print('emb_list_seq',np.shape(emb_list))
        #print('lenght of the tensor' , len(x.size())) 
        return torch.cat(emb_list, dim=len(x.size()) - 1)


class RNN_MLP(nn.Module):
    print('enter RNN MLP')
    """module containing 1 rnn 1 mlp they respective encoders with int embedding layers and a decoder"""
    def __init__(self, in_seq_list, out_seq_list, in_stat_list, out_stat_list, in_seq_real, in_stat_real, in_rnn_dim, out_rnn_dim, out_mlp_dim):
        super().__init__()

        # Initialize the embedding layers
        self.embed_rnn = MultiEmbedding(in_seq_list, out_seq_list, sequenced=True)  # Temporal embedding
        self.embed_mlp = MultiEmbedding(in_stat_list, out_stat_list, sequenced=False)  # Static embedding

        # Initialize the temporal and static encoders
        #print('IS 107?', in_seq_real, sum(out_seq_list))
        self.encoder_rnn = Perceptron(in_seq_real + sum(out_seq_list), in_rnn_dim, 0) #with 0 layers it just a linear layer
        self.encoder_mlp = Perceptron(in_stat_real + sum(out_stat_list), out_mlp_dim, 2) #2 hidden layers for our mlp

        #talk aboput the details

        self.rnn = nn.LSTM(in_rnn_dim, out_rnn_dim, 1, batch_first=True) #fused lstm, if we have problems we can switch to the cell version.
        self.decoder = Perceptron(out_mlp_dim + out_rnn_dim, 1, 0) #linear layer as decoder that takes output of mlp and rnn concatenated as inputs

    def forward(self, t_real, t_int, s_real, s_int):
        #get embedding for discrete variables for both rnn and mlp
        #print('t and s sizes ' , t_real.size(), t_int.size()) #t and s sizes  torch.Size([8, 28, 3]) torch.Size([8, 28, 2])
        emb_rnn = self.embed_rnn(t_int)
        emb_mlp = self.embed_mlp(s_int)
        #print('embrnn and embmlp sizes ' , emb_rnn.size(), emb_mlp.size()) #embrnn and embmlp sizes  torch.Size([8, 28, 64]) torch.Size([8, 32])
        #combines the real features and the embedding into a encoder
        rnn_in = torch.cat((t_real, emb_rnn), dim=2)
        rnn_in_size = rnn_in.size()
        #print('rnn_in_size' , rnn_in_size) #rnn_in_size torch.Size([8, 28, 67])
        # flattens rnn_in from batch,seq,in_size to batch*seq,in_size pass it to encoder and reshape it back to batch*seq,out_size
        #print('rnn_in.view(-1, rnn_in_size[2])' , rnn_in.view(-1, rnn_in_size[2]).size()) #torch.Size([bs*28, 67])
        #print('self.encoder_rnn(rnn_in.view(-1, rnn_in_size[2]))' , self.encoder_rnn(rnn_in.view(-1, rnn_in_size[2])).size()) #torch.Size([bs*28, 16])
        rnn_enc = self.encoder_rnn(rnn_in.view(-1, rnn_in_size[2])).view(rnn_in_size[0], rnn_in_size[1], -1)
        #print('rnn_enc' , rnn_enc.size()) #torch.Size([2, 28, 16])
        #encodes the mlp floats together with outputs of mlp embs
        #print('torch.cat((s_real,emb_mlp), dim=1)' , torch.cat((s_real,emb_mlp), dim=1).size())
        mlp_enc = self.encoder_mlp(torch.cat((s_real,emb_mlp), dim=1))
        #print('mlp_enc' , mlp_enc.size()) #torch.Size([2, 16])

        # pass the input sequence to the rnn (currently using fused_rnn)
        #care that rnn_out might be (seq_len, batch, instead of batch, seq_len which is how the batcher is natively packing our data
        rnn_out, c_n = self.rnn(rnn_enc)
        #print('rnn_out' , rnn_out, ' c_n',c_n)

        #get the sizes for rnn and mlp
        rnn_sizes = rnn_out.size()
        #print('rnn_sizes', rnn_sizes)
        mlp_sizes = mlp_enc.size() #bs,out_mlp
        #print('mlp_sizes' , mlp_sizes)
        #broadcasts mlp_out from batch_size,mlp_out_size to batch_size,seq_len,mlp_out_size and then flattens it to batch_size*seq_len,mlp_out_size
        mlp_out = mlp_enc.unsqueeze(1).expand(mlp_sizes[0], rnn_sizes[1], mlp_sizes[1])
        mlp_out_sizes = mlp_out.size() #bs,out_mlp
        #print('mlp_out_sizes' , mlp_out ,mlp_out_sizes) #torch.Size([2, 28, 16])

        mlp_out=mlp_out.contiguous().view(-1, mlp_out_sizes[-1])
        mlp_out_sizes = mlp_out.size() #bs,out_mlp
        #print('mlp_out_sizesv2' , mlp_out_sizes) #([56, 16])

        #flattens rnn_out to batch_size*seq_len,rnn_out_size
        rnn_out = rnn_out.contiguous().view(rnn_sizes[0] * rnn_sizes[1], rnn_sizes[2])
        #print('rnn_out_v2' , rnn_out.size()) #([56, 16])
        #cats rnn_out with mlp_out to batch_size*seq_len,rnn_out_size+,mlp_out_size
        # pass it through decoder to  batch_size*seq_len,1 , reshape to  batch_size,seq_len,1
        preds = self.decoder(torch.cat((rnn_out, mlp_out), dim=1)).view(rnn_sizes[0], rnn_sizes[1])
        #print('prediction' , preds , preds.size())
        return preds