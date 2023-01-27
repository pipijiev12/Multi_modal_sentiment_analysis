# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from layers.complexnn import *
from .SimpleNet import SimpleNet


class MLPSubNet(nn.Module):
    '''
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(MLPSubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        #        x = torch.mean(x, dim = 1,keepdim = False)
        #        normed = self.norm(x)
        #        dropped = self.drop(normed)
        dropped = self.drop(x)
        y_1 = torch.relu(self.linear_1(dropped))
        y_2 = torch.relu(self.linear_2(y_1))
        y_3 = torch.relu(self.linear_3(y_2))

        return y_3


class LSTMSubNet(nn.Module):
    '''
    The LSTM-based subnetwork that is used in TFN for text
    '''

    def __init__(self, in_size, hidden_size, out_size, dropout=0.2, device=torch.device('cpu')):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(LSTMSubNet, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(in_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        batch_size = x.shape[0]
        time_stamps = x.shape[1]
        all_h = []
        all_c = []
        _h = torch.zeros(batch_size, self.hidden_size).to(self.device)
        _c = torch.zeros(batch_size, self.hidden_size).to(self.device)

        for t in range(time_stamps):
            _h, _c = self.lstm(x[:, t, :], (_h, _c))
            all_h.append(_h)
            all_c.append(_c)

        h = self.dropout(torch.stack(all_h, dim=1))
        y_1 = self.linear_1(h)
        return y_1

class uQDNN_ATTENTION(torch.nn.Module):
    def __init__(self, opt):
        """
        max_sequence_len: input sentence length
        embedding_dim: input dimension
        num_measurements: number of measurement units, also the output dimension

        """
        super(uQDNN_ATTENTION, self).__init__()
        self.max_sequence_len = opt.max_seq_len
        self.input_dims = opt.input_dims
        self.text_hidden_dim = opt.text_hidden_dim
        if type(opt.subnet_dropout_rates) == float:
            self.subnet_dropout_rates = [opt.subnet_dropout_rates]
        else:
            self.subnet_dropout_rates = [float(s) for s in opt.subnet_dropout_rates.split(',')]
        self.feature_indexes = opt.feature_indexes
        self.device = opt.device

        self.num_measurements = opt.measurement_size
        self.output_cell_dim = opt.output_cell_dim
        self.output_dropout_rate = opt.output_dropout_rate
        self.output_dim = opt.output_dim
        if type(opt.contracted_dims) == int:
            self.contracted_dims = [opt.contracted_dims]
        else:
            self.contracted_dims = [int(s) for s in opt.contracted_dims.split(',')]

        self.modality_weights = nn.Parameter(torch.zeros(len(self.input_dims)))

        # Only textual input, QDNN performed for word pivots
        # Amplitude and Phase embedding addressed

        self.num_modalities = len(self.input_dims)
        self.modality_specific_weights = nn.Parameter(torch.FloatTensor(self.num_modalities))
        self.embedding_matrix = torch.tensor(opt.lookup_table, dtype=torch.float)
        self.embed = nn.Embedding.from_pretrained(self.embedding_matrix, freeze=opt.amplitude_freeze)

        # 1. Temporal convolutional layers
        self.proj_layers = nn.ModuleList([])
        for i in range(len(self.input_dims)):
            if i == 0:
                self.proj_layers.append(LSTMSubNet(self.input_dims[i], self.text_hidden_dim, self.contracted_dims[i],
                                                   dropout=self.subnet_dropout_rates[i], device=self.device))
            else:
                self.proj_layers.append(
                    MLPSubNet(self.input_dims[i], self.contracted_dims[i], self.subnet_dropout_rates[i]))

        self.phase_embed = nn.ModuleList([PhaseEmbedding(opt.lookup_table.shape[0], self.contracted_dims[0],
                                                         sentiment_dic=None, freeze=opt.phase_freeze)])
        for contracted_dim in self.contracted_dims[1:]:
            self.phase_embed.append(PhaseEmbedding(opt.lookup_table.shape[0], contracted_dim, freeze=opt.phase_freeze))

        self.weight_embed = nn.ModuleList(
            [WeightEmbedding(opt.lookup_table.shape[0], freeze=opt.weight_freeze) for i in range(len(self.input_dims))])

        self.embedding_dim = self.input_dims[-1]
        self.l2_norm = L2Norm(dim=-1, keep_dims=True)
        self.l2_normalization = L2Normalization(dim=-1)
        self.activation = nn.Softmax(dim=1)
        self.complex_multiply = ComplexMultiply()
        self.mixture = ComplexMixture(device=self.device)

        self.measurement_dim = 1
        for dim in self.contracted_dims:
            self.measurement_dim = self.measurement_dim * dim

        self.measurement = ComplexMeasurement2(self.measurement_dim, units=self.num_measurements, device=self.device)

        if self.output_dim == 1:
            self.fc_out = nn.Sequential(nn.Dropout(self.output_dropout_rate),
                                        nn.Linear(self.num_measurements, self.output_cell_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.output_cell_dim, self.output_cell_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.output_cell_dim, self.output_dim))
        else:
            self.fc_out = nn.Sequential(nn.Dropout(self.output_dropout_rate),
                                        nn.Linear(self.num_measurements, self.output_cell_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.output_cell_dim, self.output_cell_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.output_cell_dim, self.output_dim))

        self.real_softmax = nn.Softmax(dim=2)
        self.imag_softmax = nn.Softmax(dim=2)


    def forward(self,in_modalities):


        batch_size = in_modalities[0].shape[0]
        seq_len = in_modalities[0].shape[1]

        word_indexes = in_modalities[0]

        weights = []

        in_modalities = [self.embed(modality) if len(modality.shape) == 2 else modality for modality in in_modalities]

        hidden_units_real = []
        hidden_units_imag = []

        for i in range(len(self.input_dims)):
            phases = self.phase_embed[i](word_indexes)
            amplitudes = self.proj_layers[i](in_modalities[i])
            norms = self.l2_norm(amplitudes)
            amplitudes = self.l2_normalization(amplitudes)
            weights.append(self.activation(norms))

            [seq_embedding_real,seq_embedding_imag] = self.complex_multiply([phases,amplitudes])
            hidden_units_real.append(seq_embedding_real)
            hidden_units_imag.append(seq_embedding_imag)

        modality_weights = nn.Softmax(dim=-1)(self.modality_weights)
        weight = torch.zeros_like(weights[0])

        for w,modality_weights in zip(weights,modality_weights):
            weight = weight+modality_weights*w

        real_tensors = []
        imag_tensors = []


        for i in range(seq_len):
            tensor_product_real = torch.ones(batch_size,1).to(self.device)
            tensor_product_imag = torch.ones(batch_size,1).to(self.device)

            for h_real,h_imag in zip(hidden_units_real,hidden_units_imag):
                h_added_real = h_real[:,i,:]
                h_added_imag = h_imag[:,i,:]

                result_real = torch.bmm(tensor_product_real.unsqueeze(2), h_added_real.unsqueeze(1)) - torch.bmm(
                    tensor_product_imag.unsqueeze(2), h_added_imag.unsqueeze(1))
                result_imag = torch.bmm(tensor_product_real.unsqueeze(2), h_added_imag.unsqueeze(1)) + torch.bmm(
                    tensor_product_imag.unsqueeze(2), h_added_real.unsqueeze(1))
                tensor_product_real = result_real.view(batch_size,-1)
                tensor_product_imag = result_imag.view(batch_size,-1)

            real_tensors.append(tensor_product_real)
            imag_tensors.append(tensor_product_imag)

        real_tensors = torch.stack(real_tensors,dim=1)
        imag_tensors = torch.stack(imag_tensors,dim=1)

        # 添加quantum self-Attention机制
        Q_tensors_real = real_tensors
        Q_tensors_imag = imag_tensors

        K_tensors_real = real_tensors
        K_tensors_imag = imag_tensors

        V_tensors_real = real_tensors
        V_tensors_imag = imag_tensors

        U_tensors_real = torch.bmm(Q_tensors_real,K_tensors_real.transpose(1,2)) - torch.bmm(Q_tensors_imag,K_tensors_imag.transpose(1,2))
        U_tensors_imag = torch.bmm(Q_tensors_imag,K_tensors_real.transpose(1,2)) + torch.bmm(Q_tensors_real,K_tensors_imag.transpose(1,2))

        # # 缩放点积
        # U_tensors_real = U_tensors_real / math.sqrt(np.shape(K_tensors_real)[2])
        # U_tensors_imag = U_tensors_imag / math.sqrt(np.shape(K_tensors_imag)[2])

        # 点积
        U_tensors_real = U_tensors_real
        U_tensors_imag = U_tensors_imag



        atten_real = self.real_softmax(U_tensors_real)
        atten_imag = self.imag_softmax(U_tensors_imag)

        atten_real_tensors = torch.bmm(atten_real,V_tensors_real) - torch.bmm(atten_imag,V_tensors_imag)
        atten_imag_tensors = torch.bmm(atten_imag,V_tensors_real) + torch.bmm(atten_real,V_tensors_imag)

        atten_real_tensors += Q_tensors_real
        atten_imag_tensors += Q_tensors_imag

        output = self.measurement([atten_real_tensors,atten_imag_tensors,weight])
        output = self.fc_out(output)

        return output

