# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn
from layers.squeeze_embedding import SqueezeEmbedding
from layers.dynamic_rnn import DynamicLSTM

INFINITY_NUMBER = 1e12

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.nonlinearity = nn.Sigmoid()
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class BERT_SPC(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_SPC, self).__init__()
        # self.squeeze_embedding = SqueezeEmbedding()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim+2*opt.hidden_dim, opt.polarities_dim)
        # self.dense = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)

        self.text_lstm = DynamicLSTM(opt.bert_dim, opt.hidden_dim, num_layers=1, batch_first=True,
                                     bidirectional=True)

        self.gc1 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)

        self.gate1 = nn.Sequential(nn.Linear(opt.hidden_dim*2, opt.hidden_dim*2), nn.Sigmoid(),
                                   nn.Linear(opt.hidden_dim * 2, opt.hidden_dim * 2), nn.Sigmoid(),
                                   nn.Linear(opt.hidden_dim * 2, opt.hidden_dim * 2), nn.Sigmoid())
        self.gate2 = nn.Sequential(nn.Linear(opt.hidden_dim*2, opt.hidden_dim*2), nn.Sigmoid(),
                                   nn.Linear(opt.hidden_dim * 2, opt.hidden_dim * 2), nn.Sigmoid(),
                                   nn.Linear(opt.hidden_dim * 2, opt.hidden_dim * 2), nn.Sigmoid())

        self.fc = nn.Linear(2 * 2 * opt.hidden_dim, opt.polarities_dim)
        self.fc2 = nn.Linear(2 * 2 * opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, adj, aspect_mask, mask, dist_to_target = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]
        text_bert_len = torch.sum(text_bert_indices != 0, dim=-1)
        max_len = max(text_bert_len.data.cpu().numpy().tolist())
        aspect_mask = aspect_mask[:,:max_len]
        mask = mask[:,:max_len]
        dist_to_target = dist_to_target[:,:max_len]
        # text_bert_indices = self.squeeze_embedding(text_bert_indices, text_bert_len)
        # bert_segments_ids = self.squeeze_embedding(bert_segments_ids, text_bert_len)
        x, pooled_output = self.bert(text_bert_indices, bert_segments_ids, output_all_encoded_layers=False)
        x, (_,_) = self.text_lstm(x, text_bert_len)
        aspect = torch.max(x.masked_fill(aspect_mask.unsqueeze(2), -INFINITY_NUMBER), 1)[0]
        gate1 = self.gate1(aspect).repeat(1, x.shape[1]).view(x.shape)
        gate2 = self.gate2(aspect).repeat(1, x.shape[1]).view(x.shape)
        gcn1 = self.gc1(x, adj[:,:max_len, :max_len])
        gcngate1 = gcn1 * gate1
        y = gcn1 * gate2
        x1 = torch.max(gcngate1.masked_fill(mask.unsqueeze(2), -INFINITY_NUMBER), 1)[0]
        y1 = torch.max(y.masked_fill(mask.unsqueeze(2), -INFINITY_NUMBER), 1)[0]
        sf1 = nn.Softmax(1)
        xy = (x1*y1).sum(1).mean()
        x = gate2 * self.gc2(gcn1, adj[:, :max_len, :max_len])
        # x = self.gc2(gcn1, adj[:, :max_len, :max_len])
        out = torch.max(x, dim=1)[0]
        pooled_output = self.dropout(pooled_output)
        out = self.dropout(out)
        logits = self.dense(torch.cat([pooled_output,out],dim=1))

        output_w = self.fc(torch.cat([x, aspect.repeat(1, x.shape[1]).view(x.shape)], dim=2))
        scores = (logits.repeat(1,x.shape[1]).view(x.shape[0], x.shape[1], -1) * output_w).sum(2)
        sf2 = nn.Softmax(2)
        kl = (sf1(scores) * sf1(dist_to_target.float())).sum(1).mean()

        return logits, xy, kl
        # return logits, xy, 0
        return logits, 0, 0
