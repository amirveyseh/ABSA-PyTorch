# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn
from layers.squeeze_embedding import SqueezeEmbedding

# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM

INFINITY_NUMBER = 1e12
EPSILON = 1e-12

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
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

class ASGCN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ASGCN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(opt.embed_dim+768, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gcn_lstm = DynamicLSTM(2*opt.hidden_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*2*opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)
        self.gc1 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)

        self.gate1 = nn.Sequential(nn.Linear(opt.hidden_dim*2, opt.hidden_dim*2), nn.Sigmoid(),
                                   nn.Linear(opt.hidden_dim * 2, opt.hidden_dim * 2), nn.Sigmoid(),
                                   nn.Linear(opt.hidden_dim * 2, opt.hidden_dim * 2), nn.Sigmoid())
        self.gate2 = nn.Sequential(nn.Linear(opt.hidden_dim*2, opt.hidden_dim*2), nn.Sigmoid(),
                                   nn.Linear(opt.hidden_dim * 2, opt.hidden_dim * 2), nn.Sigmoid(),
                                   nn.Linear(opt.hidden_dim * 2, opt.hidden_dim * 2), nn.Sigmoid())

    def get_mask(self, x, aspect_double_idx, reverse=False):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i,0]):
                if reverse:
                    mask[i].append(1)
                else:
                    mask[i].append(0)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                if reverse:
                    mask[i].append(0)
                else:
                    mask[i].append(1)
            for j in range(aspect_double_idx[i,1]+1, seq_len):
                if reverse:
                    mask[i].append(1)
                else:
                    mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.opt.device)
        return mask.byte()

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj, dist_to_target, bert = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        text_mask = (text_indices == 0).unsqueeze(2)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        text = self.embed(text_indices)
        text = torch.cat([text, bert], dim=2)



        text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, text_len)

        mask = self.get_mask(text_out, aspect_double_idx, reverse=True)
        aspect = torch.max(text_out.masked_fill(mask, -INFINITY_NUMBER), 1)[0]
        gate1 = self.gate1(aspect).repeat(1, text_out.shape[1]).view(text_out.shape)
        gate2 = self.gate2(aspect).repeat(1, text_out.shape[1]).view(text_out.shape)
        gcn1 = self.gc1(text_out, adj)
        x = gcn1 * gate1
        y = gcn1 * gate2
        x1 = torch.max(x.masked_fill(text_mask.byte(), -INFINITY_NUMBER), 1)[0]
        y1 = torch.max(y.masked_fill(text_mask.byte(), -INFINITY_NUMBER), 1)[0]
        sf1 = nn.Softmax(1)
        xy = (x1*y1).sum(1).mean()
        gcn2 = self.gc2(x, adj)
        x = gcn2 * gate2

        mask = self.get_mask(x, aspect_double_idx)
        sent = torch.max(x.masked_fill(mask, -INFINITY_NUMBER), 1)[0]
        mask = self.get_mask(x, aspect_double_idx, reverse=True)
        aspect = torch.max(x.masked_fill(mask, -INFINITY_NUMBER), 1)[0]
        output = self.fc(torch.cat([aspect, sent], dim=1))

        output_w = self.fc(torch.cat([x, aspect.repeat(1, x.shape[1]).view(x.shape)], dim=2))
        scores = (output.repeat(1,text_indices.shape[1]).view(text_indices.shape[0], text_indices.shape[1], -1) * output_w).sum(2)
        sf2 = nn.Softmax(2)
        # output_p = sf2(output.repeat(1,text_indices.shape[1]).view(text_indices.shape[0], text_indices.shape[1], -1))
        # output_w_p = sf2(output_w)
        # scores = (output_p * torch.log(output_p/output_w_p+EPSILON)).sum(2)
        # scores_p = sf1(scores)
        # dist_to_target_p = sf1(dist_to_target.float())
        # kl = (dist_to_target_p * torch.log(dist_to_target_p / scores_p + EPSILON)).sum(1).mean()

        kl = (sf1(scores) * sf1(dist_to_target.float())).sum(1).mean()

        return output, xy, kl

class BERT_SPC(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_SPC, self).__init__()
        # self.squeeze_embedding = SqueezeEmbedding()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, adj, untok_tok_mapping, length = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]

        print(len(text_bert_indices))
        print(len(bert_segments_ids))
        print(len(untok_tok_mapping))
        print(len(adj))
        print(len(length))
        exit(1)

        # text_bert_len = torch.sum(text_bert_indices != 0, dim=-1)
        # text_bert_indices = self.squeeze_embedding(text_bert_indices, text_bert_len)
        # bert_segments_ids = self.squeeze_embedding(bert_segments_ids, text_bert_len)
        x, pooled_output = self.bert(text_bert_indices, bert_segments_ids, output_all_encoded_layers=False)

        bert_emb = []
        for i in range(x.shape[0]):
            y = []
            for j in range(length[i]):
                try:
                    y.append(torch.mean(x[i, untok_tok_mapping[i][j][0]:untok_tok_mapping[i][j][-1] + 1, :],
                            dim=0))
                except:
                    print(len(untok_tok_mapping), i)
                    print(len(untok_tok_mapping[i]), j)
                    print(len(untok_tok_mapping[i][j]))
                    exit(1)
            y = torch.stack(y, dim=0)
            # y = torch.stack(
            # [torch.mean(x[i, untok_tok_mapping[i][j][0]:untok_tok_mapping[i][j][-1] + 1, :],
            #             dim=0)
            # for j in range(length[i])], dim=0)
            bert_emb.append(y)
        bert_emb = torch.stack(bert_emb, dim=0)
        print(bert_emb.shape)
        print(adj.shape)
        exit(1)


        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits
