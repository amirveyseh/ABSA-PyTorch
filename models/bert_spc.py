# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn
from layers.squeeze_embedding import SqueezeEmbedding
from layers.dynamic_rnn import DynamicLSTM


class BERT_SPC(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_SPC, self).__init__()
        # self.squeeze_embedding = SqueezeEmbedding()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        # self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.dense = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)

        self.text_lstm = DynamicLSTM(768, opt.hidden_dim, num_layers=1, batch_first=True,
                                     bidirectional=True)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        text_bert_len = torch.sum(text_bert_indices != 0, dim=-1)
        # text_bert_indices = self.squeeze_embedding(text_bert_indices, text_bert_len)
        # bert_segments_ids = self.squeeze_embedding(bert_segments_ids, text_bert_len)
        x, pooled_output = self.bert(text_bert_indices, bert_segments_ids, output_all_encoded_layers=False)
        x, (_,_) = self.text_lstm(x, text_bert_len)
        pooled_output = torch.max(x, dim=1)[0]
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits
