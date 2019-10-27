# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_pretrained_bert import BertTokenizer
from collections import defaultdict
import random, math

def match_tokenized_to_untokenized(tokenized_sent, untokenized_sent):
    '''Aligns tokenized and untokenized sentence given subwords "##" prefixed
    Assuming that each subword token that does not start a new word is prefixed
    by two hashes, "##", computes an alignment between the un-subword-tokenized
    and subword-tokenized sentences.
    Args:
      tokenized_sent: a list of strings describing a subword-tokenized sentence
      untokenized_sent: a list of strings describing a sentence, no subword tok.
    Returns:
      A dictionary of type {int: list(int)} mapping each untokenized sentence
      index to a list of subword-tokenized sentence indices
    '''
    mapping = defaultdict(list)
    untokenized_sent_index = 0
    tokenized_sent_index = 1
    while (untokenized_sent_index < len(untokenized_sent) and
        tokenized_sent_index < len(tokenized_sent)):
      while (tokenized_sent_index + 1 < len(tokenized_sent) and
          tokenized_sent[tokenized_sent_index + 1].startswith('##')):
        mapping[untokenized_sent_index].append(tokenized_sent_index)
        tokenized_sent_index += 1
      mapping[untokenized_sent_index].append(tokenized_sent_index)
      untokenized_sent_index += 1
      tokenized_sent_index += 1
    return mapping

def build_tokenizer(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def _load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else './glove.42B.300d.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post', raw=False):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        if not raw:
            return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)
        else:
            return sequence


class ABSADataset(object):
    def __init__(self, fname, tokenizer,  batch_size=16, sort_key='text_indices', shuffle=True, sort=True):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        fin = open(fname+'.graph', 'rb')
        idx2gragh = pickle.load(fin)
        fin.close()

        fin = open(fname+'.dist', 'rb')
        idx2dist = pickle.load(fin)
        fin.close()

        all_data = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()

            text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            aspect_len = np.sum(aspect_indices != 0)
            polarity = int(polarity) + 1


            text_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")

            bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (aspect_len + 1))
            bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)

            dependency_graph = idx2gragh[i]
            length = len(dependency_graph)
            bert_input = "[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]"
            tokenized_sent = tokenizer.tokenizer.wordpiece_tokenizer.tokenize(bert_input)
            untok_tok_mapping = match_tokenized_to_untokenized(tokenized_sent, bert_input)

            dist_to_target = idx2dist[i]

            text_indices = [1 for _ in (text_left + " " + aspect + " " + text_right).split(" ")]
            context_indices = [1 for _ in (text_left + " " + text_right).split(" ")]
            aspect_indices = [1 for _ in aspect.split(" ")]
            left_indices = [1 for _ in text_left.split(" ")]

            data = {
                'text_indices': text_indices,
                'context_indices': context_indices,
                'aspect_indices': aspect_indices,
                'left_indices': left_indices,
                'text_bert_indices': text_bert_indices,
                'bert_segments_ids': bert_segments_ids,
                'polarity': polarity,
                'dependency_graph': dependency_graph,
                'untok_tok_mapping': untok_tok_mapping,
                'length': length,
                'distance_to_target': dist_to_target,
            }

            all_data.append(data)
        self.data = all_data

        if shuffle:
            random.shuffle(self.data)

        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.batches = self.sort_and_pad(self.data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batches

    def pad_data(self, batch_data):
        batch_text_bert_indices = []
        batch_bert_segments_ids = []
        batch_polarity = []
        batch_dependency_graph = []
        batch_untok_tok_mapping = []
        batch_dist_to_target = []
        batch_length = []
        batch_text_indices = []
        batch_context_indices = []
        batch_aspect_indices = []
        batch_left_indices = []
        max_len = max([len(t[self.sort_key]) for t in batch_data])
        # max_len = 80
        for item in batch_data:
            text_indices, context_indices, aspect_indices, left_indices, text_bert_indices, bert_segments_ids, polarity, dependency_graph, untok_tok_mapping, length, distance_to_target = \
                item['text_indices'], item['context_indices'], item['aspect_indices'], item['left_indices'], item['text_bert_indices'], item['bert_segments_ids'], item['polarity'], item['dependency_graph'],\
                item['untok_tok_mapping'], item['length'], item['distance_to_target']
            batch_text_bert_indices.append(text_bert_indices)
            batch_bert_segments_ids.append(bert_segments_ids)
            batch_polarity.append(polarity)
            # batch_dependency_graph.append(dependency_graph)
            batch_dependency_graph.append(np.pad(dependency_graph, \
                ((0,max_len-len(dependency_graph)),(0,max_len-len(dependency_graph))), 'constant'))
            batch_untok_tok_mapping.append(untok_tok_mapping)
            batch_length.append(length)
            dist_padding = [0] * (max_len - len(distance_to_target))
            batch_dist_to_target.append(distance_to_target + dist_padding)
            text_padding = [0] * (max_len - len(text_indices))
            context_padding = [0] * (max_len - len(context_indices))
            aspect_padding = [0] * (max_len - len(aspect_indices))
            left_padding = [0] * (max_len - len(left_indices))
            batch_text_indices.append(text_indices + text_padding)
            batch_context_indices.append(context_indices + context_padding)
            batch_aspect_indices.append(aspect_indices + aspect_padding)
            batch_left_indices.append(left_indices + left_padding)

        return { \
                'text_indices': torch.tensor(batch_text_indices), \
                'context_indices': torch.tensor(batch_context_indices), \
                'aspect_indices': torch.tensor(batch_aspect_indices), \
                'left_indices': torch.tensor(batch_left_indices), \
                'text_bert_indices': torch.tensor(batch_text_bert_indices), \
                'bert_segments_ids': torch.tensor(batch_bert_segments_ids), \
                'polarity': torch.tensor(batch_polarity), \
                'dependency_graph': torch.tensor(batch_dependency_graph), \
                'untok_tok_mapping': batch_untok_tok_mapping, \
                'distance_to_target': torch.tensor(batch_dist_to_target), \
                'length': batch_length
            }

    def __getitem__(self, index):
        return self.batches[index]
        # return self.data[index]

    def __len__(self):
        return len(self.data)
