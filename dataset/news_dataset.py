from torch.utils.data import Dataset
import torch

import random
import os
import pickle
from tqdm import tqdm
import json
import pdb
import pandas as pd


class CLSDataset(Dataset):
    def __init__(self, corpus_path, vocab_path, max_seq_len=512):

        self.max_seq_len = max_seq_len
        self.corpus_path = corpus_path

        with open(vocab_path, 'r') as f:
            words = [x.strip() for x in f.readlines()]
        self.word2idx = {w: i for i, w in enumerate(words)}
        print("size of vocab:", len(self.word2idx))
        

        # define special symbols
        self.pad_index = self.word2idx['[PAD]']
        self.unk_index = self.word2idx['[UNK]']
        self.cls_index = self.word2idx['[CLS]']
        self.sep_index = self.word2idx['[SEP]']
        self.mask_index = self.word2idx['[MASK]']

        self.df = pd.read_csv(self.corpus_path, sep='\t')
        self.corpus_lines = len(self.df)
        self.has_label = self.df.shape[1] == 2
        print("dataset size:", self.corpus_lines, 'has_label:', self.has_label)

    def __len__(self):
        return self.corpus_lines
        # return 500

    def __getitem__(self, item):
        row = self.df.iloc[item]
        text_input = self.tokenize_char(row.text)
        text_input = [self.cls_index] + text_input + [self.sep_index]
        attn_mask = [1] * len(text_input)

        text_input = text_input[:self.max_seq_len]
        attn_mask = attn_mask[:self.max_seq_len]

        num_padding = self.max_seq_len - len(text_input)
        if num_padding > 0:
            text_input += [self.pad_index] * num_padding
            attn_mask += [0] * num_padding

        label = torch.LongTensor([row.label]) if self.has_label else None

        return torch.LongTensor(text_input), torch.tensor(attn_mask), label
    
    def tokenize_char(self, text):
        return [self.word2idx.get(x, self.unk_index) for x in text.split(' ')]



class MultiSentenceDataset(Dataset):
    def __init__(self, corpus_path, vocab_path, max_seq_len=256, max_doc_len=16):

        self.max_seq_len = max_seq_len
        self.max_doc_len = max_doc_len
        self.corpus_path = corpus_path

        with open(vocab_path, 'r') as f:
            words = [x.strip() for x in f.readlines()]
        self.word2idx = {w: i for i, w in enumerate(words)}
        print("size of vocab:", len(self.word2idx))

        # define special symbols
        self.pad_index = self.word2idx['[PAD]']
        self.unk_index = self.word2idx['[UNK]']
        self.cls_index = self.word2idx['[CLS]']
        self.sep_index = self.word2idx['[SEP]']
        self.mask_index = self.word2idx['[MASK]']

        self.df = pd.read_csv(self.corpus_path, sep='\t')
        cache_file = self.corpus_path + '.pkl'
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.docs = pickle.load(f)
        else:
            self.docs = [self.tokenize_char(text) for text in tqdm(self.df['text'], desc='Word to token_id')]
            with open(cache_file, 'wb') as f:
                pickle.dump(self.docs, f)
        self.corpus_lines = len(self.docs)
        
        self.has_label = self.df.shape[1] == 2
        print("dataset size:", self.corpus_lines, 'has_label:', self.has_label)
        if self.has_label:
            self.labels = list(self.df['label'])


    def __len__(self):
        return self.corpus_lines
        # return 500

    def __getitem__(self, item):
        tokens = self.docs[item]
        length = len(tokens)

        text_inputs, attn_masks = [], []
        for s in range(self.max_doc_len):
            start = s * (self.max_seq_len - 1)
            end = start + self.max_seq_len - 1
            if start < length:
                text_input = [self.cls_index] + tokens[start: end]
                attn_mask = [1] * len(text_input)
            else:
                text_input = []
                attn_mask = []
            # padding
            num_padding = self.max_seq_len - len(text_input)
            if num_padding > 0:
                text_input += [self.pad_index] * num_padding
                attn_mask += [0] * num_padding
            
            text_inputs.append(text_input)
            attn_masks.append(attn_mask)


        label = torch.LongTensor([self.labels[item]]) if self.has_label else [-1]

        return torch.LongTensor(text_inputs), torch.tensor(attn_masks), label
    
    def tokenize_char(self, text):
        return [self.word2idx.get(x, self.unk_index) for x in text.split(' ')]

