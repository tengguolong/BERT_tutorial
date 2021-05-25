import torch
from torch.utils.data import Dataset
from typing import List
import json
import pandas as pd
import torch
import random
from tqdm import tqdm
import pdb


class BERTDataset(Dataset):
    def __init__(self, corpus_path, word2idx_path, seq_len):
        # define path of dicts
        self.word2idx_path = word2idx_path
        # define max length
        self.seq_len = seq_len
        # directory of corpus dataset
        self.corpus_path = corpus_path
        # define special symbols
        self.pad_index = 0
        self.unk_index = 1
        self.cls_index = 2
        self.sep_index = 3
        self.mask_index = 4

        self.mask_p = 0.10

        # 加载字典
        with open(word2idx_path, "r", encoding="utf-8") as f:
            self.word2idx = json.load(f)

        # 加载语料
        self.df = pd.read_csv(self.corpus_path, sep='\t')
        self.docs = [self.tokenize_char(text) for text in tqdm(self.df['text'], desc='map word to token id...')]
        self.docs = [doc for doc in self.docs if len(doc) > 50]
        self.corpus_lines = len(self.docs)

        print("*********************************")
        print("dataset size:", self.corpus_lines)
        print("seq len:", self.seq_len)
        print("mask prob:", self.mask_p)


    def __len__(self):
        return self.corpus_lines


    def __getitem__(self, item):
        tokens = self.docs[item]
        length = len(tokens)
        start = random.randrange(length - self.seq_len) if length > self.seq_len else 0
        end = start + self.seq_len - 1
        tokens = tokens[start: end]
        tokens, label = self.random_char(tokens)


        bert_input = [self.cls_index] + tokens + [self.sep_index]
        bert_label = [self.pad_index] + label + [self.pad_index]
        attn_mask = [1] * len(bert_input)

        if len(bert_input) > self.seq_len:
            bert_input = bert_input[:self.seq_len]
            bert_label = bert_label[:self.seq_len]
            attn_mask = attn_mask[:self.seq_len]
        elif len(bert_input) < self.seq_len:
            num_pad = self.seq_len - len(bert_input)
            bert_input += [0] * num_pad
            bert_label += [0] * num_pad
            attn_mask += [0] * num_pad

        return torch.LongTensor(bert_input), torch.LongTensor(bert_label), torch.tensor(attn_mask)


    def tokenize_char(self, text):
        return [self.word2idx.get(x, self.unk_index) for x in text.split(' ')]


    def random_char(self, char_tokens: List):
        output_label = [0] * len(char_tokens)
        
        token_len = int(min(self.seq_len, len(char_tokens)))
        num_mask_tokens = int(max(1, self.mask_p * token_len))
        mask_pos = random.sample(range(token_len), num_mask_tokens)
        for i in mask_pos:
            output_label[i] = char_tokens[i]
            prob = random.random()
            # 80% randomly change token to mask token
            if prob < 0.8:
                char_tokens[i] = self.mask_index
            # 10% randomly change token to random token
            elif prob < 0.9:
                char_tokens[i] = random.randrange(len(self.word2idx))
            # 10% unchange

        return char_tokens, output_label