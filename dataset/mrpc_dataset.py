from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd

from transformers import BertTokenizer, BertTokenizerFast


class MRPCDataset(Dataset):
    def __init__(self, corpus_path, max_seq_len=128):

        self.max_seq_len = max_seq_len
        self.corpus_path = corpus_path

        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.cls_token = 101
        self.sep_token = 102
        
        self.df = pd.read_csv(self.corpus_path, sep=',')
        self.corpus_lines = len(self.df)
        self.has_label = 'label' in self.df
        print("dataset size:", self.corpus_lines, 'has_label:', self.has_label)
        print("max seq len:", self.max_seq_len)

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        row = self.df.iloc[item]

        text1 = row.sentence1.strip()
        text2 = row.sentence2.strip()
        text_input1 = self.tokenizer(text1)["input_ids"]
        text_input2 = self.tokenizer(text2)["input_ids"][1:] # remove [CLS] token

        text_input = text_input1 + text_input2
        type_id = [0] * len(text_input1) + [1] * len(text_input2)
        attn_mask = [1] * len(text_input)
        if len(text_input) > self.max_seq_len:
            text_input = text_input[:self.max_seq_len]
            type_id = type_id[:self.max_seq_len]
            attn_mask = attn_mask[:self.max_seq_len]
        elif len(text_input) < self.max_seq_len:
            num_pad = self.max_seq_len - len(text_input)
            text_input += [0] * num_pad
            type_id += [0] * num_pad
            attn_mask += [0] * num_pad
        
        label = torch.LongTensor([row.label]) if self.has_label else None

        return torch.tensor(text_input), torch.LongTensor(type_id), torch.tensor(attn_mask), label

