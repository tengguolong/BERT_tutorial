
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pandas as pd
import tqdm
import numpy as np
import pdb

from dataset.news_dataset import MultiSentenceDataset as CLSDataset
from models.bert_base import BertModel, BertConfig
from models.document_classifier import DocClassifier
from utils.lr_scheduler import WarmupCosineLR, WarmupMultiStepLR

from train_news import config, bert_config


class Trainer:
    def __init__(self,
                 train_epoches=3,
                 start_epoch=0,
                 load_pretrain=False
                 ):
        self.vocab_size = int(config["vocab_size"])
        self.start_epoch = start_epoch
        self.train_epoches = train_epoches

        bertconfig = BertConfig(**bert_config)
        self.bert_model = DocClassifier(config=bertconfig)

        self.bert_model = self.bert_model.cuda()
        # self.bert_model = torch.nn.DataParallel(self.bert_model)

        test_dataset = CLSDataset(corpus_path=config["test_corpus_path"],
                                  vocab_path=config['vocab_path'],
                                  max_seq_len=config['max_seq_len'])
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=config['batch_size'],
                                          shuffle=False,
                                          num_workers=0)


    def load_model(self, model, dir_path="../output", load_bert=False):
        checkpoint_dir = self.find_most_recent_state_dict(dir_path)
        checkpoint = torch.load(checkpoint_dir)
        res = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(res)
        print("{} loaded!".format(checkpoint_dir))

    def test(self, epoch=0):
        self.bert_model.eval()
        with torch.no_grad():
            return self.iteration(epoch, self.test_dataloader, train=False)


    def iteration(self, epoch, data_loader, train=False, df_name="df_log.pickle"):
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="testing",
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        
        all_predictions = []

        for i, data in data_iter:
            token_id, attn_mask, _ = data
            token_id = token_id.cuda()
            attn_mask = attn_mask.cuda()

            logits = self.bert_model(text_input=token_id, attn_mask=attn_mask)

            predictions = torch.argmax(logits, dim=-1).detach().cpu().numpy().reshape(-1).tolist()
            all_predictions.extend(predictions)
        
        return all_predictions


    def find_most_recent_state_dict(self, dir_path):
        dic_lis = [i for i in os.listdir(dir_path)]
        if len(dic_lis) == 0:
            raise FileNotFoundError("can not find any state dict in {}!".format(dir_path))
        dic_lis = [i for i in dic_lis if "model" in i]
        dic_lis = sorted(dic_lis, key=lambda k: int(k.split(".")[-1]))
        return dir_path + "/" + dic_lis[-1]



if __name__ == '__main__':
    trainer = Trainer()
    trainer.load_model(trainer.bert_model, dir_path=config["output_path"])
    predictions = trainer.test()
    print(len(predictions))
    submission = pd.DataFrame({'label': predictions})
    submission.to_csv('save/submission.csv', index=False)
