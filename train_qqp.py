import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import math
from sklearn import metrics
from metrics import *
import tqdm
import pandas as pd
import numpy as np
import configparser
import pdb

from dataset.qqp_dataset import QQPDataset
from models.bert_base import BertModel, BertConfig
from utils.lr_scheduler import WarmupCosineLR, WarmupMultiStepLR

config = {}
config["pretrained"] = '/glusterfs/data/transformers/pretrained/bert-base-uncased/pytorch_model.bin'
config["output_path"] = "save/mrpc"
config["batch_size"] = 32
config["grad_accum_steps"] = 1
config["max_seq_len"] = 128
config["lr"] = 3e-5
config["scheduler"] = "step"
config["weight_decay"] = 1e-2
config["num_workers"] = 0
config["log_interval"] = 50
config["val_interval"] = 1


bert_config = dict(vocab_size=30522, # 字典字数
                   hidden_size=768, # 隐藏层维度也就是字向量维度
                   num_hidden_layers=12, # transformer block 的个数
                   num_attention_heads=12, # 注意力机制"头"的个数
                   intermediate_size=768*4, # feedforward层线性映射的维度
                   hidden_act="gelu", # 激活函数
                   hidden_dropout_prob=0.1, # dropout的概率
                   attention_probs_dropout_prob=0.1,
                   max_position_embeddings=512,
                   type_vocab_size=2, # 用来做next sentence预测, 这里预留了256个分类, 其实我们目前用到的只有0和1
                   initializer_range=0.02 # 用来初始化模型参数的标准差
)


_skip_keys = []
def preprocess_bert_weights(ckpt):
    keys = list(ckpt.keys())
    for key in keys:
        if "LayerNorm" in key:
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            elif "beta" in key:
                new_key = key.replace("beta", "bias")
            ckpt[new_key] = ckpt.pop(key)
    for key in _skip_keys:
        ckpt.pop(key)
    return ckpt


class Bert_Pair_Classification(nn.Module):
    def __init__(self, config):
        super(Bert_Pair_Classification, self).__init__()
        self.bert = BertModel(config, add_pooling_layer=True)
        self.num_classes = 2

        # self.dropout = nn.Dropout(0.1)
        # self.second_last_dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self._init_extra_weights(self.second_last_dense)

        self.final_dense = nn.Linear(config.hidden_size, self.num_classes)
        self._init_extra_weights(self.final_dense)
    

    def _init_extra_weights(self, module):
        """Initialize the weights"""
        name = module.__class__.__name__
        module.weight.data.normal_(mean=0.0, std=0.001)
        if module.bias is not None:
            module.bias.data.zero_()


    def compute_loss(self, predictions, labels):
        predictions = predictions.view(-1, self.num_classes)
        labels = labels.view(-1)
        loss = F.cross_entropy(predictions, labels)
        return loss

    def forward(self, text_input, type_id, attn_mask, labels=None):
        outputs = self.bert(text_input, token_type_ids=type_id, attention_mask=attn_mask)
        token_embeddings, pooled_output = outputs
        if pooled_output is None:
            pooled_output = token_embeddings[:, 0]
            pooled_output = self.second_last_dense(pooled_output)

        # token_embeddings: [bs, L, H]
        # pooled_out: [bs, H]

        # pooled_output = self.dropout(pooled_output)
        predictions = self.final_dense(pooled_output)

        if labels is not None:
            # 计算loss
            loss = self.compute_loss(predictions, labels)
            return torch.argmax(predictions, dim=-1), loss
        else:
            return torch.argmax(predictions, dim=-1)


class Trainer:
    def __init__(self, max_seq_len,
                 batch_size,
                 lr, # 学习率
                 train_epoches=3,
                 start_epoch=0,
                 with_cuda=True, # 是否使用GPU, 如未找到GPU, 则自动切换CPU
                 load_pretrain=True
                 ):
        config_ = configparser.ConfigParser()
        config_.read("./config/qqp.ini")
        self.config = config_["DEFAULT"]
        self.vocab_size = int(self.config["vocab_size"])
        self.batch_size = batch_size
        self.lr = lr
        self.start_epoch = start_epoch
        self.train_epoches = train_epoches
        self.max_seq_len = max_seq_len


        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        bertconfig = BertConfig(vocab_size=self.vocab_size)
        self.bert_model = Bert_Pair_Classification(config=bertconfig)

        if load_pretrain:
            ckpt_file = config["pretrained"]
            ckpt = torch.load(ckpt_file)
            ckpt = preprocess_bert_weights(ckpt)
            info = self.bert_model.load_state_dict(ckpt, strict=False)
            print("******missing:", info[0])
            print("******unexpected:", info[1])

        self.bert_model = self.bert_model.cuda()

        train_dataset = QQPDataset(corpus_path=self.config["train_corpus_path"],
                                    max_seq_len=self.max_seq_len)
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           num_workers=0
                                           )

        test_dataset = QQPDataset(corpus_path=self.config["test_corpus_path"],
                                    max_seq_len=self.max_seq_len)
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=self.batch_size,
                                          shuffle=False,
                                          num_workers=0)

        # optim_parameters = list(self.bert_model.parameters())
        # self.optimizer = torch.optim.Adam(optim_parameters, lr=config["lr"], weight_decay=config["weight_decay"])

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.bert_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config["weight_decay"],
            },
            {
                "params": [p for n, p in self.bert_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=config["lr"])

        # learning rate schedule
        num_iter_per_epoch = math.ceil(len(train_dataset) / batch_size / config["grad_accum_steps"])
        max_iters = train_epoches * num_iter_per_epoch
        print("max iters:", max_iters)
        warmup_iters = int(max_iters*0.1)
        print("warmup iters:", warmup_iters)
        if config["scheduler"] == "cosine":
            self.scheduler = WarmupCosineLR(self.optimizer,
                                            max_iters=max_iters,
                                            warmup_iters=warmup_iters,
                                            warmup_factor=0.01)
        else:
            milestones = [int(max_iters*0.9)]
            self.scheduler = WarmupMultiStepLR(self.optimizer,
                                               milestones=milestones,
                                               gamma=0.1,
                                               warmup_factor=0.01,
                                               warmup_iters=warmup_iters)
        iters_passed = int(num_iter_per_epoch * start_epoch)
        for _ in range(iters_passed):
            self.scheduler.step()

        print("Total Parameters:", sum([p.nelement() for p in self.bert_model.parameters()]))

    def run(self):
        for epoch in range(self.start_epoch, self.train_epoches):
            self._train(epoch)
            self.save_state_dict(trainer.bert_model, epoch, config["output_path"], "bert.model")
            self._test(epoch)

    def load_model(self, model, model_dir="../output", load_bert=False):
        checkpoint_dir = self.find_most_recent_state_dict(model_dir)
        checkpoint = torch.load(checkpoint_dir)

        if load_bert:
            checkpoint["model_state_dict"] = {k[5:]: v for k, v in checkpoint["model_state_dict"].items()
                                              if k[:4] == "bert" and "pooler" not in k}
        res = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(res)
        torch.cuda.empty_cache()
        model = model.cuda()
        print("{} loaded!".format(checkpoint_dir))

    def _train(self, epoch):
        self.bert_model.train()
        self.iteration(epoch, self.train_dataloader, train=True)

    def _test(self, epoch):
        self.bert_model.eval()
        with torch.no_grad():
            return self.iteration(epoch, self.test_dataloader, train=False)

    def iteration(self, epoch, data_loader, train=True, df_name="df_log.pickle"):
        # 进度条显示
        str_code = "train" if train else "test"
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        total_loss = 0
        all_predictions, all_labels = [], []
        grad_step = 0

        for i, data in data_iter:
            token_id, type_id, attn_mask, labels = data
            token_id = token_id.cuda()
            type_id = type_id.cuda()
            attn_mask = attn_mask.cuda()
            labels = labels.cuda()

            assert token_id.size(1) <= self.max_seq_len, "input length: {}".format(token_id.size(1))
            predictions, loss = self.bert_model.forward(text_input=token_id,
                                                        type_id=type_id,
                                                        attn_mask=attn_mask,
                                                        labels=labels
                                                        )
            loss = loss / config["grad_accum_steps"]

            predictions = predictions.detach().cpu().numpy().reshape(-1).tolist()
            labels = labels.cpu().numpy().reshape(-1).tolist()
            all_predictions.extend(predictions)
            all_labels.extend(labels)

            # 计算fscore
            fscore = metrics.f1_score(all_labels, all_predictions, average='macro')
            equ = np.array(all_predictions) == np.array(all_labels)
            acc = np.sum(equ) / len(equ)

            if train:
                loss.backward()
                grad_step += 1
                
                if grad_step == config["grad_accum_steps"]:
                    self.optimizer.step()
                    self.scheduler.step()

                    self.optimizer.zero_grad()
                    grad_step = 0

            # 为计算当前epoch的平均loss
            total_loss += loss.item() * config["grad_accum_steps"]

            if train:
                current_lr = self.optimizer.param_groups[0]['lr']
                log_dic = {
                    "epoch": epoch,
                    "lr": current_lr,
                    "train_loss": total_loss/(i+1),
                    "train_score": fscore,
                    "train_acc": acc,
                    "test_loss": 0,
                    "test_score": 0,
                    "test_acc": 0
                }

            else:
                log_dic = {
                    "epoch": epoch,
                    "train_loss": 0,
                    "train_score": 0,
                    "train_acc": 0,
                    "test_loss": total_loss/(i+1),
                    "test_score": fscore,
                    "test_acc": acc
                }

            if (i+1) % config["log_interval"] == 0:
                data_iter.write(str({k: v for k, v in log_dic.items() if v != 0}))

        data_iter.write(str({k: v for k, v in log_dic.items() if v != 0}))

        return fscore

    def find_most_recent_state_dict(self, dir_path):
        """
        :param dir_path: 存储所有模型文件的目录
        :return: 返回最新的模型文件路径, 按模型名称最后一位数进行排序
        """
        dic_lis = [i for i in os.listdir(dir_path)]
        if len(dic_lis) == 0:
            raise FileNotFoundError("can not find any state dict in {}!".format(dir_path))
        dic_lis = [i for i in dic_lis if "model" in i]
        dic_lis = sorted(dic_lis, key=lambda k: int(k.split(".")[-1]))
        return dir_path + "/" + dic_lis[-1]

    def save_state_dict(self, model, epoch, state_dict_dir="../output", file_path="bert.model"):
        """存储当前模型参数"""
        if not os.path.exists(state_dict_dir):
            os.mkdir(state_dict_dir)
        save_path = state_dict_dir + "/" + file_path + ".epoch.{}".format(str(epoch))
        torch.save({"model_state_dict": model.state_dict()}, save_path)
        print("{} saved!".format(save_path))


if __name__ == '__main__':
    start_epoch = 0
    train_epoches = 3
    load_model = False

    trainer = Trainer(  max_seq_len=config["max_seq_len"],
                        batch_size=config["batch_size"],
                        lr=config["lr"],
                        start_epoch=start_epoch,
                        train_epoches=train_epoches,
                        with_cuda=True)
    if load_model:
        trainer.load_model(trainer.bert_model, model_dir=config["output_path"])

    trainer.run()

    

