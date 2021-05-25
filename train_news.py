import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import math
from sklearn import metrics
from metrics import *
import tqdm
import numpy as np
import pdb

from dataset.news_dataset import MultiSentenceDataset as CLSDataset
from models.bert_base import BertModel, BertConfig
from models.document_classifier import DocClassifier
from utils.lr_scheduler import WarmupCosineLR, WarmupMultiStepLR

config = dict(
    vocab_path = 'save/rank6/vocab.txt',
    vocab_size = 6982,
    num_workers = 0,
    train_corpus_path = 'corpus/news/train.csv',
    val_corpus_path = 'corpus/news/val.csv',
    test_corpus_path = 'corpus/news/test_a.csv',
)

# config["pretrained"] = 'bert_state_dict/uncased_L-6_H-256_A-4/pytorch_model.bin'
config["pretrained"] = 'save/rank6/pytorch_model.bin'
config["output_path"] = "save/rank6_pretrain_bert_lstm_attention"
config["batch_size"] = 12
config["grad_accum_steps"] = 3
config["max_seq_len"] = 256
config["max_doc_len"] = 16
config["lr"] = 2.5e-5
config["scheduler"] = "cosine"
config["warmup_ratio"] = 0.25
config["weight_decay"] = 1e-2
config["num_workers"] = 0
config["log_interval"] = 200
config["val_interval"] = 1
if not os.path.exists(config['output_path']):
    os.makedirs(config['output_path'])


bert_config = dict(vocab_size=6982, # 字典字数
                   hidden_size=256, # 隐藏层维度也就是字向量维度
                   num_hidden_layers=4, # transformer block 的个数
                   num_attention_heads=4, # 注意力机制"头"的个数
                   intermediate_size=256*4, # feedforward层线性映射的维度
                   hidden_act="gelu", # 激活函数
                   hidden_dropout_prob=0.1, # dropout的概率
                   attention_probs_dropout_prob=0.1,
                   max_position_embeddings=256,
                   type_vocab_size=2, # 用来做next sentence预测, 这里预留了256个分类, 其实我们目前用到的只有0和1
                   initializer_range=0.02 # 用来初始化模型参数的标准差
)


import logging
logging.basicConfig(level=logging.INFO, 
                    filename=os.path.join(config['output_path'], 'log.txt'),
                    filemode='w',
                    format='%(asctime)-15s %(levelname)s: %(message)s')


# _skip_keys = ['bert.embeddings.word_embeddings.weight']
_skip_keys = []
def preprocess_bert_weights(ckpt):
    keys = list(ckpt.keys())
    for key in keys:
        if "LayerNorm" in key:
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
                ckpt[new_key] = ckpt.pop(key)
            elif "beta" in key:
                new_key = key.replace("beta", "bias")
                ckpt[new_key] = ckpt.pop(key)                
    for key in _skip_keys:
        if key in ckpt.keys():
            ckpt.pop(key)
    return ckpt


class Trainer:
    def __init__(self,
                 train_epoches=3,
                 start_epoch=0,
                 load_pretrain=True
                 ):
        self.vocab_size = int(config["vocab_size"])
        self.start_epoch = start_epoch
        self.train_epoches = train_epoches

        bertconfig = BertConfig(**bert_config)
        self.bert_model = DocClassifier(config=bertconfig)

        if load_pretrain:
            ckpt_file = config["pretrained"]
            ckpt = torch.load(ckpt_file)
            ckpt = preprocess_bert_weights(ckpt)
            if 'model_state_dict' in ckpt.keys():
                ckpt = ckpt['model_state_dict']
            info = self.bert_model.load_state_dict(ckpt, strict=False)
            logging.info("**missing keys: {}".format(info[0]))
            logging.info("**unexpected keys: {}".format(info[1]))

        self.bert_model = self.bert_model.cuda()
        self.bert_model = torch.nn.DataParallel(self.bert_model)

        train_dataset = CLSDataset(corpus_path=config["train_corpus_path"],
                                   vocab_path=config['vocab_path'],
                                   max_seq_len=config['max_seq_len'])
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=config['batch_size'],
                                           shuffle=True,
                                           num_workers=0
                                           )

        test_dataset = CLSDataset(corpus_path=config["val_corpus_path"],
                                  vocab_path=config['vocab_path'],
                                  max_seq_len=config['max_seq_len'])
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=config['batch_size'],
                                          shuffle=False,
                                          num_workers=0)
        
        no_decay = ["bias", "LayerNorm.weight"]
        parameters_base_with_decay, parameters_base_no_decay, parameters_extra = [], [], []
        keys_base_with_decay, keys_base_no_decay, keys_extra = [], [], []
        for n, p in self.bert_model.named_parameters():
            if 'bert' in n:
                if any(nd in n for nd in no_decay):
                    parameters_base_no_decay.append(p)
                    keys_base_no_decay.append(n)
                else:
                    parameters_base_with_decay.append(p)
                    keys_base_with_decay.append(n)
            else:
                parameters_extra.append(p)
                keys_extra.append(n)
        logging.info("base_with_decay: {}".format(keys_base_with_decay))
        logging.info("base_no_decay: {}".format(keys_base_no_decay))
        logging.info("extra: {}".format(keys_extra))

        optimizer_grouped_parameters = [
            {
                "params": parameters_base_with_decay,
                "weight_decay": config["weight_decay"]
            },
            {
                "params": parameters_base_no_decay,
                "weight_decay": 0.0
            },
        ]

        self.optimizer_base = torch.optim.AdamW(optimizer_grouped_parameters, lr=config["lr"])
        self.optimizer_extra = torch.optim.Adam(parameters_extra, lr=config["lr"] * 10)

        # learning rate schedule
        num_iter_per_epoch = math.ceil(len(train_dataset) / config['batch_size'] / config["grad_accum_steps"])
        max_iters = train_epoches * num_iter_per_epoch
        logging.info("max iters: {}".format(max_iters))
        warmup_iters = int(num_iter_per_epoch * config["warmup_ratio"])
        logging.info("warmup iters: {}".format(warmup_iters))
        if config["scheduler"] == "cosine":
            self.scheduler_base = WarmupCosineLR(self.optimizer_base,
                                            max_iters=max_iters,
                                            warmup_iters=warmup_iters,
                                            warmup_factor=0.001)
            self.scheduler_extra = WarmupCosineLR(self.optimizer_extra,
                                            max_iters=max_iters,
                                            warmup_iters=warmup_iters,
                                            warmup_factor=0.001)
        else:
            milestones = [int(max_iters*0.6), int(max_iters*0.95)]
            self.scheduler_base = WarmupMultiStepLR(self.optimizer_base,
                                               milestones=milestones,
                                               gamma=0.1,
                                               warmup_factor=0.001,
                                               warmup_iters=warmup_iters)
            self.scheduler_extra = WarmupMultiStepLR(self.optimizer_extra,
                                               milestones=milestones,
                                               gamma=0.1,
                                               warmup_factor=0.001,
                                               warmup_iters=warmup_iters)
        iters_passed = int(num_iter_per_epoch * start_epoch)
        for _ in range(iters_passed):
            self.scheduler.step()

        logging.info("Total Parameters: {}".format(sum([p.nelement() for p in self.bert_model.parameters()])))

    def run(self):
        for epoch in range(self.start_epoch, self.train_epoches):
            self._train(epoch)
            self.save_state_dict(trainer.bert_model, epoch, config["output_path"], "bert.model")
            self._test(epoch)
    

    def compute_loss(self, predictions, labels):
        # predictions = predictions.view(-1, 14)
        labels = labels.view(-1)
        loss = F.cross_entropy(predictions, labels)
        return loss


    def load_model(self, model, model_dir="../output", load_bert=False):
        checkpoint_dir = self.find_most_recent_state_dict(model_dir)
        checkpoint = torch.load(checkpoint_dir)

        if load_bert:
            checkpoint["model_state_dict"] = {k[5:]: v for k, v in checkpoint["model_state_dict"].items()
                                              if k[:4] == "bert" and "pooler" not in k}
        res = model.module.load_state_dict(checkpoint["model_state_dict"], strict=False)
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
            token_id, attn_mask, labels = data
            token_id = token_id.cuda()
            attn_mask = attn_mask.cuda()
            labels = labels.cuda()

            logits = self.bert_model(text_input=token_id, attn_mask=attn_mask)
            loss = self.compute_loss(logits, labels)
            loss = loss / config["grad_accum_steps"]

            predictions = torch.argmax(logits, dim=-1).detach().cpu().numpy().reshape(-1).tolist()
            labels = labels.cpu().numpy().reshape(-1).tolist()
            all_predictions.extend(predictions)
            all_labels.extend(labels)

            if train:
                loss.backward()
                grad_step += 1
                
                if grad_step == config["grad_accum_steps"]:
                    self.optimizer_base.step()
                    self.optimizer_extra.step()
                    self.scheduler_base.step()
                    self.scheduler_extra.step()

                    self.optimizer_base.zero_grad()
                    self.optimizer_extra.zero_grad()
                    grad_step = 0

            total_loss += loss.item() * config["grad_accum_steps"]

            if train:
                current_lr = self.optimizer_base.param_groups[0]['lr']
                # current_lr = [self.scheduler_base.get_lr(), self.scheduler_extra.get_lr()]
                log_dic = {
                    "epoch": epoch,
                    "lr": current_lr,
                    "loss": total_loss/(i+1),
                    "step": i+1
                }

            else:
                log_dic = {
                    "epoch": epoch,
                    "loss": total_loss/(i+1),
                    "step": i+1
                }

            if (i+1) % config["log_interval"] == 0:
                fscore = metrics.f1_score(all_labels, all_predictions, average='macro')
                equ = np.array(all_predictions) == np.array(all_labels)
                acc = np.sum(equ) / len(equ)
                log_dic['f1_score'] = fscore
                log_dic['acc'] = acc
                data_iter.write(str({k: v for k, v in log_dic.items()}))

                log_str = ''
                for k, v in log_dic.items():
                    log_str += '{}: {}\t'.format(k, v)
                logging.info(log_str)
        
        # epoch score
        fscore = metrics.f1_score(all_labels, all_predictions, average='macro')
        equ = np.array(all_predictions) == np.array(all_labels)
        acc = np.sum(equ) / len(equ)
        if not train:
            logging.info('***************')
        logging.info("{} Epoch: {}.\tfscore: {:.3f}.\tacc: {:.3f}.".format(str_code, epoch, fscore, acc))

        return fscore


    def find_most_recent_state_dict(self, dir_path):
        dic_lis = [i for i in os.listdir(dir_path)]
        if len(dic_lis) == 0:
            raise FileNotFoundError("can not find any state dict in {}!".format(dir_path))
        dic_lis = [i for i in dic_lis if "model" in i]
        dic_lis = sorted(dic_lis, key=lambda k: int(k.split(".")[-1]))
        return dir_path + "/" + dic_lis[-1]


    def save_state_dict(self, model, epoch, state_dict_dir="../output", file_path="bert.model"):
        if not os.path.exists(state_dict_dir):
            os.mkdir(state_dict_dir)
        save_path = state_dict_dir + "/" + file_path + ".epoch.{}".format(str(epoch))
        torch.save({"model_state_dict": model.module.state_dict()}, save_path)
        print("{} saved!".format(save_path))


if __name__ == '__main__':
    start_epoch = 0
    train_epoches = 10
    load_model = False

    trainer = Trainer(start_epoch=start_epoch,
                      train_epoches=train_epoches)
    if load_model:
        trainer.load_model(trainer.bert_model, model_dir=config["output_path"])

    trainer.run()

    

