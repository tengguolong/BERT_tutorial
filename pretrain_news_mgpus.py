import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import tqdm
import math
import pdb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.news_pretrain_dataset import BERTDataset
from models.bert_base import BertModel, BertConfig, gelu
from utils.lr_scheduler import WarmupCosineLR, WarmupMultiStepLR


config = {}
config["train_corpus_path"] = "corpus/news/trainval.csv"
config["test_corpus_path"] = "corpus/news/test_a.csv"
config["word2idx_path"] = "corpus/news_word2idx.json"
config["output_path"] = "save/news_pretrain/"
config["batch_size"] = 32
config["grad_accum_steps"] = 4
config["max_seq_len"] = 512
config["lr"] = 5e-5
config["weight_decay"] = 1e-2
config["scheduler"] = 'cosine'
config["num_workers"] = 0
config["log_interval"] = 200
config["val_interval"] = 1


bert_config = dict(vocab_size=6874, # 字典字数
                   hidden_size=256, # 隐藏层维度也就是字向量维度
                   num_hidden_layers=6, # transformer block 的个数
                   num_attention_heads=8, # 注意力机制"头"的个数
                   intermediate_size=256*4, # feedforward层线性映射的维度
                   hidden_act="gelu", # 激活函数
                   hidden_dropout_prob=0.1, # dropout的概率
                   attention_probs_dropout_prob=0.1,
                   max_position_embeddings=512,
                   type_vocab_size=2, # 用来做next sentence预测, 这里预留了256个分类, 其实我们目前用到的只有0和1
                   initializer_range=0.02 # 用来初始化模型参数的标准差
)


class BertMLMPretrain(nn.Module):
    def __init__(self, config):
        super(BertMLMPretrain, self).__init__()
        self.bert = BertModel(config, add_pooling_layer=False)
        self.num_classes = config.vocab_size

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = gelu
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

        self._init_extra_weights(self.decoder)
        self._init_extra_weights(self.dense)
        self._init_layer_norm(self.LayerNorm)
    

    def _init_extra_weights(self, module):
        """Initialize the weights"""
        name = module.__class__.__name__
        print("init", name)
        module.weight.data.normal_(mean=0.0, std=0.001)
        if module.bias is not None:
            module.bias.data.zero_()
    

    def _init_layer_norm(self, module):
        name = module.__class__.__name__
        print("init", name)
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


    def forward(self, text_input, attn_mask, type_id=None, labels=None):
        outputs = self.bert(text_input, token_type_ids=type_id, attention_mask=attn_mask)
        token_embeddings, pooled_output = outputs
        # token_embeddings: [bs, L, H]

        hidden_states = self.dense(token_embeddings)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        return self.decoder(hidden_states)
        


class Pretrainer:
    def __init__(self,
                 train_epoches=10,
                 start_epoch=0
                 ):
        self.start_epoch = start_epoch
        self.train_epoches = train_epoches

        # model
        bertconfig = BertConfig(**bert_config)
        self.bert_model = BertMLMPretrain(bertconfig).cuda()
        self.bert_model = torch.nn.DataParallel(self.bert_model)

        # dataloader
        train_dataset = BERTDataset(corpus_path=config["train_corpus_path"],
                                    word2idx_path=config["word2idx_path"],
                                    seq_len=config["max_seq_len"]
                                    )
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=config["batch_size"],
                                           shuffle=False,
                                           num_workers=config["num_workers"])
        test_dataset = BERTDataset(corpus_path=config["test_corpus_path"],
                                   word2idx_path=config["word2idx_path"],
                                   seq_len=config["max_seq_len"]
                                   )
        self.test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"],
                                          num_workers=config["num_workers"])
        
        # optimizer
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
        num_iter_per_epoch = math.ceil(len(train_dataset) / config["batch_size"] / config["grad_accum_steps"])
        max_iters = train_epoches * num_iter_per_epoch
        print("max iters:", max_iters)
        warmup_iters = int(max_iters*0.02)
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
            self.do_train(epoch)
            self.save_state_dict(trainer.bert_model, epoch, dir_path=config["output_path"],
                                    file_path="bert.model")
            self.do_test(epoch)

    def do_test(self, epoch, df_path="./output_wiki_bert/df_log.pickle"):
        self.bert_model.eval()
        with torch.no_grad():
            return self.iteration(epoch, self.test_dataloader, train=False, df_path=df_path)

    def load_model(self, model, dir_path="./output"):
        # 加载模型
        checkpoint_dir = self.find_most_recent_state_dict(dir_path)
        checkpoint = torch.load(checkpoint_dir)
        res = model.module.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(res)
        torch.cuda.empty_cache()
        model = model.cuda()
        print("{} loaded for training!".format(checkpoint_dir))

    def do_train(self, epoch, df_path=None):
        self.bert_model.train()
        self.iteration(epoch, self.train_dataloader, train=True, df_path=df_path)

    def compute_loss(self, predictions, labels, num_class=2, ignore_index=None):
        if ignore_index is None:
            loss_func = torch.nn.CrossEntropyLoss()
        else:
            loss_func = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        return loss_func(predictions.view(-1, num_class), labels.view(-1))

    def get_mlm_accuracy(self, predictions, labels):
        predictions = torch.argmax(predictions, dim=-1, keepdim=False)
        mask = (labels > 0).cuda()
        mlm_accuracy = torch.sum((predictions == labels) * mask).float()
        mlm_accuracy /= (torch.sum(mask).float() + 1e-8)
        return mlm_accuracy.item()


    def iteration(self, epoch, data_loader, train=True, df_path=None):
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        total_mlm_loss = 0
        total_mlm_acc = 0
        grad_step = 0

        for i, data in data_iter:
            token_id, labels, attn_mask = data
            token_id = token_id.cuda()
            attn_mask = attn_mask.cuda()
            labels = labels.cuda()

            mlm_preds = self.bert_model(text_input=token_id, attn_mask=attn_mask, labels=labels)
            mlm_acc = self.get_mlm_accuracy(mlm_preds, labels)
            loss = self.compute_loss(mlm_preds, labels, bert_config['vocab_size'], ignore_index=0)
            loss = loss / config["grad_accum_steps"]

            # 3. backward and optimization only in train
            if train:
                loss.backward()
                grad_step += 1
                
                if grad_step == config["grad_accum_steps"]:
                    self.optimizer.step()
                    self.scheduler.step()

                    self.optimizer.zero_grad()
                    grad_step = 0


            total_mlm_loss += loss.item() * config["grad_accum_steps"]
            total_mlm_acc += mlm_acc

            if train:
                current_lr = self.optimizer.param_groups[0]['lr']
                log_dic = {
                    "epoch": epoch,
                    "lr": current_lr,
                   "train_mlm_loss": total_mlm_loss / (i + 1),
                   "train_mlm_acc": total_mlm_acc / (i + 1),
                   "test_next_sen_loss": 0, "test_mlm_loss": 0,
                   "test_next_sen_acc": 0, "test_mlm_acc": 0
                }

            else:
                log_dic = {
                    "epoch": epoch,
                   "test_mlm_loss": total_mlm_loss / (i + 1),
                   "test_mlm_acc": total_mlm_acc / (i + 1),
                   "train_next_sen_loss": 0, "train_mlm_loss": 0,
                   "train_next_sen_acc": 0, "train_mlm_acc": 0
                }


            if (i+1) % config["log_interval"] == 0:
                data_iter.write(str({k: v for k, v in log_dic.items() if v != 0 and k != "epoch"}))

        data_iter.write(str({k: v for k, v in log_dic.items() if v != 0}))

        if not train:
            log_dic = {k: v for k, v in log_dic.items() if v != 0 and k != "epoch"}
            return float(log_dic["test_mlm_loss"])

    def find_most_recent_state_dict(self, dir_path):
        dic_lis = [i for i in os.listdir(dir_path)]
        if len(dic_lis) == 0:
            raise FileNotFoundError("can not find any state dict in {}!".format(dir_path))
        dic_lis = [i for i in dic_lis if "model" in i]
        dic_lis = sorted(dic_lis, key=lambda k: int(k.split(".")[-1]))
        return dir_path + "/" + dic_lis[-1]

    def save_state_dict(self, model, epoch, dir_path="./output", file_path="bert.model"):
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        save_path = dir_path+ "/" + file_path + ".epoch.{}".format(str(epoch))
        torch.save({"model_state_dict": model.module.state_dict()}, save_path)
        print("{} saved!".format(save_path))


if __name__ == '__main__':
    start_epoch = 0
    train_epoches = 20
    load_model = False

    trainer = Pretrainer(start_epoch=start_epoch,
                        train_epoches=train_epoches)
    if load_model:
        trainer.load_model(trainer.bert_model, model_dir=config["output_path"])

    trainer.run()