
from torch.utils.data import DataLoader

from dataset.news_dataset import CLSDataset
from models.bert_news_classification import *

import tqdm
import pandas as pd
import numpy as np
import configparser
import os
import json
import pdb


class Trainer:
    def __init__(self, max_seq_len,
                 batch_size,
                 lr, # 学习率
                 with_cuda=True, # 是否使用GPU, 如未找到GPU, 则自动切换CPU
                 ):
        config_ = configparser.ConfigParser()
        config_.read("./config/news_cls_config.ini")
        self.config = config_["DEFAULT"]
        self.vocab_size = int(self.config["vocab_size"])
        self.batch_size = batch_size
        self.lr = lr
        # 加载字典
        with open(self.config["word2idx_path"], "r", encoding="utf-8") as f:
            self.word2idx = json.load(f)
        # 判断是否有可用GPU
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        # 允许的最大序列长度
        self.max_seq_len = max_seq_len
        # 定义模型超参数
        bertconfig = BertConfig(vocab_size=self.vocab_size)
        # 初始化BERT模型
        self.bert_model = Bert_News_Classification(config=bertconfig)
        # 将模型发送到计算设备(GPU或CPU)
        self.bert_model.to(self.device)
        # 声明训练数据集, 按照pytorch的要求定义数据集class
        
        # 声明测试数据集
        test_dataset = CLSDataset(corpus_path=self.config["test_corpus_path"],
                                  word2idx=self.word2idx,
                                  max_seq_len=self.max_seq_len,
                                  data_regularization=False
                                  )
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=self.batch_size,
                                          shuffle=False,
                                          num_workers=1,
                                          collate_fn=lambda x: x)
        # 初始化位置编码
        self.hidden_dim = bertconfig.hidden_size
        self.positional_enc = self.init_positional_encoding()
        # 扩展位置编码的维度, 留出batch维度,
        # 即positional_enc: [batch_size, embedding_dimension]
        self.positional_enc = torch.unsqueeze(self.positional_enc, dim=0)


    def init_positional_encoding(self):
        position_enc = np.array([
            [pos / np.power(10000, 2 * i / self.hidden_dim) for i in range(self.hidden_dim)]
            if pos != 0 else np.zeros(self.hidden_dim) for pos in range(self.max_seq_len)])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
        denominator = np.sqrt(np.sum(position_enc**2, axis=1, keepdims=True))
        # 归一化
        position_enc = position_enc / (denominator + 1e-8)
        position_enc = torch.from_numpy(position_enc).type(torch.FloatTensor)
        return position_enc


    def load_model(self, model, dir_path="../output", load_bert=False):
        checkpoint_dir = self.find_most_recent_state_dict(dir_path)
        checkpoint = torch.load(checkpoint_dir)
        # 情感分析模型刚开始训练的时候, 需要载入预训练的BERT,
        # 这是我们不载入模型原本用于训练Next Sentence的pooler
        # 而是重新初始化了一个
        if load_bert:
            checkpoint["model_state_dict"] = {k[5:]: v for k, v in checkpoint["model_state_dict"].items()
                                              if k[:4] == "bert" and "pooler" not in k}
        res = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(res)
        torch.cuda.empty_cache()
        model.to(self.device)
        print("{} loaded!".format(checkpoint_dir))

    def test(self, epoch=0):
        # 一个epoch的测试, 并返回测试集的auc
        self.bert_model.eval()
        with torch.no_grad():
            return self.iteration(epoch, self.test_dataloader, train=False)

    def padding(self, output_dic_lis):
        """动态padding, 以当前mini batch内最大的句长进行补齐长度"""
        text_input = [i["text_input"] for i in output_dic_lis]
        text_input = torch.nn.utils.rnn.pad_sequence(text_input, batch_first=True)
        # label = torch.cat([i["label"] for i in output_dic_lis])
        return {"text_input": text_input}

    def iteration(self, epoch, data_loader, train=False, df_name="df_log.pickle"):
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="testing",
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        
        all_predictions = []

        for i, data in data_iter:
            # padding
            data = self.padding(data)
            # 将数据发送到计算设备
            data = {key: value.to(self.device) for key, value in data.items()}
            # 根据padding之后文本序列的长度截取相应长度的位置编码,
            # 并发送到计算设备
            positional_enc = self.positional_enc[:, :data["text_input"].size()[-1], :].to(self.device)

            # 正向传播, 得到预测结果和loss
            predictions = self.bert_model.forward(text_input=data["text_input"],
                                                  positional_enc=positional_enc)

            predictions = predictions.detach().cpu().numpy().reshape(-1).tolist()
            all_predictions.extend(predictions)
        
        return all_predictions


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



if __name__ == '__main__':
    def init_trainer(dynamic_lr, batch_size):
        trainer = Trainer(max_seq_len=384,
                          batch_size=batch_size,
                          lr=dynamic_lr,
                          with_cuda=True,)
        return trainer, dynamic_lr

    trainer, dynamic_lr = init_trainer(dynamic_lr=1e-6, batch_size=24)
    trainer.load_model(trainer.bert_model, dir_path=trainer.config["state_dict_dir"])
    predictions = trainer.test()
    print(len(predictions))
    submission = pd.DataFrame({'label': predictions})
    submission.to_csv('submission.csv', index=False)
    



