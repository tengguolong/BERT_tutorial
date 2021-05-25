from torch import nn
import torch.nn.functional as F
from models.bert_model import *



class Bert_News_Classification(nn.Module):
    def __init__(self, config):
        super(Bert_News_Classification, self).__init__()
        self.bert = BertModel(config)
        self.num_classes = 14
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.final_dense = nn.Linear(config.hidden_size, self.num_classes) # 14 is the number of types

    def compute_loss(self, predictions, labels):
        # 将预测和标记的维度展平, 防止出现维度不一致
        predictions = predictions.view(-1, self.num_classes)
        labels = labels.view(-1)
        loss = F.cross_entropy(predictions, labels)
        return loss

    def forward(self, text_input, positional_enc, labels=None):
        encoded_layers, _ = self.bert(text_input, positional_enc,
                                    output_all_encoded_layers=True)

        sequence_output = encoded_layers[-1][:, 0, :]
        # # sequence_output的维度是[batch_size, embed_dim]
        # 下面是[batch_size, hidden_dim] 到 [batch_size, num_classes]的映射
        predictions = self.final_dense(sequence_output)

        if labels is not None:
            # 计算loss
            loss = self.compute_loss(predictions, labels)
            return torch.argmax(predictions, dim=-1), loss
        else:
            return torch.argmax(predictions, dim=-1)
