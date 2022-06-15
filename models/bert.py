# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2022/6/14 下午2:13
@summary: Bert 模型
"""
import torch
from torch import nn
from torch.nn import functional
from transformers import BertForSequenceClassification

class BertModel(nn.Module):
    def __init__(self, model_path='bert-base-chinese'):
        super(BertModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)  # /bert_pretrain/
        self.device = torch.device("cuda")
        for param in self.bert.parameters():
            param.requires_grad = True  # 每个参数都要 求梯度

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.bert(input_ids=batch_seqs, attention_mask=batch_seq_masks,
                              token_type_ids=batch_seq_segments, labels=labels)
        probabilities = functional.softmax(logits, dim=-1)
        return loss, logits, probabilities