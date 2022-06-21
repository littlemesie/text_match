# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2022/6/21 下午3:24
@summary: albert model
"""

import torch
from torch import nn
from torch.nn import functional
from transformers import AlbertForSequenceClassification

class AlbertModel(nn.Module):
    def __init__(self, model_path='albert_chinese_base'):
        super(AlbertModel, self).__init__()
        self.albert = AlbertForSequenceClassification.from_pretrained(model_path, num_labels=2)
        self.device = torch.device("cuda")
        for param in self.albert.parameters():
            param.requires_grad = True  # 每个参数都要 求梯度

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.albert(input_ids=batch_seqs, attention_mask=batch_seq_masks,
                              token_type_ids=batch_seq_segments, labels=labels)[:2]
        probabilities = functional.softmax(logits, dim=-1)
        return loss, logits, probabilities