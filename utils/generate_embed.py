# -*- coding:utf-8 -*-

"""
@date: 2023/4/17 上午11:20
@summary:
"""
import torch
import numpy as np


class GenerateEmbed:
    def __init__(self,
                 model,
                 tokenizer,
                 device="cpu",
                 max_seq_length=128,
                 batch_size=32):
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.device = device

    def convert_ids(self, text):
        """文本 可以是单文本 也可以是list"""
        source = self.tokenizer(text, max_length=self.max_seq_length, truncation=True, padding='max_length',
                           return_tensors='pt')
        input_ids = source.get('input_ids').squeeze(1)
        attention_mask = source.get('attention_mask').squeeze(1)
        token_type_ids = source.get('token_type_ids').squeeze(1)
        return input_ids, attention_mask, token_type_ids

    def generate_one(self, text):
        """"""
        with torch.no_grad():
            input_ids, attention_mask, token_type_ids = self.convert_ids(text)
            input_ids = input_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            embed = self.model(input_ids, token_type_ids, attention_mask)
            embed = np.array(embed)
        return embed

    def generate_many(self, data):
        """"""
        all_embeddings = None
        with torch.no_grad():
            for i in range(int(len(data) / self.batch_size) + 1):
                print(i)
                batch_data = data[i * self.batch_size: (i+1) * self.batch_size]
                input_ids, attention_mask, token_type_ids = self.convert_ids(batch_data)
                input_ids = input_ids.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                embed = self.model(input_ids, token_type_ids, attention_mask)
                embed = np.array(embed)
                if i == 0:
                    all_embeddings = embed
                else:
                    all_embeddings = np.concatenate((all_embeddings, embed), axis=0)
        return all_embeddings
