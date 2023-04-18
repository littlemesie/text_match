# -*- coding:utf-8 -*-

"""
@date: 2023/4/11 下午5:34
@summary: 数据加载
"""
import torch
import pandas as pd
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, filepath, ul=True):
        """
        Args:
            filepath: 文件路径
            ul: True:无监督 False:有监督
        """
        super(TextDataset, self).__init__()
        self.ul = ul
        if not self.ul:
            self.train, self.label = self.load_data(filepath)
        else:
            self.train = self.load_data(filepath)

    def load_data(self, path):
        train = pd.read_csv(path, error_bad_lines=False)

        texts = train.text.to_list()
        if not self.ul:
            labels = train.label.to_list()
            return texts, labels
        else:
            return texts

    def __len__(self):
        return len(self.train)

    def __getitem__(self, item):
        text = self.train[item]
        if not self.ul:
            label = self.label[item]
            return text, label
        else:
            return text

class BatchTextDataset:
    def __init__(self, tokenizer, max_len=512, ul=True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.ul = ul

    def __call__(self, batch):
        batch_text = [item[0] for item in batch]

        batch_token, batch_segment, batch_mask = list(), list(), list()
        for text in batch_text:
            if len(text) > self.max_len - 2:
                text = text[:self.max_len - 2]
            token = self.tokenizer.tokenize(text)
            token = ['[CLS]'] + token + ['[SEP]']
            token_id = self.tokenizer.convert_tokens_to_ids(token)

            padding = [0] * (self.max_len - len(token_id))
            token_id = token_id + padding
            batch_token.append(token_id)

        batch_tensor_token = torch.tensor(batch_token)
        if not self.ul:
            batch_label = [item[1] for item in batch]
            batch_tensor_label = torch.tensor(batch_label)
            return batch_tensor_token, batch_tensor_label
        else:
         return batch_tensor_token

class BertBatchTextDataset:
    """
    call function for tokenizing and getting batch text
    """
    def __init__(self, tokenizer, max_len=312,  ul=True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.ul = ul

    def text2id(self, batch_text):
        return self.tokenizer(batch_text, max_length=self.max_len,
                              truncation=True, padding='max_length', return_tensors='pt')

    def __call__(self, batch):
        batch_text = [item[0] for item in batch]

        source = self.text2id(batch_text)
        token = source.get('input_ids').squeeze(1)
        mask = source.get('attention_mask').squeeze(1)
        segment = source.get('token_type_ids').squeeze(1)
        if not self.ul:
            batch_label = [item[1] for item in batch]
            label = torch.tensor(batch_label)
            return token, segment, mask, label
        else:
            return token, segment, mask


def convert_ids(text, tokenizer, max_seq_length=312):
    """
    Args:
        text: 文本 可以是单文本 也可以是list
        tokenizer:
        max_seq_length:

    Returns:

    """
    source = tokenizer(text, max_length=max_seq_length, truncation=True, padding='max_length',
                       return_tensors='pt')
    input_ids = source.get('input_ids').squeeze(1)
    attention_mask = source.get('attention_mask').squeeze(1)
    token_type_ids = source.get('token_type_ids').squeeze(1)
    return input_ids, attention_mask, token_type_ids
