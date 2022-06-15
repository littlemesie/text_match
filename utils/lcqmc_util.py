# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2022/6/10 下午3:46
@summary: lsqmc data process
"""
import re
import numpy as np
import pandas as pd
import torch
from hanziconv import HanziConv
from torch.utils.data import Dataset
from utils.load_util import load_vocab

class LCQMC_Dataset(Dataset):
    def __init__(self, LCQMC_file, vocab_file, max_char_len):
        p, h, self.label = load_sentences(LCQMC_file)
        word2idx, _, _ = load_vocab(vocab_file)
        self.p_list, self.p_lengths, self.h_list, self.h_lengths = word_index(p, h, word2idx, max_char_len)
        self.p_list = torch.from_numpy(self.p_list).type(torch.long)
        self.h_list = torch.from_numpy(self.h_list).type(torch.long)
        self.max_length = max_char_len

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.p_list[idx], self.p_lengths[idx], self.h_list[idx], self.h_lengths[idx], self.label[idx]


# 加载word_index训练数据
# def load_sentences(file, data_size=None):
#     df = pd.read_csv(file)
#     p = map(get_word_list, df['sentence1'].values[0:data_size])
#     h = map(get_word_list, df['sentence2'].values[0:data_size])
#     label = df['label'].values[0:data_size]
#     #p_c_index, h_c_index = word_index(p, h)
#     return p, h, label

def load_sentences(file, data_size=None):
    df = pd.read_csv(file, sep='\t', names=['sentence1', 'sentence2', 'label'])
    p = map(get_word_list, df['sentence1'].values[0:data_size])
    h = map(get_word_list, df['sentence2'].values[0:data_size])
    label = df['label'].values[0:data_size]
    # p_c_index, h_c_index = word_index(p, h)
    return p, h, label


# word->index
def word_index(p_sentences, h_sentences, word2idx, max_char_len):
    p_list, p_length, h_list, h_length = [], [], [], []
    for p_sentence, h_sentence in zip(p_sentences, h_sentences):
        p = [word2idx[word] for word in p_sentence if word in word2idx.keys()]
        h = [word2idx[word] for word in h_sentence if word in word2idx.keys()]
        p_list.append(p)
        p_length.append(min(len(p), max_char_len))
        h_list.append(h)
        h_length.append(min(len(h), max_char_len))
    p_list = pad_sequences(p_list, maxlen=max_char_len)
    h_list = pad_sequences(h_list, maxlen=max_char_len)
    return p_list, p_length, h_list, h_length

''' 把句子按字分开，中文按字分，英文数字按空格, 大写转小写，繁体转简体'''

def get_word_list(query):
    query = HanziConv.toSimplified(query.strip())
    regEx = re.compile('[\\W]+')  # 我们可以使用正则表达式来切分句子，切分的规则是除单词，数字外的任意字符串
    res = re.compile(r'([\u4e00-\u9fa5])')  # [\u4e00-\u9fa5]中文范围
    sentences = regEx.split(query.lower())
    str_list = []
    for sentence in sentences:
        if res.split(sentence) == None:
            str_list.append(sentence)
        else:
            ret = res.split(sentence)
            str_list.extend(ret)
    return [w for w in str_list if len(w.strip()) > 0]

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post',
                  truncating='post', value=0.):
    """ pad_sequences
    把序列长度转变为一样长的，如果设置了maxlen则长度统一为maxlen，如果没有设置则默认取
    最大的长度。填充和截取包括两种方法，post与pre，post指从尾部开始处理，pre指从头部
    开始处理，默认都是从尾部开始。
    Arguments:
        sequences: 序列
        maxlen: int 最大长度
        dtype: 转变后的数据类型
        padding: 填充方法'pre' or 'post'
        truncating: 截取方法'pre' or 'post'
        value: float 填充的值
    Returns:
        x: numpy array 填充后的序列维度为 (number_of_sequences, maxlen)
    """
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x


class BertDataPrecessForSentence(Dataset):
    """
    对文本进行处理
    """
    def __init__(self, bert_tokenizer, LCQMC_file, max_char_len=103):
        """
        bert_tokenizer :分词器
        LCQMC_file     :语料文件
        """
        self.bert_tokenizer = bert_tokenizer
        self.max_seq_len = max_char_len
        self.seqs, self.seq_masks, self.seq_segments, self.labels = self.get_input(LCQMC_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seqs[idx], self.seq_masks[idx], self.seq_segments[idx], self.labels[idx]

    # 获取文本与标签
    def get_input(self, file):
        """
        通对输入文本进行分词、ID化、截断、填充等流程得到最终的可用于模型输入的序列。
        入参:
            dataset     : pandas的dataframe格式，包含三列，第一,二列为文本，第三列为标签。标签取值为{0,1}，其中0表示负样本，1代表正样本。
            max_seq_len : 目标序列长度，该值需要预先对文本长度进行分别得到，可以设置为小于等于512（BERT的最长文本序列长度为512）的整数。
        出参:
            seq         : 在入参seq的头尾分别拼接了'CLS'与'SEP'符号，如果长度仍小于max_seq_len，则使用0在尾部进行了填充。
            seq_mask    : 只包含0、1且长度等于seq的序列，用于表征seq中的符号是否是有意义的，如果seq序列对应位上为填充符号，
                          那么取值为1，否则为0。
            seq_segment : shape等于seq，因为是单句，所以取值都为0。
            labels      : 标签取值为{0,1}，其中0表示负样本，1代表正样本。
        """
        # df = pd.read_csv(file)
        df = pd.read_csv(file, sep='\t', names=['sentence1', 'sentence2', 'label'])
        sentences_1 = map(HanziConv.toSimplified, df['sentence1'].values)
        sentences_2 = map(HanziConv.toSimplified, df['sentence2'].values)
        labels = df['label'].values
        # 切词
        tokens_seq_1 = list(map(self.bert_tokenizer.tokenize, sentences_1))
        tokens_seq_2 = list(map(self.bert_tokenizer.tokenize, sentences_2))
        # 获取定长序列及其mask
        result = list(map(self.trunate_and_pad, tokens_seq_1, tokens_seq_2))
        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]
        return torch.Tensor(seqs).type(torch.long), torch.Tensor(seq_masks).type(torch.long), \
               torch.Tensor(seq_segments).type(torch.long), torch.Tensor(labels).type(torch.long)

    def trunate_and_pad(self, tokens_seq_1, tokens_seq_2):
        """
        1. 如果是单句序列，按照BERT中的序列处理方式，需要在输入序列头尾分别拼接特殊字符'CLS'与'SEP'，
           因此不包含两个特殊字符的序列长度应该小于等于max_seq_len-2，如果序列长度大于该值需要那么进行截断。
        2. 对输入的序列 最终形成['CLS',seq,'SEP']的序列，该序列的长度如果小于max_seq_len，那么使用0进行填充。
        入参:
            seq_1       : 输入序列，在本处其为单个句子。
            seq_2       : 输入序列，在本处其为单个句子。
            max_seq_len : 拼接'CLS'与'SEP'这两个特殊字符后的序列长度

        出参:
            seq         : 在入参seq的头尾分别拼接了'CLS'与'SEP'符号，如果长度仍小于max_seq_len，则使用0在尾部进行了填充。
            seq_mask    : 只包含0、1且长度等于seq的序列，用于表征seq中的符号是否是有意义的，如果seq序列对应位上为填充符号，
                          那么取值为1，否则为0。
            seq_segment : shape等于seq，单句，取值都为0 ，双句按照01切分

        """
        # 对超长序列进行截断
        if len(tokens_seq_1) > ((self.max_seq_len - 3) // 2):
            tokens_seq_1 = tokens_seq_1[0:(self.max_seq_len - 3) // 2]
        if len(tokens_seq_2) > ((self.max_seq_len - 3) // 2):
            tokens_seq_2 = tokens_seq_2[0:(self.max_seq_len - 3) // 2]
        # 分别在首尾拼接特殊符号
        seq = ['[CLS]'] + tokens_seq_1 + ['[SEP]'] + tokens_seq_2 + ['[SEP]']
        seq_segment = [0] * (len(tokens_seq_1) + 2) + [1] * (len(tokens_seq_2) + 1)
        # ID化
        seq = self.bert_tokenizer.convert_tokens_to_ids(seq)
        # 根据max_seq_len与seq的长度产生填充序列
        padding = [0] * (self.max_seq_len - len(seq))
        # 创建seq_mask
        seq_mask = [1] * len(seq) + padding
        # 创建seq_segment
        seq_segment = seq_segment + padding
        # 对seq拼接填充序列
        seq += padding
        assert len(seq) == self.max_seq_len
        assert len(seq_mask) == self.max_seq_len
        assert len(seq_segment) == self.max_seq_len
        return seq, seq_mask, seq_segment