# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2022/6/10 下午3:50
@summary: 加载文件或者模型
"""
import gensim
import numpy as np

# 加载字典
def load_vocab(vocab_file):
    vocab = [line.strip() for line in open(vocab_file, encoding='utf-8').readlines()]
    word2idx = {word: index for index, word in enumerate(vocab)}
    idx2word = {index: word for index, word in enumerate(vocab)}
    return word2idx, idx2word, vocab

def load_embeddings(embdding_path):
    model = gensim.models.KeyedVectors.load_word2vec_format(embdding_path, binary=False)
    embedding_matrix = np.zeros((len(model.index_to_key) + 1, model.vector_size))
    # 填充向量矩阵
    for idx, word in enumerate(model.index_to_key):
        embedding_matrix[idx + 1] = model[word]  # 词向量矩阵
    return embedding_matrix