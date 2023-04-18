# -*- coding:utf-8 -*-

"""
@date: 2023/4/18 下午2:48
@summary:
"""
from gensim import corpora, similarities
from gensim.models.tfidfmodel import TfidfModel
from utils.cut_word_util import cut_word


def tfidf_similarity(words, text, threshold=0.5):
    """tf-idf 计算相似"""
    # dictionary = corpora.Dictionary(words)
    # dictionary.save('../model_file/doc2bow.dict')
    dictionary = corpora.Dictionary.load('../model_file/doc2bow.dict')
    # corpus = [dictionary.doc2bow(text) for text in words]
    # tf_idf_model = TfidfModel(corpus, normalize=False)
    # tf_idf_model.save('../model_file/doc2bow.model')
    tf_idf_model = TfidfModel.load('../model_file/doc2bow.model')
    text_list = cut_word(text)

    query_text_corpus = dictionary.doc2bow(text_list)
    # 获取相似性矩阵
    # index = similarities.SparseMatrixSimilarity(tf_idf_model[corpus], num_features=len(dictionary.keys()))
    # index.save('../model_file/doc2bow.index')
    index = similarities.Similarity.load('../model_file/doc2bow.index')
    # 查询文本query_text与其他文本的tf-idf相似性
    score_list = list(index[tf_idf_model[query_text_corpus]])
    result = {}
    for i, score in enumerate(score_list):
        if score > threshold:
            result[i] = score

    print(result)

