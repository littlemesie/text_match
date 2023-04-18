# -*- coding:utf-8 -*-

"""
@date: 2023/4/18 下午2:54
@summary: 分词
"""
import jieba

def load_punctuation():
    """加载标点符号"""
    punctuation = []
    try:
        with open(f"../lib/stopwords/punctuation.txt", 'r', encoding='utf-8') as f:
            for line in f.readlines():
                punctuation.append(line.strip('\n'))
    except FileNotFoundError:
        raise "Please check if the file exists！"

    punctuation = list(set(punctuation))
    return punctuation

punctuation = load_punctuation()


def cut_word(text, filter_punctuation=True):
    """分词"""
    seg_list = jieba.cut(text,  cut_all=False)
    if filter_punctuation:
        words = [seg for seg in seg_list if seg not in punctuation]
    else:
        words = [seg for seg in seg_list]
    return words
