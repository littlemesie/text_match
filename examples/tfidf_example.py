# -*- coding:utf-8 -*-

"""
@date: 2023/4/18 下午3:19
@summary: TFIDF例子
"""
import pandas as pd
from utils.cut_word_util import cut_word
from models.tfidf import tfidf_similarity

def process_data():
    df = pd.read_csv("../data/train.csv", error_bad_lines=False)
    df['words'] = df['text'].apply(lambda x: cut_word(x))
    df.to_csv("../data/process.csv", index=False)
    print(df)

def clac():
    import time

    df = pd.read_csv("../data/process.csv")
    df['words'] = df['words'].apply(lambda x: eval(x))
    text = "为什么黄码"
    t1 = time.time()
    words = df['words'].tolist()
    tfidf_similarity(words, text)
    print(time.time() - t1)

# process_data()
clac()