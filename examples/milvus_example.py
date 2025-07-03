# -*- coding:utf-8 -*-

"""
@date: 2024/10/21 下午6:56
@summary: 基于milvus的验证
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from pymilvus import connections, db
from utils.milvus_conn import MilvusConnV2


def create_db(host, port, db_name):
    """create database"""
    conn = connections.connect(host=host, port=port)
    database = db.create_database(db_name)

def vector_insert():
    """插入"""
    df = pd.read_csv("/home/mesie/python/aia-nlp-service/data/fuzhou_jj.csv")
    print(df.shape)
    # print(df.loc[0, 'text'])
    all_embeds = np.load("//media/mesie/F0E66F06E66ECC82/数据/福州/fuzhou_jj.npy")
    print(all_embeds.shape)
    db_name = "aia"
    # 创建db
    # create_db(host='10.115.6.210', port='19530', db_name=db_name)
    collection_name = "fuzhou_jj"
    partition_name = 'partition_1'
    params_config = {
        'uri': 'http://10.115.6.210:19530',
        'db_name': db_name,
        'dimension': 1024,
        'field_name': 'vector',
        'index_name': 'vector_index',
        'metric_type': 'COSINE',
        "nlist": 256,
        "nprobe": 8,
    }
    client = MilvusConnV2(**params_config)
    # client.create_partition(collection_name, partition_name)
    data_size = all_embeds.shape[0]

    batch_size = 100
    for i in tqdm(range(0, data_size, batch_size)):
        cur_end = i + batch_size
        if (cur_end > data_size):
            cur_end = data_size
        ids = np.arange(i, cur_end)
        batch_data = []
        for id in ids:
            batch_data.append({'id': id, 'vector': all_embeds[id], 'text': df.loc[id, 'text']})
        # print(batch_data)
        res = client.insert(data=batch_data, collection_name=collection_name, partition_name=partition_name)
        print(res)


def search_vector():
    all_embeds = np.load("//media/mesie/F0E66F06E66ECC82/数据/福州/fuzhou_jj.npy")
    print(all_embeds.shape)
    db_name = "aia"
    collection_name = "fuzhou_jj"
    partition_name = 'partition_1'
    params_config = {
        'uri': 'http://10.115.6.210:19530',
        'db_name': db_name,
        'dimension': 1024,
        'field_name': 'vector',
        'index_name': 'vector_index',
        'metric_type': 'COSINE',
        "nlist": 256,
        "nprobe": 8,
    }
    client = MilvusConnV2(**params_config)
    data = [all_embeds[0]]
    import time
    t1 = time.time()
    res = client.search(data=data, collection_name=collection_name, partition_names=[partition_name],
                        output_fields=['text'], top_k=10)
    print(time.time() - t1)
    print(res)


if __name__ == '__main__':
    """"""
    # vector_insert()
    search_vector()
