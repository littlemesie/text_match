# -*- coding:utf-8 -*-

"""
@date: 2023/4/17 下午3:40
@summary: 向搭建好的 Milvus 系统插入向量
"""
import os
import sys
import numpy as np
from tqdm import tqdm
project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_path)
from utils.milvus_conn import MilvusConn

def vector_insert(all_embeds, params_config, collection_name, partition_tag):

    client = MilvusConn(**params_config)
    client.client.create_partition(collection_name, partition_tag, timeout=300)
    data_size = all_embeds.shape[0]
    embedding_ids = [i for i in range(data_size)]
    batch_size = 100000
    for i in tqdm(range(0, data_size, batch_size)):
        cur_end = i + batch_size
        if (cur_end > data_size):
            cur_end = data_size

        batch_emb = all_embeds[np.arange(i, cur_end)]
        status, ids = client.insert(collection_name=collection_name,
                                    vectors=batch_emb.tolist(),
                                    ids=embedding_ids[i:i + batch_size],
                                    partition_tag=partition_tag)

def search_result(client, embed, collection_name, partition_tag):

    status, ids = client.search(collection_name=collection_name,
                               vectors=embed.tolist(),
                               top_k=10,
                               partition_tag=partition_tag)

    return ids

if __name__ == "__main__":
    """"""
