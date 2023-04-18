# -*- coding:utf-8 -*-

"""
@date: 2023/4/17 上午11:18
@summary:
"""
import logging
from milvus import *
from milvus import MetricType, IndexType

class MilvusConn(object):

    def __init__(self, host, port, dimension=256, index_file_size=256, metric_type=MetricType.L2,
                 index_type=IndexType.IVF_FLAT, nlist=1000, nprobe=20, **kwargs):
        self.client = Milvus(host=host, port=port)

        self.collection_param = {
            'dimension': dimension,
            'index_file_size': index_file_size,
            'metric_type': metric_type
        }
        self.index_type = index_type
        self.index_param = {'nlist': nlist}
        self.search_param = {'nprobe': nprobe}

    def has_collection(self, collection_name):
        try:
            status, ok = self.client.has_collection(collection_name)
            return ok
        except Exception as e:
            print("Milvus has_table error:", e)
            raise e

    def creat_collection(self, collection_name):
        try:
            self.collection_param['collection_name'] = collection_name
            status = self.client.create_collection(self.collection_param)
            logging.info(status)
            return status
        except Exception as e:
            print("Milvus create collection error:", e)
            raise e

    def create_index(self, collection_name):
        try:
            status = self.client.create_index(collection_name, self.index_type,
                                              self.index_param)
            logging.info(status)
            return status
        except Exception as e:
            print("Milvus create index error:", e)
            raise e

    def has_partition(self, collection_name, partition_tag):
        try:
            status, ok = self.client.has_partition(collection_name,
                                                   partition_tag)
            return ok
        except Exception as e:
            print("Milvus has partition error: ", e)
            raise e

    def create_partition(self, collection_name, partition_tag):
        try:
            status = self.client.create_partition(collection_name,
                                                  partition_tag)
            logging.info('create partition {} successfully'.format(partition_tag))
            return status
        except Exception as e:
            print('Milvus create partition error: ', e)
            raise e

    def insert(self, vectors, collection_name, ids=None, partition_tag=None):
        try:
            if not self.has_collection(collection_name):
                self.creat_collection(collection_name)
                self.create_index(collection_name)
                logging.info('collection info: {}'.format(
                    self.client.get_collection_info(collection_name)[1]))
            if (partition_tag is not None) and (not self.has_partition(
                    collection_name, partition_tag)):
                self.create_partition(collection_name, partition_tag)
            status, ids = self.client.insert(collection_name=collection_name,
                                             records=vectors,
                                             ids=ids,
                                             partition_tag=partition_tag)
            self.client.flush([collection_name])
            logging.info(
                'Insert {} entities, there are {} entities after insert data.'.
                format(len(ids),
                       self.client.count_entities(collection_name)[1]))
            return status, ids
        except Exception as e:
            print("Milvus insert error:", e)
            raise e

    def search(self, vectors, collection_name, top_k=20, partition_tag=None):
        try:
            status, results = self.client.search(
                collection_name=collection_name,
                query_records=vectors,
                top_k=top_k,
                params=self.search_param,
                partition_tag=partition_tag)
            return status, results
        except Exception as e:
            print('Milvus recall error: ', e)
            raise e