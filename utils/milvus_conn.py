# -*- coding:utf-8 -*-

"""
@date: 2023/4/17 上午11:18
@summary: milvus1.1版本
"""
import logging
from pymilvus import MilvusClient, DataType, IndexType
from pymilvus.client.types import MetricType
from pymilvus import connections, db

class MilvusConn(object):
    """
    milvus1.1版本
    from milvus import *
    from milvus import MetricType, IndexType
    """

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

class MilvusConnV2:
    """
     milvus2.x版本
    """

    def __init__(self, uri, db_name='', dimension=768, field_name='vector', index_name='vector_index',
                 index_type='IVF_FLAT', metric_type='COSINE', nlist=128, nprobe=8, **kwargs):
        self.uri = uri
        self.db_name = db_name
        self.dimension = dimension
        self.field_name = field_name
        self.index_name = index_name
        self.nlist = nlist
        self.nprobe = nprobe
        self.index_type = index_type
        self.metric_type = metric_type

        self.client = MilvusClient(uri, db_name=self.db_name)


    def create_conn(self, db_name):
        """create conn"""
        conn = connections.connect(
            host=self.host,
            port=self.port,
            db_name=db_name
        )
        return conn

    def has_collection(self, collection_name):
        try:
            status = self.client.has_collection(collection_name)
            return status
        except Exception as e:
            print("Milvus has_table error:", e)
            raise e

    def creat_collection(self, collection_name):
        try:
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=True,
            )
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dimension)
            status = self.client.create_collection(collection_name=collection_name, schema=schema)
            logging.info(status)
            return status
        except Exception as e:
            print("Milvus create collection error:", e)
            raise e

    def create_index(self, collection_name):
        try:
            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(
                field_name=self.field_name,
                metric_type=self.metric_type,
                index_type=self.index_type,
                index_name=self.index_name,
                params={"nlist": self.nlist}
            )
            status = self.client.create_index(collection_name=collection_name, index_params=index_params)
            # logging.info(status)
            return status
        except Exception as e:
            print("Milvus create index error:", e)
            raise e

    def has_partition(self, collection_name, partition_name):
        try:
            status = self.client.has_partition(collection_name, partition_name)
            return status
        except Exception as e:
            print("Milvus has partition error: ", e)
            raise e

    def create_partition(self, collection_name, partition_name):
        try:
            status = self.client.create_partition(collection_name, partition_name)
            logging.info('create partition {} successfully'.format(partition_name))
            return status
        except Exception as e:
            print('Milvus create partition error: ', e)
            raise e

    def insert(self, data, collection_name, partition_name=None):
        try:
            if not self.has_collection(collection_name):
                self.creat_collection(collection_name)

                # self.create_index(collection_name)
                logging.info('collection info: {}'.format(
                    self.client.describe_collection(collection_name)))
            if (partition_name is not None) and (not self.has_partition(
                    collection_name, partition_name)):
                self.create_partition(collection_name, partition_name)

            res = self.client.insert(collection_name=collection_name,
                                     data=data,
                                     partition_name=partition_name)
            logging.info('Insert {} entities'.format(res['insert_count']))
            return res['insert_count']
        except Exception as e:
            print("Milvus insert error:", e)
            raise e

    def search(self, data, collection_name, partition_names=None, output_fields=None, top_k=10):
        try:
            search_params = {"metric_type": self.metric_type, "params": {'nprobe': self.nprobe}}
            res = self.client.search(
                collection_name=collection_name,
                data=data,
                limit=top_k,
                params=search_params,
                partition_names=partition_names,
                output_fields=output_fields
            )
            return res
        except Exception as e:
            print('Milvus recall error: ', e)
            raise e