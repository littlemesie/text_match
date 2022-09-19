# -*- coding:utf-8 -*-

"""
@date: 2022/9/14 下午3:02
@summary:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCSE(nn.Module):

    def __init__(self,
                 pretrained_model,
                 dropout=None,
                 margin=0.0,
                 scale=20):

        super().__init__()

        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)

        self.margin = margin
        # Used scaling cosine similarity to ease converge
        self.sacle = scale

    def get_pooled_embedding(self,
                             input_ids,
                             token_type_ids=None,
                             position_ids=None,
                             attention_mask=None,
                             with_pooler=True):

        # Note: cls_embedding is poolerd embedding with act tanh
        sequence_output, cls_embedding = self.ptm(input_ids, token_type_ids,
                                                  position_ids, attention_mask)

        if with_pooler == False:
            cls_embedding = sequence_output[:, 0, :]

        cls_embedding = self.dropout(cls_embedding)
        cls_embedding = F.normalize(cls_embedding, p=2, dim=-1)

        return cls_embedding

    def get_semantic_embedding(self, data_loader):
        self.eval()
        with torch.no_grad():
            for batch_data in data_loader:
                input_ids, token_type_ids = batch_data
                text_embeddings = self.get_pooled_embedding(
                    input_ids, token_type_ids=token_type_ids)

                yield text_embeddings

    def cosine_sim(self,
                   query_input_ids,
                   title_input_ids,
                   query_token_type_ids=None,
                   query_position_ids=None,
                   query_attention_mask=None,
                   title_token_type_ids=None,
                   title_position_ids=None,
                   title_attention_mask=None,
                   with_pooler=True):

        query_cls_embedding = self.get_pooled_embedding(query_input_ids,
                                                        query_token_type_ids,
                                                        query_position_ids,
                                                        query_attention_mask,
                                                        with_pooler=with_pooler)

        title_cls_embedding = self.get_pooled_embedding(title_input_ids,
                                                        title_token_type_ids,
                                                        title_position_ids,
                                                        title_attention_mask,
                                                        with_pooler=with_pooler)

        cosine_sim = torch.sum(query_cls_embedding * title_cls_embedding, dim=-1)
        return cosine_sim

    def forward(self,
                query_input_ids,
                title_input_ids,
                query_token_type_ids=None,
                query_position_ids=None,
                query_attention_mask=None,
                title_token_type_ids=None,
                title_position_ids=None,
                title_attention_mask=None):

        query_cls_embedding = self.get_pooled_embedding(query_input_ids,
                                                        query_token_type_ids,
                                                        query_position_ids,
                                                        query_attention_mask)

        title_cls_embedding = self.get_pooled_embedding(title_input_ids,
                                                        title_token_type_ids,
                                                        title_position_ids,
                                                        title_attention_mask)
        #
        cosine_sim = torch.matmul(query_cls_embedding, title_cls_embedding)

        # substract margin from all positive samples cosine_sim()
        margin_diag = torch.full(size=[query_cls_embedding.shape[0]], fill_value=self.margin)

        cosine_sim = cosine_sim - torch.diag(margin_diag)

        # scale cosine to ease training converge
        cosine_sim *= self.sacle

        labels = torch.arange(0, query_cls_embedding.shape[0], dtype=torch.int64)
        labels = torch.reshape(labels, shape=[-1, 1])

        loss = F.cross_entropy(input=cosine_sim, target=labels)

        return loss