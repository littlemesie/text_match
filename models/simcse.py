# -*- coding:utf-8 -*-

"""
@date: 2022/9/14 下午3:02
@summary:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCSE(nn.Module):

    def __init__(self, pretrained_model, device, pooling="cls", scale=0.05):
        super(SimCSE, self).__init__()
        self.ptm = pretrained_model
        self.device = device
        self.pooling = pooling
        self.sacle = scale

    def get_pooled_embedding(self, input_ids, token_type_ids, attention_mask):
        """"""
        out = self.ptm(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
        if self.pooling == 'cls':
            return out.last_hidden_state[:, 0]  # [batch, 768]

        if self.pooling == 'pooler':
            return out.pooler_output  # [batch, 768]

        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]

        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]

    def unsup_loss(self, cls_embedding):
        """无监督loss"""
        cosine_sim = F.cosine_similarity(cls_embedding.unsqueeze(1), cls_embedding.unsqueeze(0), dim=-1)
        #  将相似度矩阵对角线置为很小的值, 消除自身的影响
        cosine_sim = cosine_sim - torch.eye(cls_embedding.shape[0]).to(self.device) * 1e12
        # label
        labels = torch.arange(cls_embedding.shape[0]).to(self.device)
        labels = (labels - labels % 2 * 2) + 1
        # 相似度矩阵除以温度系数
        cosine_sim = cosine_sim / self.sacle

        loss = F.cross_entropy(input=cosine_sim, target=labels)

        return torch.mean(loss)

    def forward(self, input_ids, token_type_ids, attention_mask):

        cls_embedding = self.get_pooled_embedding(input_ids, token_type_ids, attention_mask)

        return cls_embedding