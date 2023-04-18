# -*- coding:utf-8 -*-

"""
@date: 2022/9/14 下午4:04
@summary:
"""
import os
import sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from milvus import MetricType, IndexType
from transformers.optimization import AdamW
from transformers import BertTokenizer, AlbertModel, BertConfig, AutoModel
import torch.nn.functional as F
from models.simcse import SimCSE
from models.dataloader import BertBatchTextDataset, TextDataset, convert_ids
from utils.generate_embed import GenerateEmbed
from utils.vector_util import vector_insert
from utils.milvus_conn import MilvusConn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def read_text(file_path):
    file = open(file_path)
    text_list = []
    for idx, line in enumerate(file.readlines()):
        if idx == 0:
            continue
        text_list.append(line.strip())
    return text_list

# 加载分词器
def load_tokenizer(model_path, special_token=None):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    if special_token:
        tokenizer.add_special_tokens(special_token)
    return tokenizer

def get_train_dataloader(tokenizer):
    dataset_batch = BertBatchTextDataset(tokenizer, max_len=128)
    train_dataset = TextDataset("../data/train.csv")
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                                  num_workers=2, collate_fn=dataset_batch)
    return train_dataloader

def eval(model, dataloader):
    """模型评估函数
    批量预测, 计算cos_sim, 转成numpy数组拼接起来, 一次性求spearman相关度
    """
    model.eval()
    sim_tensor = torch.tensor([]).to(device)
    label_array = np.array([])
    with torch.no_grad():
        for source, target, label in dataloader:
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source['input_ids'].squeeze(1).to(device)
            source_attention_mask = source['attention_mask'].squeeze(1).to(device)
            source_token_type_ids = source['token_type_ids'].squeeze(1).to(device)
            source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)
            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target['input_ids'].squeeze(1).to(device)
            target_attention_mask = target['attention_mask'].squeeze(1).to(device)
            target_token_type_ids = target['token_type_ids'].squeeze(1).to(device)
            target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids)
            # concat
            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            label_array = np.append(label_array, np.array(label))

    return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation


def train():
    """"""
    tokenizer_path = "../lib/albert_chinese_base"
    tokenizer = load_tokenizer(tokenizer_path)
    get_train_dataloader(tokenizer)

    train_dataloader = get_train_dataloader(tokenizer)
    valid_dataloader = ''
    config = BertConfig.from_pretrained(tokenizer_path)
    pretrained_model = AlbertModel.from_pretrained(tokenizer_path, config)

    model = SimCSE(pretrained_model, device)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    model.train()
    best = 0.5
    best_loss = 0.0001
    save_model_path = "../model_file/simcse_best.bin"
    early_stop_batch = 0
    for epoch in range(10):
        tqdm_bar = tqdm(train_dataloader, desc="training epoch{epoch}".format(epoch=epoch))
        loss_total, loss_diffs = [], []
        for batch_idx, (token, segment, mask) in enumerate(tqdm_bar):
            token = token.to(device)
            segment = segment.to(device)
            mask = mask.to(device)
            model.zero_grad()
            out = model(token, segment, mask)
            loss = model.unsup_loss(out)
            loss.backward()
            optimizer.step()

            loss_total.append(loss.detach().item())
            if batch_idx != 0:
                loss_diffs.append(abs(loss.detach().item() - loss_total[-2]))
            # 评估 通过正负样本评估
            # if batch_idx % 10 == 0:
            #     corrcoef = eval(model, valid_dataloader)
            #     model.train()
            #     if best < corrcoef:
            #         early_stop_batch = 0
            #         best = corrcoef
            #         torch.save(model.state_dict(), save_model_path)
            #         continue
            #     early_stop_batch += 1
            #     if early_stop_batch == 10:
            #         return
            # 通过loss 评估保存模型
            if batch_idx != 0 and batch_idx % 20 == 0:
                mean_diff_loss = np.mean(loss_diffs[-10:])
                print(loss.detach().item(), mean_diff_loss)
                if best_loss > mean_diff_loss:
                    early_stop_batch = 0
                    best_loss = mean_diff_loss
                    torch.save(model.state_dict(), save_model_path)
                    continue
                early_stop_batch += 1
                if early_stop_batch == 20:
                    torch.save(model.state_dict(), save_model_path)
                    return
def predict():
    """"""
    file_path = "../data/train.csv"
    tokenizer_path = "../lib/albert_chinese_base"
    save_model_path = "../model_file/simcse_best.bin"
    text_list = read_text(file_path)

    tokenizer = load_tokenizer(tokenizer_path)
    config = BertConfig.from_pretrained(tokenizer_path)
    pretrained_model = AlbertModel.from_pretrained(tokenizer_path, config)

    model = SimCSE(pretrained_model, device)
    model.to(device)
    model.load_state_dict(torch.load(save_model_path, map_location=torch.device('cpu')))
    model.eval()
    # ge = GenerateEmbed(model, tokenizer, device=device, max_seq_length=128, batch_size=32)
    # all_embeds = ge.generate_many(text_list)
    # np.save(f"../model_file/corpus_embedding", all_embeds)
    all_embeds = np.load("../model_file/corpus_embedding.npy")
    params_config = {
        "host": "",
        "port": "19530",
        "dimension": 768,
        "index_file_size": 20,
        "metric_type": MetricType.L2,
        "index_type": IndexType.IVF_FLAT,
        "nlist": 1000,
        "nprobe": 20,
    }
    collection_name = "test1"
    partition_tag = 'partition_1'
    # vector_insert(all_embeds, params_config, collection_name, partition_tag)
    client = MilvusConn(**params_config)
    text = "健康码"
    with torch.no_grad():
        input_ids, attention_mask, token_type_ids = convert_ids(text, tokenizer, max_seq_length=128)
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        embed = model(input_ids, token_type_ids, attention_mask)
        # embed = np.array(embed)
        status, ids = client.search(collection_name=collection_name,
                                    vectors=embed.tolist(),
                                    top_k=10,
                                    partition_tag=partition_tag)

        print(ids)


# train()
predict()
