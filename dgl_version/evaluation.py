import ast

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm #产生进度条
import dataloader4kg
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

from dgl_version.KGCN import KGCN, predict


def fault_map(entity_file, fault_test_file, output_file):
    # 读取故障实体.xlsx文件
    entity_df = pd.read_excel(entity_file)
    # 读取FaultKnowledge.xlsx文件
    fault_knowledge_df = pd.read_excel(fault_test_file)
    # 创建一个映射字典，将故障实体.xlsx的第一列映射到第二列
    mapping_dict = dict(zip(entity_df.iloc[:, 0], entity_df.iloc[:, 1]))
    # 根据映射字典替换FaultKnowledge.xlsx的第一列
    fault_knowledge_df.iloc[:, 0] = fault_knowledge_df.iloc[:, 0].map(mapping_dict)
    # 根据映射字典替换FaultKnowledge.xlsx的第二列
    fault_knowledge_df.iloc[:, 1] = fault_knowledge_df.iloc[:, 1].map(mapping_dict)
    # 设置第一行第一列为0
    fault_knowledge_df.iloc[0, 0] = 0
    fault_knowledge_df.iloc[0, 1] = 341
    fault_knowledge_df.to_excel(output_file,  index=False, header=False)

def gnn(test_set_file, kg_index_file, output_file):
    n_neighbors = 10
    n_layers = 2  # 设置消息传递次数

    users, items, train_set, test_set = dataloader4kg.readRecData(dataloader4kg.Ml_100K.RATING)
    entitys, relations, kgTriples = dataloader4kg.readKgData(dataloader4kg.Ml_100K.KG)
    adj_kg = dataloader4kg.construct_kg(kgTriples)
    adj_entity, adj_relation = dataloader4kg.construct_adj(n_neighbors, adj_kg, len(entitys))
    # 加载训练好的模型
    model = KGCN( n_users = max( users ) + 1, n_entitys = max( entitys ) + 1,
                  n_relations = max( relations ) + 1, e_dim = 10,
                  adj_entity = adj_entity, adj_relation = adj_relation,
                  n_neighbors = n_neighbors, aggregator_method = 'sum',
                  act_method = F.relu, drop_rate = 0.5, n_layers=n_layers )
    model.load_state_dict(torch.load('model/KGCN_model.pth'))
    model.eval()

    # 读取测试集-编号表示.xlsx文件
    test_set_df = pd.read_excel(test_set_file, header=None)

    # 读取kg_index.tsv文件
    kg_index_df = pd.read_csv(kg_index_file, sep='\t', header=None)

    predictions = []
    pre_causes = []

    # 创建一个字典来存储kg_index.tsv中的映射关系
    kg_index_map = {}
    for index, row in kg_index_df.iterrows():
        if len(row) >= 3:  # 确保每行至少有三列
            fault_id = row[0]  # 确保fault_id是整数
            cause_id = row[2]
            if fault_id not in kg_index_map:
                kg_index_map[fault_id] = []
            kg_index_map[fault_id].append(cause_id)


    # 遍历测试集-编号表示.xlsx文件的每一行
    for index, row in test_set_df.iterrows():
        if len(row) >= 1:  # 确保每行至少有一列
            test_fault_id = row[0]  # 确保test_fault_id是整数
            fault_ids  = []
            cause_ids  = []
            pre_cause = []
            if test_fault_id in kg_index_map:
                for cause_id in kg_index_map[test_fault_id]:
                    fault_ids.append(test_fault_id)
                    cause_ids.append(cause_id)
            prediction = predict(model, fault_ids, cause_ids)

            for i, pred in enumerate(prediction):
                if pred == 1:
                    pre_cause.append(cause_ids[i])
                else:
                    pre_cause.append(-1)
            predictions.append(prediction)
            pre_causes.append(pre_cause)

    test_set_df[2] = pd.Series(pre_causes).apply(lambda x: [int(i) for i in x])
    print(predictions)
    test_set_df.to_excel(output_file, index=False, header=False)


def evaluate_model(result_file):
    result_df = pd.read_excel(result_file, header=None)
    result_len = len(result_df)
    true_len = 0

    for index, row in result_df.iterrows():
        if index == 0:
            print(f"第一行真实值: {row[1]} (类型: {type(row[1])})")
            print(f"第一行预测值: {row[2]} (类型: {type(row[2])})")

        # 尝试将 row[2] 从字符串解析为列表
        try:
            pred_list = ast.literal_eval(row[2])
        except (ValueError, SyntaxError):
            pred_list = []

        # 确保 pred_list 是一个列表并且不为空
        if isinstance(pred_list, list) and len(pred_list) > 0:
            # 打印调试信息
            # print(
            #     f"行 {index} 真实值: {row[1]} (类型: {type(row[1])}), 预测值列表: {pred_list} (类型: {type(pred_list)}), 第一个元素: {pred_list[0]} (类型: {type(pred_list[0])})")

            # 比较 row[1] 和 pred_list 列表中的第一个元素
            if len(pred_list) == 1 and row[1] == pred_list[0]:
                true_len += 1
        else:
            # 如果 pred_list 不是列表或为空，直接比较
            # print(f"行 {index} 真实值: {row[1]} (类型: {type(row[1])}), 预测值: {row[2]} (类型: {type(row[2])})")

            if row[1] == row[2]:
                true_len += 1

    print("hit@1：", true_len / result_len)


def evaluate_model_rag(result_file):
    result_df = pd.read_excel(result_file, header=None)
    result_len = len(result_df)
    true_len = 0

    for index, row in result_df.iterrows():
        if index == 0:
            print(f"第一行真实值: {row[1]} (类型: {type(row[1])})")
            print(f"第一行预测值: {row[2]} (类型: {type(row[2])})")

        # 尝试将 row[2] 从字符串解析为列表
        try:
            pred_list = ast.literal_eval(row[2])
        except (ValueError, SyntaxError):
            pred_list = []

        # 确保 pred_list 是一个列表并且不为空
        if isinstance(pred_list, list) and len(pred_list) > 0:
            # 打印调试信息
            # print(
            #     f"行 {index} 真实值: {row[1]} (类型: {type(row[1])}), 预测值列表: {pred_list} (类型: {type(pred_list)}), 第一个元素: {pred_list[0]} (类型: {type(pred_list[0])})")

            # 比较 row[1] 和 pred_list 列表中的第一个元素
            if row[1] == pred_list[0]:
                true_len += 1
        else:
            # 如果 pred_list 不是列表或为空，直接比较
            # print(f"行 {index} 真实值: {row[1]} (类型: {type(row[1])}), 预测值: {row[2]} (类型: {type(row[2])})")

            if row[1] == row[2]:
                true_len += 1

    print("hit@1：", true_len / result_len)



if __name__ == '__main__':
    # fault_map('gnn_data/data/故障实体.xlsx', 'gnn_data/data/测试集.xlsx', 'gnn_data/data/测试集-编号表示.xlsx')
    # gnn('gnn_data/data/测试集-编号表示.xlsx', 'gnn_data/data/kg_index.tsv', 'output/gnn运行结果.xlsx')
    print("gnn运行结果")
    evaluate_model('output/gnn运行结果.xlsx')
    print("gnn-rag运行结果")
    evaluate_model_rag('output/gnn-rag运行结果.xlsx')