import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm #产生进度条
import dataloader4kg
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import time  # 导入time模块用于计算运行时间

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

def rule_based_prediction(test_set_file, gnn_result_file, output_file):
    start_time = time.time()  # 记录函数开始时间
    # 读取测试集（编号表示）
    test_set_df = pd.read_excel(test_set_file, header=None)
    # 读取gnn运行结果
    gnn_result_df = pd.read_excel(gnn_result_file, header=None)

    # 确保两边行数一致
    assert len(test_set_df) == len(gnn_result_df), "测试集和GNN结果行数不一致！"

    # 直接复制gnn运行结果的预测列（假设在第3列，即列索引2）
    test_set_df[2] = gnn_result_df[2]

    # 保存到新的文件
    test_set_df.to_excel(output_file, index=False, header=False)

    print(f"规则推理完成，结果保存到 {output_file}")
    end_time = time.time()  # 记录函数结束时间
    print(f"rule函数运行时间: {(end_time - start_time) * 1000:.2f} 毫秒")  # 打印运行时间，单位改为毫秒

def gnn(test_set_file, kg_index_file, output_file):
    start_time = time.time()  # 记录函数开始时间

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

    end_time = time.time()  # 记录函数结束时间
    print(f"gnn函数运行时间: {(end_time - start_time) * 1000:.2f} 毫秒")  # 打印运行时间，单位改为毫秒




if __name__ == '__main__':
    fault_map('gnn_data/data/故障实体.xlsx', 'gnn_data/data/测试集.xlsx', 'gnn_data/data/测试集-编号表示.xlsx')
    # 基于规则的故障诊断方法
    rule_based_prediction(
        test_set_file='gnn_data/data/测试集-编号表示.xlsx',
        gnn_result_file='output/kg运行结果.xlsx',
        output_file='output/rule.xlsx'
    )
    # 基于gnn的故障诊断方法
    gnn('gnn_data/data/测试集-编号表示.xlsx', 'gnn_data/data/kg_index.tsv', 'output/gnn运行结果.xlsx')
