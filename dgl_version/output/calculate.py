import numpy as np
import pandas as pd


# ✅ 在原函数中添加 NDCG@K 的计算逻辑
def calculate_metrics_with_ndcg(file_path, ks=[1, 2, 3, 5]):
    # 读取 Excel 文件
    df = pd.read_excel(file_path, header=None)
    df.columns = ['编号', '真实原因', '候选原因列表']

    # 将字符串形式的候选列表转为实际列表
    df['候选原因列表'] = df['候选原因列表'].apply(lambda x: eval(x) if isinstance(x, str) else x)

    # 命中率（是否命中任意一个候选）
    df['是否命中'] = df.apply(lambda row: row['真实原因'] in row['候选原因列表'], axis=1)
    accuracy = df['是否命中'].mean()

    # Top-K 命中率
    hit_at_k = {}
    for k in ks:
        df[f'Top{k}命中'] = df.apply(lambda row: row['真实原因'] in row['候选原因列表'][:k], axis=1)
        hit_at_k[k] = df[f'Top{k}命中'].mean()

    # MRR
    def reciprocal_rank(row):
        try:
            rank = row['候选原因列表'].index(row['真实原因']) + 1
            return 1.0 / rank
        except ValueError:
            return 0.0
    df['ReciprocalRank'] = df.apply(reciprocal_rank, axis=1)
    mrr_score = df['ReciprocalRank'].mean()

    # NDCG@K
    def ndcg_at_k(row, k):
        rels = [1 if x == row['真实原因'] else 0 for x in row['候选原因列表'][:k]]
        dcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(rels)])
        idcg = 1.0  # 只有一个相关项，理想 DCG 就是 1 / log2(1+1)
        return dcg / idcg if idcg > 0 else 0.0

    ndcg_at_k_scores = {k: df.apply(lambda row: ndcg_at_k(row, k), axis=1).mean() for k in ks}

    return accuracy, hit_at_k, mrr_score, ndcg_at_k_scores
# 更直观地输出评估结果（整齐格式化）
def print_metrics(model_name, accuracy, hit_at_k, mrr_score, ndcg_at_k):
    print(f"\n📌 模型名称：{model_name}")
    print("-" * 40)
    print(f"✅ Hit@all（命中率）   : {accuracy * 100:.2f}%")
    print(f"✅ MRR（平均倒数排名） : {mrr_score:.4f}")
    print("\n🎯 Top-K 命中率（Hit@K）:")
    for k, v in hit_at_k.items():
        print(f"  - Hit@{k:<2}: {v * 100:.2f}%")
    print("\n📈 排序质量（NDCG@K）:")
    for k, v in ndcg_at_k.items():
        print(f"  - NDCG@{k:<2}: {v:.4f}")




# 示例调用
# 使用函数打印 GNN 模型结果

# 分别计算 KGCN 和 KGCN-RAG 模型的指标
kg_result = calculate_metrics_with_ndcg("kg运行结果.xlsx")
kgcn_result = calculate_metrics_with_ndcg("gnn运行结果-备份.xlsx")
kgcn_rag_result = calculate_metrics_with_ndcg("gnn-rag运行结果.xlsx")

# 打印结果
print_metrics("传统基于规则模型", *kg_result)
print_metrics("KGCN 模型", *kgcn_result)
print_metrics("KGCN-RAG 模型", *kgcn_rag_result)
