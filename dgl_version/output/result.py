import pandas as pd

# ✅ 封装成一个函数，接受文件路径作为参数
def calculate_metrics_from_file(file_path):
    # 读取 Excel 文件
    df = pd.read_excel(file_path, header=None)
    df.columns = ['编号', '真实原因', '候选原因列表']

    # 将字符串形式的候选列表转为实际列表
    df['候选原因列表'] = df['候选原因列表'].apply(lambda x: eval(x) if isinstance(x, str) else x)

    # 命中率（Accuracy）
    df['是否命中'] = df.apply(lambda row: row['真实原因'] in row['候选原因列表'], axis=1)
    accuracy = df['是否命中'].mean()

    # Top-1 命中率（Hit@1）
    df['Top1命中'] = df.apply(lambda row: row['候选原因列表'][0] == row['真实原因'], axis=1)
    top1_acc = df['Top1命中'].mean()

    # Top-2 命中率（Hit@2）
    df['Top2命中'] = df.apply(lambda row: row['真实原因'] in row['候选原因列表'][:2], axis=1)
    top2_acc = df['Top2命中'].mean()

    # Top-3 命中率（Hit@3）
    df['Top3命中'] = df.apply(lambda row: row['真实原因'] in row['候选原因列表'][:3], axis=1)
    top3_acc = df['Top3命中'].mean()

    # 平均倒数排名（MRR）
    def reciprocal_rank(row):
        try:
            rank = row['候选原因列表'].index(row['真实原因']) + 1
            return 1.0 / rank
        except ValueError:
            return 0.0

    df['ReciprocalRank'] = df.apply(reciprocal_rank, axis=1)
    mrr_score = df['ReciprocalRank'].mean()

    return accuracy, top1_acc, top2_acc, top3_acc, mrr_score

# ✅ 添加 main 函数，自定义文件路径
def main(file_path):
    accuracy, top1_acc, top2_acc, top3_acc, mrr_score = calculate_metrics_from_file(file_path)

    # 输出结果（百分比格式）
    print(f"命中率 Hit@all: {accuracy * 100:.2f}%")
    print(f"Top-1 命中率 Hit@1: {top1_acc * 100:.2f}%")
    print(f"Top-2 命中率 Hit@2: {top2_acc * 100:.2f}%")
    print(f"Top-3 命中率 Hit@3: {top3_acc * 100:.2f}%")
    print(f"平均倒数排名 MRR: {mrr_score:.3f}")

if __name__ == "__main__":
    main("kg运行结果.xlsx")
    main("gnn运行结果-备份.xlsx")
    main("gnn-rag运行结果.xlsx")
