import pandas as pd
import numpy as np

# 读取原始数据，假定为制表符分隔
df = pd.read_csv("rating_index.tsv", sep="\t")
print("原始数据形状:", df.shape)

# ----- 步骤1：添加噪声 -----
# 假定 rating 为 0/1 二分类，随机将 10% 的样本的 rating 进行翻转
noise_fraction = 0.1  # 噪声比例
if 'rating' in df.columns:
    num_rows = df.shape[0]
    noise_indices = np.random.choice(df.index, size=int(num_rows * noise_fraction), replace=False)
    # 对噪声样本，翻转 0->1 或 1->0
    df.loc[noise_indices, 'rating'] = df.loc[noise_indices, 'rating'].apply(lambda x: 1 - x if x in [0, 1] else x)

# ----- 步骤2：制造类别不平衡 -----
# 假定数据是二分类且 rating 列只有 0 和 1
if 'rating' in df.columns and set(df['rating'].unique()) == {0, 1}:
    count_class0 = df[df['rating'] == 0].shape[0]
    count_class1 = df[df['rating'] == 1].shape[0]
    print("原始类别数量：0类={}, 1类={}".format(count_class0, count_class1))

    # 对多数类进行欠采样（这里设置采样比例为 50%），制造不平衡
    if count_class0 > count_class1:
        df_majority = df[df['rating'] == 0].sample(frac=0.5, random_state=42)
        df_minority = df[df['rating'] == 1]
    else:
        df_majority = df[df['rating'] == 1].sample(frac=0.5, random_state=42)
        df_minority = df[df['rating'] == 0]
    df = pd.concat([df_majority, df_minority]).sample(frac=1, random_state=42)  # 打乱顺序
    print("不平衡后数据形状:", df.shape)

# ----- 步骤3：增加难样本（复制部分边缘样本）-----
# 随机复制 5% 的数据，并拼接到数据集中
df_extra = df.sample(frac=0.05, random_state=42)
df_modified = pd.concat([df, df_extra]).reset_index(drop=True)
print("增加难样本后数据形状:", df_modified.shape)

# 保存修改后的数据
df_modified.to_csv("rating_index.tsv", sep="\t", index=False)
print("已保存修改后的数据到 rating_index.tsv")
