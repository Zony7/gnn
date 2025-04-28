import matplotlib.pyplot as plt
import numpy as np


# 设置全局字体为支持CJK字符的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或其他支持CJK的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 数据定义
labels = ['B-FP', 'I-FP', 'B-FL', 'I-FL', 'B-FT', 'I-FT', 'B-FC', 'I-FC', 'B-RS', 'I-RS', 'O']
precision = [88.32, 85.95, 93.83, 91.26, 78.43, 86.36, 89.36, 88.14, 87.12, 87.80, 90.11]
recall = [96.99, 53.77, 25.42, 43.62, 14.76, 46.91, 54.31, 96.81, 52.32, 95.75, 91.82]
f1_score = [92.46, 66.15, 40.00, 59.03, 24.84, 60.80, 67.56, 92.27, 65.38, 91.60, 90.96]

x = np.arange(len(labels))  # x轴位置
width = 0.25  # 柱状图宽度

# 创建图形和子图
fig, ax = plt.subplots(figsize=(14, 8))

# 绘制Precision、Recall和F1-score的柱状图
rects1 = ax.bar(x - width, precision, width, label='Precision (%)', color='skyblue')
rects2 = ax.bar(x, recall, width, label='Recall (%)', color='orange')
rects3 = ax.bar(x + width, f1_score, width, label='F1-score (%)', color='green')

# 添加标签、标题和图例
ax.set_xlabel('标注类型', fontsize=14)
ax.set_ylabel('百分比 (%)', fontsize=14)
ax.set_title('不同标注类型的BERT-BiLSTM-CRF模型故障实体识别结果', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(labels)  # 设置x轴标签为标注类型
ax.legend(loc='upper center', fontsize=12)
# ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)
# 设置纵轴的固定高度为110%
ax.set_ylim(0, 110)

# 在柱状图上添加数值标签
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',  # 显示一位小数
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 偏移量
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

add_labels(rects1)
add_labels(rects2)
add_labels(rects3)

# 调整布局并显示图形
fig.tight_layout()
plt.show()