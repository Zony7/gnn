import matplotlib.pyplot as plt
from pylab import mpl

# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 统一设置字体大小
mpl.rcParams['font.size'] = 12  # 设置全局字体大小

# 抽取绘图函数
def plot_metrics(seq_labels, x_pos, precision, recall, f1_score, title, xlabel, ylabel, yrange, filename):
    # 绘图
    plt.figure(figsize=(8, 5))
    plt.plot(x_pos, precision, marker='o', label='Precision')
    plt.plot(x_pos, recall, marker='s', label='Recall')
    plt.plot(x_pos, f1_score, marker='^', label='F1-score')

    # 样式设置
    plt.xticks(x_pos, seq_labels)        # 显示原始标签
    plt.xlim(-1, len(x_pos))
    plt.title(title, fontsize=12)  # 设置图表标题
    plt.xlabel(xlabel, fontsize=12)                                # 设置X轴标签
    plt.ylabel(ylabel)                                                 # 设置Y轴标签
    plt.ylim(yrange) # 设置Y轴范围
    plt.grid(True, alpha=0.5)
    plt.legend()                                                            # 显示图例
    plt.tight_layout()                                                      # 自动调整子图参数，使之填充整个图像区域

    # 保存图像
    plt.savefig(filename, dpi=300)

    # 显示图像
    plt.show()

# 原始数据
seq_labels = ['32', '64', '128', '256']
x_pos = [0, 1, 2, 3]  # 等间隔坐标
precision = [84.04, 88.29, 88.12, 88.09]
recall =    [88.41, 88.33, 88.14, 88.10]
f1_score =  [86.17, 88.31, 88.13, 88.09]

# 调用绘图函数
plot_metrics(seq_labels, x_pos, precision, recall, f1_score, 
             title='最大序列长度对BERT-BiLSTM-CRF的影响', 
             xlabel='最大序列长度', 
             ylabel='百分比 (%)', 
             yrange=(83, 90), 
             filename='bert_1.png')