import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体为支持CJK字符的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或其他支持CJK的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def plot_metrics(labels, precision, recall, f1_score, title, xlabel, ylabel, output_filename):
    # x轴位置
    x = np.arange(len(labels))*1.5  # x轴位置
    width = 0.25  # 柱状图宽度

    # 创建图形和子图
    fig, ax = plt.subplots(figsize=(16, 8))

    # 绘制Precision、Recall和F1-score的柱状图
    rects1 = ax.bar(x - width, precision, width, label='Precision (%)', color='skyblue')
    rects2 = ax.bar(x, recall, width, label='Recall (%)', color='orange')
    rects3 = ax.bar(x + width, f1_score, width, label='F1-score (%)', color='green')

    # 添加标签、标题和图例
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x)  # 设置x轴刻度位置
    ax.set_xticklabels(labels)  # 设置x轴标签为标注类型
    ax.legend(loc='upper center',ncol=3,  fontsize=12)
    # ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)
    # 设置纵轴的固定高度为110%
    ax.set_xlim([x[0]-1, x[-1]+1])
    ax.set_ylim(0, 110)

    # 在柱状图上添加数值标签
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',  # 显示2位小数
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 偏移量
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)

    # 调整布局并显示图形
    fig.tight_layout()
    # 保存图像
    plt.savefig(output_filename, dpi=300)

    plt.show()

if __name__ == '__main__':
    '''
    不同标注类型的BERT-BiLSTM-CRF模型故障实体识别结果
    '''
    # labels = ['B-FP', 'I-FP', 'B-FL', 'I-FL', 'B-FT', 'I-FT', 'B-FC', 'I-FC', 'B-RS', 'I-RS', 'O']
    # precision = [88.32, 85.95, 93.83, 91.26, 78.43, 86.36, 89.36, 88.14, 87.12, 87.80, 90.11]
    # recall = [96.99, 53.77, 25.42, 43.62, 14.76, 46.91, 54.31, 96.81, 52.32, 95.75, 91.82]
    # f1_score = [92.46, 66.15, 40.00, 59.03, 24.84, 60.80, 67.56, 92.27, 65.38, 91.60, 90.96]
    # plot_metrics(labels, precision, recall, f1_score, '不同标注类型的BERT-BiLSTM-CRF模型故障实体识别结果', '标注类型', '百分比 (%)', "pictures/bert_labels.png")

    '''
    最大序列长度对BERT-BiLSTM-CRF的影响
    '''
    labels = ['32', '64', '128', '256']
    precision = [84.04, 88.29, 88.12, 88.09]
    recall = [88.41, 88.33, 88.14, 88.10]
    f1_score = [86.17, 88.31, 88.13, 88.09]
    plot_metrics(labels, precision, recall, f1_score, '最大序列长度对BERT-BiLSTM-CRF的影响', '最大序列长度', '百分比 (%)', "pictures/bert_max_seq_len.png")