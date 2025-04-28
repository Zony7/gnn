import matplotlib.pyplot as plt
import numpy as np

# 字体配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_metrics_separated(labels, precision, recall, f1_score, title, xlabel, ylabel, output_filename):
    N = len(labels)
    base_x = np.arange(N) * 3.0  # 每组之间留更大空间

    # 三个柱子的位置往左右扩一点
    x_p = base_x - 0.6
    x_r = base_x
    x_f = base_x + 0.6
    width = 0.5

    fig, ax = plt.subplots(figsize=(16, 8))
    rects1 = ax.bar(x_p, precision, width, label='Precision (%)', color='skyblue')
    rects2 = ax.bar(x_r, recall, width, label='Recall (%)', color='orange')
    rects3 = ax.bar(x_f, f1_score, width, label='F1-score (%)', color='green')

    # 设置X轴
    ax.set_xticks(base_x)
    ax.set_xticklabels(labels, fontfamily='monospace', fontsize=16)
    ax.set_xlim(base_x[0] - 2, base_x[-1] + 2)
    ax.set_ylim(0, 105)
    ax.tick_params(axis='y', labelsize=16)  # 设置y轴字体大小为14

    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(title, fontsize=20)

    # 横向图例
    ax.legend(loc='upper center', ncol=3, fontsize=18)

    # 添加数值标签
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=15)

    add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.show()

if __name__ == '__main__':
    '''
    最大序列长度对BERT-BiLSTM-CRF的影响
    '''
    labels = ['32', '64', '128', '256']
    precision = [84.04, 88.29, 88.12, 88.09]
    recall = [88.41, 88.33, 88.14, 88.10]
    f1_score = [86.17, 88.31, 88.13, 88.09]
    plot_metrics_separated(
        labels, precision, recall, f1_score,
        title='最大序列长度对BERT-BiLSTM-CRF的影响',
        xlabel='最大序列长度',
        ylabel='百分比 (%)',
        output_filename='pictures/bert_1.png'
    )

    '''
    5-2
    '''
    labels = ['1e-5', '5e-5', '1e-4', '5e-4']
    precision = [88.29, 88.30, 88.24, 51.03]
    recall =    [88.33, 88.35, 88.29, 50.67]
    f1_score =  [88.31, 88.32, 88.27, 50.85]

    plot_metrics_separated(
        labels, precision, recall, f1_score,
        title='BERT层学习率对BERT-BiLSTM-CRF的影响',
        xlabel='BERT层学习率',
        ylabel='百分比 (%)',
        output_filename='pictures/bert_2.png'
    )

    '''
    5-3
    '''
    labels = ['1e-4', '1e-3', '1e-2', '1e-1']
    precision = [88.29, 88.30, 88.29, 80.32]
    recall = [88.34, 88.35, 88.34, 82.33]
    f1_score = [88.31, 88.32, 88.31, 81.32]

    plot_metrics_separated(
        labels, precision, recall, f1_score,
        title='下层学习率对BERT-BiLSTM-CRF的影响',
        xlabel='下层学习率',
        ylabel='百分比 (%)',
        output_filename='pictures/bert_3.png'
    )

    '''
    5-4
    '''
    labels = ['8', '16', '32', '64']
    precision = [88.30, 88.30, 88.28, 86.22]
    recall = [88.35, 88.35, 88.33, 87.61]
    f1_score = [88.32, 88.32, 88.31, 86.91]

    plot_metrics_separated(
        labels, precision, recall, f1_score,
        title='批处理大小对BERT-BiLSTM-CRF的影响',
        xlabel='批处理大小',
        ylabel='百分比 (%)',
        output_filename='pictures/bert_4.png'
    )

    '''
    5-5
    '''
    labels = ['B-FP', 'I-FP', 'B-FL', 'I-FL', 'B-FT', 'I-FT', 'B-FC', 'I-FC', 'B-RS', 'I-RS', 'O']
    precision = [88.32, 85.95, 93.83, 91.26, 78.43, 86.36, 89.36, 88.14, 87.12, 87.80, 90.11]
    recall = [96.99, 53.77, 25.42, 43.62, 14.76, 46.91, 54.31, 96.81, 52.32, 95.75, 91.82]
    f1_score = [92.46, 66.15, 40.00, 59.03, 24.84, 60.80, 67.56, 92.27, 65.38, 91.60, 90.96]
    plot_metrics_separated(
        labels, precision, recall, f1_score,
        title='不同标注类型的BERT-BiLSTM-CRF模型故障实体识别结果',
        xlabel='标注类型',
        ylabel='百分比 (%)',
        output_filename='pictures/bert_5.png'
    )