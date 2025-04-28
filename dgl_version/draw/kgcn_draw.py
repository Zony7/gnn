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
    ax.set_ylim(0, 110)
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
    5-6
    '''
    labels = ['5', '10', '15']
    precision = [85.27, 89.36, 86.02]
    recall = [84.13, 88.42, 87.90]
    f1_score = [84.70, 88.89, 86.95]

    plot_metrics_separated(
        labels, precision, recall, f1_score,
        title='邻居采样数量对KGCN模型的影响',
        xlabel='邻居采样数量',
        ylabel='百分比 (%)',
        output_filename='pictures/5-6.png'
    )

    '''
    5-7
    '''
    labels = ['Neighbor', 'Sum', 'Concat']
    precision = [84.11, 89.36, 91.26]
    recall = [88.24, 88.42, 88.68]
    f1_score = [86.12, 88.89, 89.95]

    plot_metrics_separated(
        labels, precision, recall, f1_score,
        title='聚合方法对KGCN模型的影响',
        xlabel='聚合方法',
        ylabel='百分比 (%)',
        output_filename='pictures/5-7.png'
    )

    '''
    5-8
    '''
    labels = ['1e-3', '1e-2', '1e-1']
    precision = [91.84, 91.26, 88.24]
    recall = [87.38, 88.68, 85.71]
    f1_score = [89.55, 89.95, 86.98]

    plot_metrics_separated(
        labels, precision, recall, f1_score,
        title='学习率对KGCN模型的影响',
        xlabel='学习率',
        ylabel='百分比 (%)',
        output_filename='pictures/5-8.png'
    )

    '''
    5-9
    '''
    labels = ['0.3', '0.4', '0.5', '0.6', '0.7']
    precision = [84.21, 88.66, 91.26, 92.71, 96.34]
    recall = [95.05, 94.51, 88.68, 85.58, 76.70]
    f1_score = [89.30, 91.49, 89.95, 89.00, 85.41]

    plot_metrics_separated(
        labels, precision, recall, f1_score,
        title='预测函数阈值对KGCN模型的影响',
        xlabel='预测函数阈值',
        ylabel='百分比 (%)',
        output_filename='pictures/5-9.png'
    )


