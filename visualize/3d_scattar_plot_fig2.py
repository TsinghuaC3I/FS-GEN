import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from data import *


def plot_3d_scatter(ax, data, color, label):
    x_str = [item[0] for item in data if isinstance(item[2], float)]
    y_str = [item[1] for item in data if isinstance(item[2], float)]
    z = [item[2] for item in data if isinstance(item[2], float)]

    x = [float(item.replace('B', '')) for item in x_str]
    y = [float(item.replace('B', '')) for item in y_str]

    x = np.log10(x)
    y = np.log10(y)

    x = x + x * np.random.uniform(-0.05, 0.05, len(x))
    y = y + y * np.random.uniform(-0.05, 0.05, len(y))

    ax.scatter(x, y, z, c=color, label=label)
    for i in range(len(x)):
        ax.plot([x[i], x[i]], [y[i], y[i]], [0, z[i]], color)

fig = plt.figure(figsize=(18, 6))

datasets = [
    (oracle_gsm8k, oracle_MMLU_STEM, oracle_MBPP, oracle_MTBench, 'Oracle'),
    (contrastive_gsm8k, contrastive_MMLU_STEM, contrastive_MBPP, contrastive_MTBench, 'Contrastive'),
    (proxy_gsm8k, proxy_MMLU_STEM, proxy_MBPP, proxy_MTBench, 'Proxy')
]

# 创建子图并绘制数据
for i, (gsm8k, MMLU_STEM, MBPP, MTBench, title) in enumerate(datasets):
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    plot_3d_scatter(ax, gsm8k, 'b', 'GSM8k')
    plot_3d_scatter(ax, MMLU_STEM, 'r', 'MMLU-STEM')
    plot_3d_scatter(ax, MBPP, 'g', 'MBPP')
    # plot_3d_scatter(ax, MTBench, 'm', 'MTBench')

    x_range = np.log10([0.5, 1.8, 4, 7, 14, 32, 72])
    y_range = np.log10([0.5, 1.8, 4, 7, 14, 32, 72])
    x, y = np.meshgrid(x_range, y_range)

    z_values = [0.05, 0.10, 0.20]
    for z_value in z_values:
        z_plane = np.full_like(x, z_value)
        ax.plot_surface(x, y, z_plane, alpha=0.3, color='gray')

    ax.set_xlabel('Large Language Model')
    ax.set_ylabel('Small Language Model')
    ax.set_zlabel('Collaboration Frequency')
    ax.set_title(title)

    ax.set_xticks(np.log10([0.5, 1.8, 4, 7, 14, 32, 72]))
    ax.set_xticklabels(['0.5B', '1.8B', '4B', '7B', '14B', '32B', '72B'])
    ax.set_yticks(np.log10([0.5, 1.8, 4, 7, 14, 32, 72]))
    ax.set_yticklabels(['0.5B', '1.8B', '4B', '7B', '14B', '32B', '72B'])

    ax.legend()
    ax.view_init(elev=20, azim=145)


plt.tight_layout()

plt.savefig('fig2.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
plt.show()
