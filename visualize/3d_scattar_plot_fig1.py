import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from data import *
import sys
sys.path.append('../')


def plot_convex_hull(ax, data, color, label):
    x_str = [item[0] for item in data if isinstance(item[2], float)]
    y_str = [item[1] for item in data if isinstance(item[2], float)]
    z = [item[2] for item in data if isinstance(item[2], float)]

    x = [float(item.replace('B', '')) for item in x_str]
    y = [float(item.replace('B', '')) for item in y_str]

    x = np.log10(x)
    y = np.log10(y)

    x = x + np.random.normal(scale=1e-5, size=len(x))
    y = y + np.random.normal(scale=1e-5, size=len(y))
    z = z + np.random.normal(scale=1e-5, size=len(z))

    points = np.column_stack((x, y, z))

    hull = ConvexHull(points)

    ax.scatter(x, y, z, c=color, label=label)

    for simplex in hull.simplices:
        ax.plot_trisurf(points[simplex, 0], points[simplex, 1], points[simplex, 2], color=color, alpha=0.2)


fig = plt.figure(figsize=(18, 6))


datasets = [
    (oracle_gsm8k, oracle_MMLU_STEM, oracle_MBPP, oracle_MTBench, 'Oracle'),
    (contrastive_gsm8k, contrastive_MMLU_STEM, contrastive_MBPP, contrastive_MTBench, 'Contrastive'),
    (proxy_gsm8k, proxy_MMLU_STEM, proxy_MBPP, proxy_MTBench, 'Proxy')
]

for i, (gsm8k, MMLU_STEM, MBPP, MTBench, title) in enumerate(datasets):
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    plot_convex_hull(ax, gsm8k, 'b', 'GSM8k')
    plot_convex_hull(ax, MMLU_STEM, 'r', 'MMLU-STEM')
    plot_convex_hull(ax, MBPP, 'g', 'MBPP')
    # plot_convex_hull(ax, MTBench, 'm', 'MTBench')

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

plt.savefig('fig1.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
plt.show()
