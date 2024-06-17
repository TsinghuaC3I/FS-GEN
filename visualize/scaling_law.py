import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import sys
from data import *


# Data extraction function
def extract_data(data):
    LLM = []
    SLM = []
    y = []
    for larger, smaller, value in data:
        LLM_size = float(larger[:-1])
        SLM_size = float(smaller[:-1])
        ratio = LLM_size / SLM_size
        LLM.append(LLM_size)
        SLM.append(SLM_size)
        y.append(value)
    return LLM, SLM, y


# Power-law fit function
def power_law_func(x, a, alpha, b):
    return -a * x ** (-alpha) + b


# Preparing datasets
datasets = {
    'Oracle': {
        'GSM8k': oracle_gsm8k,
        'MMLU-STEM': oracle_MMLU_STEM,
        'MBPP': oracle_MBPP,
        'MTBench': oracle_MTBench
    },
    'Contrastive': {
        'GSMk': contrastive_gsm8k,
        'MMLU-STEM': contrastive_MMLU_STEM,
        'MBPP': contrastive_MBPP,
        'MTBench': contrastive_MTBench
    },
    'Proxy': {
        'GSM8k': proxy_gsm8k,
        'MMLU-STEM': proxy_MMLU_STEM,
        'MBPP': proxy_MBPP,
        'MTBench': proxy_MTBench
    }
}

# Define colors and markers
colors = ['b', 'r', 'g', 'm']
markers = ['o', 's', '^', 'd']
linestyles = ['-', '--', '-.', ':']

# Creating subplots
fig, axes = plt.subplots(1, 3, figsize=(24, 6))

# Plotting each subplot
for i, (title, data) in enumerate(datasets.items()):
    ax = axes[i]
    for j, (label, dataset) in enumerate(data.items()):
        if label == 'MTBench':
            continue
        LLM, SLM, y = extract_data(dataset)
        ratio = np.array(LLM) / np.array(SLM)
        try:
            # popt, _ = curve_fit(power_law_func, ratio, y, p0=[1.0, 0.5, 1.0], maxfev=5000)
            popt, _ = curve_fit(
                power_law_func, ratio, y, maxfev=2000,
                p0=[1.0, 1.0, 1.0],  # Initial guesses for a, alpha, b
                bounds=(0.05, np.inf)  # Ensure all parameters are positive
            )
            a, alpha, b = popt
            x_fit = np.linspace(min(ratio), max(ratio), 100)
            y_fit = power_law_func(x_fit, *popt)
            print(f"a = {round(a, 3)}", f"alpha={round(alpha, 3)}", f"b={round(b, 3)}")
            # Transforming x-axis
            x_transformed = ratio ** (-alpha)
            x_fit_transformed = x_fit ** (-alpha)

            # Plotting scatter points and fitted curves with enhanced styles
            ax.scatter(x_transformed, y, color=colors[j], marker=markers[j], s=100, alpha=0.7)
            formula_label = f'{label}: ${a:.2f} \cdot x^{{-{alpha:.2f}}} + {b:.2f}$'
            ax.plot(x_fit_transformed, y_fit, linestyle=linestyles[j], color=colors[j], linewidth=2,
                    label=formula_label)
        except RuntimeError as e:
            print(f"Fit for {label} in {title} failed: {e}")

    ax.set_xlabel(r'$\mathrm{Scale \ Ratio}^{-\alpha}$', fontsize=14)
    ax.set_ylabel('Collaboration Frequency', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='lower left')
    ax.grid(True)

plt.tight_layout()
plt.savefig('fig4.pdf', format='pdf')
plt.show()
