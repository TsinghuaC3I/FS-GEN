import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json
import matplotlib.cm as cm
import sys
sys.path.append('../')

coolwarm = cm.get_cmap('coolwarm')

def visualize_mismatch_positions_heatmap(data, output_file, bin_width=0.1):
    """
    Visualize the mismatch position of model combinations on different datasets.

    Parameters:
    data: list
        Each element is a list containing the dataset name, large model name, small model name, and a list of mismatch locations.
    output_file: str
        Path and name of the file where the chart is saved, e.g. 'output.pdf'.
    bin_width: float
        Width of each bin, defaults to 0.1 (i.e. 10%).

    """
    plot_data = []
    for entry in data:
        dataset, big_model, small_model, mismatch_positions = entry
        for pos in mismatch_positions:
            plot_data.append([f"{big_model}-{small_model.split('-')[-1]}", dataset, pos])
    df = pd.DataFrame(plot_data, columns=['model_combination', 'dataset', 'mismatch_position'])
    y_offsets = {comb: idx for idx, comb in enumerate(df['model_combination'].unique())}

    bins = np.arange(0, 1 + bin_width, bin_width)
    df['bin'] = pd.cut(df['mismatch_position'], bins=bins, include_lowest=True, labels=False)

    bin_counts = df.groupby(['dataset', 'model_combination', 'bin']).size().reset_index(name='count')
    total_counts = df.groupby(['dataset', 'model_combination']).size().reset_index(name='total')
    merged_counts = pd.merge(bin_counts, total_counts, on=['dataset', 'model_combination'])
    merged_counts['proportion'] = merged_counts['count'] / merged_counts['total']

    heatmap_data = {}
    for dataset in df['dataset'].unique():
        data_subset = merged_counts[merged_counts['dataset'] == dataset]
        pivot_data = data_subset.pivot(index='model_combination', columns='bin', values='proportion').fillna(0)
        heatmap_data[dataset] = pivot_data

    plt.rcParams.update({'font.size': 14})

    fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharex=True, sharey=True)
    axes = axes.flatten()

    first = True
    for ax, (dataset, data) in zip(axes, heatmap_data.items()):
        sns.heatmap(data, ax=ax, cmap='Blues', cbar=True, annot=False, linewidths=.5)
        ax.set_title(f'{dataset}', fontsize=16)
        ax.set_ylabel('Model Combination', fontsize=14)
        ax.set_xlabel('Mismatch Position', fontsize=14)
        ax.set_xticklabels([f'{int(bin_width*i*100)}-{int(bin_width*(i+1)*100)}%' for i in range(len(bins)-1)], rotation=45)
        ax.yaxis.set_tick_params(width=1)
        # ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
        if first:
            y_labels = list(y_offsets.keys())
            ax.set_yticks(list(y_offsets.values()))
            ax.set_yticklabels(y_labels, fontsize=12)
            first = False

    for ax in axes[len(heatmap_data):]:
        fig.delaxes(ax)

    plt.tight_layout()

    plt.savefig(output_file, format='pdf')

    plt.show()


def read_json_files_in_outputs(dir_path):
    files_path = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                files_path.append(file_path)
    return files_path


if __name__ == '__main__':
    files_path = read_json_files_in_outputs('outputs_logits')
    mismatch_position_list = []
    for file_path in files_path:
        # print(file_path)
        if file_path.split('/')[-1] == 'overall_eval.json':
            continue
        outputs_list = []
        with open(file_path, "r") as json_file:
            for line in json_file:
                # print(line)
                # try:
                outputs_list.append(json.loads(line))
                # except json.decoder.JSONDecodeError:
                #     break
        # print(len(outputs_logits_list))
        mismatch_position = []
        for outputs_logits in outputs_list:
            # print(outputs_logits)

            dataset = outputs_logits['dataset'].split('_')[0]
            if dataset == 'gsm8k':
                dataset = 'GSM8k'
            elif dataset == 'mmlu':
                dataset = 'MMLU-STEM'
            elif dataset == 'mbpp':
                dataset = 'MBPP'
            else:
                dataset = 'MTBench'
            llm = outputs_logits['llm_name'].replace('-Chat', '')
            slm = outputs_logits['slm_name'].replace('-Chat', '')
            for index, j in enumerate(outputs_logits['if_match_now']):
                if j == 1:
                    mismatch_position.append(index / len(outputs_logits['if_match_now']))
            # print(len(mismatch_position))
            # print(np.sum(outputs_logits['if_match_now']))
            # mismatch_position = (dataset, llm, slm, mismatch_position)
        mismatch_position_list.append([dataset, llm, slm, mismatch_position])
        # print(mismatch_position_list)
    # print(mismatch_position_list)
    visualize_mismatch_positions_heatmap(mismatch_position_list, 'fig3.pdf')