import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
import sys
sys.path.append('../')

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
    now = 0
    for file_path in files_path:
        # print(file_path)
        # print(file_path.replace('outputs_logits', 'outputs_logits_plus'))
        if file_path.split('/')[-1] == 'overall_eval.json':
            continue
        outputs_list = []
        try:
            with open(file_path, "r") as json_file:
                for line in json_file:
                    # print(line)
                    outputs_list.append(json.loads(line))
        # print(len(outputs_logits_list))
        except json.decoder.JSONDecodeError:
            continue

        sample_list = []

        for outputs_logits in outputs_list:


            match_type = outputs_logits['if_match_now']
            tokens = outputs_logits['tokens']
            if len(tokens) > 80:
                continue
            # print(outputs_logits['correct'])
            if outputs_logits['correct'] is False:
                continue
            # print(len(match_type), len(tokens))
            tokens_prob = []
            # print(outputs_logits.keys())

            # print(outputs_logits['correct'])
            for i in range(len(outputs_logits['llm_logits_prob'])):

                tokens_ids = outputs_logits['llm_logits_prob'][i][1]
                logits = outputs_logits['slm_logits_prob'][i][0][0][:10]
                # if outputs_logits['if_match_now'][i] == 1:
                #     logits = outputs_logits['llm_logits_prob'][i][0][0][:10]
                # else:
                #     logits = outputs_logits['slm_logits_prob'][i][0][0][:10]
                tokens_prob.append(logits)
            # print(len(tokens_prob))
            visual_data = []
            for i in range(len(tokens_prob)):
                visual_data.append((tokens[i], tokens_prob[i], match_type[i]))
            sample_list.append(visual_data)
            # print(outputs_logits['inputs'])
            # print(outputs_logits['dataset'])
            # print(outputs_logits['llm_name'])
            # print(outputs_logits['slm_name'])
            # print(now)
            # print(visual_data)
            # print(outputs_logits['id'])

            data = visual_data
            num_tokens = len(data)
            cols = 12
            rows = (num_tokens + cols - 1) // cols 

            fig, axes = plt.subplots(rows, cols, figsize=(24, 12))

            for i, (token, token_prob, mismatch) in enumerate(data):
                row = i // cols
                col = i % cols
                ax = axes[row, col]
                color = 'red' if mismatch else 'blue'
                ax.bar(range(len(token_prob)), token_prob, color=color)
                ax.set_title(f'{token}', fontsize=24)
                # ax.set_ylabel('Probability', fontsize=10)
                # ax.set_xlabel('Index', fontsize=10)
                ax.set_ylim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])

            if num_tokens < rows * cols:
                for i in range(num_tokens, rows * cols):
                    fig.delaxes(axes.flatten()[i])

            plt.tight_layout()
            plt.savefig(f'fig5-{now}.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
            # plt.show()
            now += 1

