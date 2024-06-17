import json
import os

"""
outputs_dict = {
    'datasets': 'gsm8k_test',
    'id': i,
    'question': q,
    'gold_answer': a,
    'large_model': large_model_name,
    'small_model': small_model_name,
    'small_ft_model': small_ft_model_name,
    'outputs_text': outputs_text,
    'correct': test_answer_gsm8k_(outputs_text, a),
    'total_tokens_num': outputs['total_tokens_num'],
    'latent_tokens_num': outputs['latent_tokens_num'],
    'mismatch_tokens_num': outputs['mismatch_tokens_num'],
    'method_info': [router, collabrate]
}
"""

def read_json_files_in_outputs(dir_path):
    files_path = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                files_path.append(file_path)
    return files_path


def extract_key_info(key):
    parts = key.split('_')
    
    if parts[2] == 'delta':
        deleted_element = parts.pop(3)
        parts[2] = f'{parts[2]}_{deleted_element}'
    
    if parts[2] == 'normal':
        parts.insert(3, 1)
    
    parts.insert(5, 'few-shot(CoT)')
    # print(parts)
    large_model_size = float(parts[0].split('-')[1][:-1])
    small_model_size = float(parts[1].split('-')[1][:-1])
    
    decoding_method = parts[4]
    
    threshold_value = float(parts[3])
    
    return (large_model_size, small_model_size, decoding_method, threshold_value)


if __name__ == "__main__":
    dir_path = './outputs_logits_500'
    results  = {}
    for i in ['bbh', 'gsm8k', 'math', 'mmlu', 'mbpp', 'mtbench']:
        results.update(
           {i: { }}
        )
    files_path = read_json_files_in_outputs(dir_path)
    for file_path in files_path:
        # print(file_path)
        if file_path.split('/')[-1] == 'overall_eval.json':
            continue
        outputs_list = []
        print(file_path)
        with open(file_path, "r") as json_file:
            for line in json_file:
                # print(line)
                try:
                    outputs_list.append(json.loads(line))
                except json.decoder.JSONDecodeError:
                    print('Decode ERROR')
                    break
                if len(outputs_list) > 100:
                    break
        # print(outputs_list[0]['outputs_text'])
        for outputs in outputs_list:
            llm_name = outputs['llm_name'].split('/')[-1]
            slm_name = outputs['slm_name'].split('/')[-1]
            collabrate_method = outputs['method_info'][1]
            router_method = outputs['method_info'][0]
            
            router_str = ''
            if router_method['method'] == 'delta_threshold':
                router_str = f"{router_method['method']}_{str(router_method['threshold'])}"
            elif router_method['method'] == 'threshold':
                router_str = f"{router_method['method']}_{str(router_method['threshold'])}"
            elif router_method['method'] == 'svm':
                router_str = f"{router_method['method']}_{router_method['trained_model_path'].split('/')[-1]}"
            else:
                router_str = f"{router_method['method']}"
                
            if collabrate_method['method'] == 'ContrastiveDecoding':
                key = f"{llm_name}_{slm_name}_{router_str}_{collabrate_method['method']}_{str(collabrate_method['alpha'])}_{str(collabrate_method['beta'])}"
            elif collabrate_method['method'] == 'ProxyFineTuning':
                key = f"{llm_name}_{slm_name}_{router_str}_{collabrate_method['method']}"    
            elif collabrate_method['method'] == 'SpeculativeDecoding':
                key = f"{llm_name}_{slm_name}_{router_str}_{collabrate_method['method']}_{str(collabrate_method['K'])}"
            else: 
                key = f"{llm_name}_{slm_name}_{router_str}_{collabrate_method['method']}"  
            
            key = f"{key}_{file_path.split('_')[-1].split('.')[0]}"
            try: 
                if file_path.split('/')[-1].split('_')[0] == 'mtbench':
                    try:
                        if outputs['correct'] is None:
                            results[file_path.split('/')[-1].split('_')[0]][key]['acc'] += 0
                        else:
                            results[file_path.split('/')[-1].split('_')[0]][key]['acc'] += int(outputs['correct'][0])
                    except KeyError:
                        results[file_path.split('/')[-1].split('_')[0]][key]['acc'] += 0
                else:
                    try:
                        results[file_path.split('/')[-1].split('_')[0]][key]['acc'] += 1 if outputs['correct'] is True else 0
                    except KeyError:
                        results[file_path.split('/')[-1].split('_')[0]][key]['acc'] += 0
                
                results[file_path.split('/')[-1].split('_')[0]][key]['total_ans'] += 1
                results[file_path.split('/')[-1].split('_')[0]][key]['total_tokens_num'] += outputs['total_tokens_num']
                results[file_path.split('/')[-1].split('_')[0]][key]['latent_tokens_num'] += outputs['latent_tokens_num']
                results[file_path.split('/')[-1].split('_')[0]][key]['mismatch_tokens_num'] += outputs['mismatch_tokens_num']
            except Exception:
                if file_path.split('/')[-1].split('_')[0] == 'mtbench':
                    try:
                        if outputs['correct'] is None:
                            acc = 0
                        else:
                            acc = int(outputs['correct'][0])
                    except KeyError:
                        acc = 0
                else:
                    try:
                        acc = 1 if outputs['correct'] is True else 0
                    except KeyError:
                        acc = 0

                results[file_path.split('/')[-1].split('_')[0]].update(
                    {   
                        key: {
                        'acc': acc,
                        'total_ans': 1,
                        'total_tokens_num': outputs['total_tokens_num'], 
                        'latent_tokens_num': outputs['latent_tokens_num'], 
                        'mismatch_tokens_num': outputs['mismatch_tokens_num'],
                        'file_name': file_path.split('/')[-1]
                    }       
                    }            
                )
    for dataset, model_status in results.items():
        for status, result in model_status.items():
            if dataset == 'mtbench':
                results[dataset][status].update({'accuracy': round(result['acc'] / result['total_ans'], 3)})
            else:
                results[dataset][status].update({'accuracy': round(result['acc']*100 / result['total_ans'], 3)})
            results[dataset][status].update({'latent_ratio': round(result['latent_tokens_num'] / result['total_tokens_num'], 3)})
            results[dataset][status].update({'mismatch_ratio': round(result['mismatch_tokens_num'] / result['latent_tokens_num'], 3)})
    indented_results = json.dumps(results, indent=4)
    results_sorted = {}
    for dataset in results:
        sorted_keys = sorted(results[dataset].keys(), key=extract_key_info)
        results_sorted[dataset] = {key: results[dataset][key] for key in sorted_keys}
    with open(f'./outputs_logits_500/overall_eval.json', 'w') as f:
        json.dump(results_sorted, f, indent=4)
    print(indented_results)