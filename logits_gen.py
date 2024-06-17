from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.conversation import Conversation, SeparatorStyle
import numpy as np
import torch.nn.functional as F
import json
import torch
from sklearn import svm
from sklearn.metrics import accuracy_score
import joblib 
import sys
import argparse
sys.path.append("./FSGen")
from router import StoppingCriteriaList, KeyWordsCriteria
import openai
import re
import time


from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset, load_from_disk
import random

from rich.console import Console
from rich.table import Table
import requests
from copy import deepcopy

import pandas as pd
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.conversation import Conversation, SeparatorStyle
from fsgen_opensource_logits import FSGenOpenSource
from utils import *
from conversation import CONVS, generate_inputs
from func_timeout import func_timeout
from func_timeout import FunctionTimedOut
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'

flash_attn = True

    
def run_dump(dataset, collabrate, router, large_model_path, small_model_path, small_ft_model_path, current_time, sampling, random_state):

    if flash_attn:
        large_model = AutoModelForCausalLM.from_pretrained(f"{large_model_path}", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto", trust_remote_code=True).eval()
        small_model = AutoModelForCausalLM.from_pretrained(f"{small_model_path}", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto", trust_remote_code=True).eval()
        if small_ft_model_path is None:
            small_ft_model = None
        else:
            small_ft_model = AutoModelForCausalLM.from_pretrained(f"{small_ft_model_path}", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto", trust_remote_code=True).eval()
    else:
        large_model = AutoModelForCausalLM.from_pretrained(f"{large_model_path}", device_map="auto", trust_remote_code=True).eval()
        small_model = AutoModelForCausalLM.from_pretrained(f"{small_model_path}", device_map="auto", trust_remote_code=True).eval()
        if small_ft_model_path is None:
            small_ft_model = None
        else:
            small_ft_model = AutoModelForCausalLM.from_pretrained(f"{small_ft_model_path}", device_map="auto", trust_remote_code=True).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(f"{small_model_path}", trust_remote_code=True)
    tokenizer.eos_token_id = tokenizer("<|endoftext|>").input_ids[0]
    tokenizer.pad_token_id = tokenizer.eos_token_id
    fsgen = FSGenOpenSource(large_model, small_model, tokenizer, small_ft_model)
    if dataset == 'gsm8k':
        gsm8k = load_from_disk('./data/gsm8k')
        gsm8k_test = gsm8k['test']
        prompt_complex = open('./lib_prompts/gsm8k_prompt.txt').read()
        i = 0
        # print(gsm8k_test)
        gsm8k_test_sample = gsm8k_test.shuffle(seed=random_state).select(range(sampling))
        with open(f'./outputs_logits/{dataset}_{current_time}.json', 'a+') as f:
            for q, a in tqdm(zip(gsm8k_test_sample['question'], gsm8k_test_sample['answer']), total=len(gsm8k_test_sample['question'])):
                prompt_q = prompt_complex + '\nQuestion: ' + q + "\nLet's think step by step\n" 
                conv = CONVS[large_model_path.split('/')[-1].split('-')[0]].copy()
                inputs = generate_inputs(conv, prompt_q, tokenizer)
                stop_words_ids = [[151645], [151644], [tokenizer('Question:')['input_ids'][0]]]
                with torch.no_grad():
                    slm_outputs = fsgen.generate_text(
                        input_ids=inputs.to(large_model.device), 
                        max_tokens=512, 
                        collabrate_method=collabrate, 
                        router_method={'method': 'none'}, 
                        temperature=0.7, 
                        stop_word_ids=stop_words_ids
                    )
                    outputs = fsgen.generate_text(
                        input_ids=inputs.to(large_model.device), 
                        max_tokens=512, 
                        collabrate_method=collabrate, 
                        router_method=router, 
                        temperature=0.7, 
                        stop_word_ids=stop_words_ids
                    )
                SaveData = {
                    'id': i,
                    'dataset': dataset,
                    'inputs': '\nQuestion: ' + q + "\nLet's think step by step\n",
                    'correct': test_answer_gsm8k_(outputs['text'], a),
                    'llm_name': large_model_path,
                    'slm_name': small_model_path,
                    'text': outputs['text'], 
                    'slm_text': slm_outputs['text'],
                    'slm_text_logits_prob': slm_outputs['slm_logits_prob'],
                    'topK': outputs['topK'],
                    'slm_logits_prob': outputs['slm_logits_prob'],
                    'llm_logits_prob': outputs['llm_logits_prob'],
                    'if_match_now': outputs['if_match_now'],
                    'total_tokens_num': outputs['total_tokens_num'],
                    'latent_tokens_num': outputs['latent_tokens_num'],
                    'mismatch_tokens_num': outputs['mismatch_tokens_num'],
                    'tokens': outputs['tokens'],
                    'method_info': [router, collabrate]
                }
                json.dump(SaveData, f)
                f.write('\n')
                i += 1
    elif dataset == 'mmlu':
        mmlu_prompt = json.load(open('./lib_prompts/mmlu-cot.json'))
        mmlu_stem = load_from_disk("./data/MMLU-STEM")['train']
        mmlu_stem_test = mmlu_stem.shuffle(seed=random_state).select(range(sampling))
        i = 0
        with open(f'./outputs_logits/{dataset}_{current_time}.json', 'a+') as f:
            for question, choices, a_index, task in tqdm(zip(mmlu_stem_test['question'], mmlu_stem_test['choices'], mmlu_stem_test['answer'], mmlu_stem_test['subject']), total=len(mmlu_stem_test)):
                q = 'Q: ' + question + '\n'
                for j, letter in enumerate(['A', 'B', 'C', 'D']):
                    q += '(' + letter + ') ' + choices[j] + ' '
                a = ['A', 'B', 'C', 'D'][a_index]
                prompt_q = mmlu_prompt[task] + "\n\n" + q + "\nA: Let's think step by step."  
                conv = CONVS[large_model_path.split('/')[-1].split('-')[0]].copy()
                conv.set_system_message("You will write beautiful compliments according to needs")
                conv.append_message("<|im_start|>user", prompt_q)
                # conv.append_message("<|im_start|>user", "My colleague works diligently")
                conv.append_message("<|im_start|>assistant", None)
                inputs = tokenizer(
                    conv.get_prompt(),
                    return_tensors='pt'
                )["input_ids"]
                stop_words_ids = [[151645], [151644], [14582]]

                with torch.no_grad():
                    slm_outputs = fsgen.generate_text(
                        input_ids=inputs.to(large_model.device), 
                        max_tokens=512, 
                        collabrate_method=collabrate, 
                        router_method={'method': 'none'}, 
                        temperature=0.7, 
                        stop_word_ids=stop_words_ids
                    )
                    outputs = fsgen.generate_text(
                        input_ids=inputs.to(large_model.device), 
                        max_tokens=512, 
                        collabrate_method=collabrate, 
                        router_method=router, 
                        temperature=0.7, 
                        stop_word_ids=stop_words_ids
                    )
                SaveData = {
                    'id': i,
                    'dataset': f'{dataset}_{task}',
                    'inputs': q + "\nA: Let's think step by step.",
                    'correct': test_answer_mmlu_(outputs['text'], a),
                    'llm_name': large_model_path,
                    'slm_name': small_model_path,
                    'text': outputs['text'], 
                    'slm_text': slm_outputs['text'],
                    'slm_text_logits_prob': slm_outputs['slm_logits_prob'],
                    'topK': outputs['topK'],
                    'slm_logits_prob': outputs['slm_logits_prob'],
                    'llm_logits_prob': outputs['llm_logits_prob'],
                    'if_match_now': outputs['if_match_now'],
                    'total_tokens_num': outputs['total_tokens_num'],
                    'latent_tokens_num': outputs['latent_tokens_num'],
                    'mismatch_tokens_num': outputs['mismatch_tokens_num'],
                    'tokens': outputs['tokens'],
                    'method_info': [router, collabrate]
                }
                json.dump(SaveData, f)
                f.write('\n')
                i += 1
    elif dataset == 'mtbench':
        mtbench = df = jsonl_to_dataframe('./data/MTbench_question.jsonl')
        i = 0
        mtbench_sample = mtbench.sample(n=sampling, random_state=random_state)
        with open(f'./outputs_logits/{dataset}_{current_time}.json', 'a+') as f:
            for question_id, category, turns, reference in tqdm(zip(mtbench_sample['question_id'], mtbench_sample['category'], mtbench_sample['turns'], mtbench_sample['reference']), total=len(mtbench_sample['question_id'])):
                outputs_turns = []
                conv = CONVS[large_model_path.split('/')[-1].split('-')[0]].copy()
                conv.set_system_message("You will write beautiful compliments according to needs")
                for j in range(len(turns)):
                    conv.append_message("<|im_start|>user", turns[j])
                    conv.append_message("<|im_start|>assistant", None)

                    inputs = tokenizer(
                        conv.get_prompt(),
                        return_tensors='pt'
                    )["input_ids"]
                    # print(conv.get_prompt())
                    # stop_words_ids = [[tokenizer.im_end_id], [tokenizer.im_start_id]]
                    # TODO:对于预训练模型，需要在这里重新定义stop_ids，否则会一直生成，比如Question:，是14582
                    stop_words_ids = [[151645], [151644], [14582]]

                    with torch.no_grad():
                        slm_outputs = fsgen.generate_text(
                            input_ids=inputs.to(large_model.device), 
                            max_tokens=512, 
                            collabrate_method=collabrate, 
                            router_method={'method': 'none'}, 
                            temperature=0.7, 
                            stop_word_ids=stop_words_ids
                        )
                        outputs = fsgen.generate_text(
                            input_ids=inputs.to(large_model.device), 
                            max_tokens=512, 
                            collabrate_method=collabrate, 
                            router_method=router, 
                            temperature=0.7, 
                            stop_word_ids=stop_words_ids
                        )
                    SaveData = {
                        'id': f"{i}_{j}",
                        'dataset': f"{dataset}_{category}",
                        'inputs': turns[j],
                        'correct': None,
                        'llm_name': large_model_path,
                        'slm_name': small_model_path,
                        'text': outputs['text'], 
                        'slm_text': slm_outputs['text'],
                        'slm_text_logits_prob': slm_outputs['slm_logits_prob'],
                        'topK': outputs['topK'],
                        'slm_logits_prob': outputs['slm_logits_prob'],
                        'llm_logits_prob': outputs['llm_logits_prob'],
                        'if_match_now': outputs['if_match_now'],
                        'total_tokens_num': outputs['total_tokens_num'],
                        'latent_tokens_num': outputs['latent_tokens_num'],
                        'mismatch_tokens_num': outputs['mismatch_tokens_num'],
                        'tokens': outputs['tokens'],
                        'method_info': [router, collabrate]
                    }
                    json.dump(SaveData, f)
                    f.write('\n')
                    outputs_text = outputs['text']
                    conv.update_last_message(outputs_text)
                    outputs_turns.append(outputs_text)
                i += 1        
    else:
        mbpp = df = jsonl_to_dataframe('./data/mbpp.jsonl')
        mbpp_test = mbpp[10:510]
        mbpp_prompt = mbpp[0:10]
        mbpp_val = mbpp[510:600]
        mbpp_train = mbpp[600:974]
        prompt_complex = np.load('./lib_prompts/prompt_mbpp_10_shot.npy')
        mbpp_test_sample = mbpp_test.sample(n=sampling, random_state=random_state)
        i = 0
        with open(f'./outputs_logits/{dataset}_{current_time}.json', 'a+') as f:
            for text, test_list, task_id, code in tqdm(zip(mbpp_test_sample['text'], mbpp_test_sample['test_list'], mbpp_test_sample['task_id'], mbpp_test_sample['code']), total=len(mbpp_test_sample['text'])):
                # print(i)
                test_codes = generate_test_codes(test_list)
                prompt_q = str(prompt_complex) + MBPP_gen_templates.format(task_id=task_id, text=text, test_codes=test_codes)
                conv = CONVS[large_model_path.split('/')[-1].split('-')[0]].copy()
                conv.set_system_message("You will write beautiful compliments according to needs")
                conv.append_message("<|im_start|>user", prompt_q)
                conv.append_message("<|im_start|>assistant", None)
                inputs = tokenizer(
                    conv.get_prompt(),
                    return_tensors='pt'
                )["input_ids"]
                # print(conv.get_prompt())
                # stop_words_ids = [[tokenizer.im_end_id], [tokenizer.im_start_id]]

                stop_words_ids = [[151645], [151644], [14582]]

                with torch.no_grad():
                    slm_outputs = fsgen.generate_text(
                        input_ids=inputs.to(large_model.device), 
                        max_tokens=512, 
                        collabrate_method=collabrate, 
                        router_method={'method': 'none'}, 
                        temperature=0.7, 
                        stop_word_ids=stop_words_ids
                    )
                    outputs = fsgen.generate_text(
                        input_ids=inputs.to(large_model.device), 
                        max_tokens=512, 
                        collabrate_method=collabrate, 
                        router_method=router, 
                        temperature=0.7, 
                        stop_word_ids=stop_words_ids
                    )
                try:
                    test_ans = func_timeout(3, test_answer_mbpp_, args=(outputs['text'], test_list, ))
                    # test_ans = test_answer_mbpp_(outputs_text, test_list)
                except FunctionTimedOut as e:
                    test_ans = ['TimeOut', False]
                SaveData = {
                    'dataset': dataset,
                    'inputs': MBPP_gen_templates.format(task_id=task_id, text=text, test_codes=test_codes),
                    'correct': test_ans,
                    'llm_name': large_model_path,
                    'slm_name': small_model_path,
                    'text': outputs['text'], 
                    'slm_text': slm_outputs['text'],
                    'slm_text_logits_prob': slm_outputs['slm_logits_prob'],
                    'topK': outputs['topK'],
                    'slm_logits_prob': outputs['slm_logits_prob'],
                    'llm_logits_prob': outputs['llm_logits_prob'],
                    'if_match_now': outputs['if_match_now'],
                    'total_tokens_num': outputs['total_tokens_num'],
                    'latent_tokens_num': outputs['latent_tokens_num'],
                    'mismatch_tokens_num': outputs['mismatch_tokens_num'],
                    'tokens': outputs['tokens'],
                    'method_info': [router, collabrate]
                }
                json.dump(SaveData, f)
                f.write('\n')
                i += 1
    large_model.to('cpu')
    small_model.to('cpu')
    del large_model
    del small_model
    torch.cuda.empty_cache()

def get_args():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Script to run models with specific configurations')
    parser.add_argument('--router', type=str, default='normal', help='Router method to use, such as none, normal, threhold, delta_threshold and svm')
    parser.add_argument('--method', type=str, default='OracleDecoding', help='Method to use, such as ContrastiveDecoding, SpeculativeDecoding, ProxyFineTuning and OracleDecoding')
    parser.add_argument('--sampling', type=int, default=500, help='Number of samples')
    parser.add_argument('--dataset', type=str, default='gsm8k', help='Dataset to use, including gsm8k, mmlu, mbpp and mtbench')
    parser.add_argument('--large-model-path', type=str, default='', help='The absoluted path of your large model')
    parser.add_argument('--small-model-path', type=str, default='', help='The absoluted path of your small model')
    parser.add_argument('--small-ft-model-path', type=str, default=None, help='None, or the absoluted path of your small fintuned model when you not use Proxy/Emulator Tuning')
    
    # Parse arguments
    return parser.parse_args()


def main():
    args = get_args()
    # task_list = [
    #     ('mbpp', 1.8, 0.5), ('mbpp', 4, 0.5), ('mbpp', 4, 1.8), ('mbpp', 7, 0.5), ('mbpp', 7, 1.8), ('mbpp', 7, 4), ('mbpp', 14, 0.5), ('mbpp', 14, 1.8),
    #     ('mbpp', 14, 4), ('mbpp', 14, 7), ('mbpp', 32, 0.5), ('mbpp', 32, 1.8), ('mbpp', 32, 4), ('mbpp', 32, 7), ('mbpp', 32, 14),
    #     ('mbpp', 72, 0.5), ('mbpp', 72, 1.8), ('mbpp', 72, 4), ('mbpp', 72, 7), ('mbpp', 72, 14), ('mbpp', 72, 32),
    #     ('mmlu', 1.8, 0.5), ('mmlu', 4, 0.5), ('mmlu', 4, 1.8), ('mmlu', 7, 0.5), ('mmlu', 7, 1.8), ('mmlu', 7, 4), ('mmlu', 14, 0.5), ('mmlu', 14, 1.8),
    #     ('mmlu', 14, 4), ('mmlu', 14, 7), ('mmlu', 32, 0.5), ('mmlu', 32, 1.8), ('mmlu', 32, 4), ('mmlu', 32, 7), ('mmlu', 32, 14),
    #     ('mmlu', 72, 0.5), ('mmlu', 72, 1.8), ('mmlu', 72, 4), ('mmlu', 72, 7), ('mmlu', 72, 14), ('mmlu', 72, 32),
    #     ('gsm8k', 1.8, 0.5), ('gsm8k', 4, 0.5), ('gsm8k', 4, 1.8), ('gsm8k', 7, 0.5), ('gsm8k', 7, 1.8), ('gsm8k', 7, 4), ('gsm8k', 14, 0.5), ('gsm8k', 14, 1.8),
    #     ('gsm8k', 14, 4), ('gsm8k', 14, 7), ('gsm8k', 32, 0.5), ('gsm8k', 32, 1.8), ('gsm8k', 32, 4), ('gsm8k', 32, 7), ('gsm8k', 32, 14),
    #     ('gsm8k', 72, 0.5), ('gsm8k', 72, 1.8), ('gsm8k', 72, 4), ('gsm8k', 72, 7), ('gsm8k', 72, 14), ('gsm8k', 72, 32)
    # ]

    sampling = args.sampling
    if args.dataset == 'gsm8k':
        random_state = 789798
    else:
        random_state = 42
    current_time = time.strftime("%Y%m%d%H%M", time.localtime())
    collabrate = collabrate_method[args.method]
    router = router_method[args.router]
    run_dump(args.dataset, collabrate, router, args.large_model_path, args.small_model_path, args.small_ft_model_path, current_time, args.sampling, random_state)



if __name__ == "__main__":
    main()
    