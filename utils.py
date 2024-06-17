import re
import json
import numpy as np
import pandas as pd
import sys
from llms_oai import LLMs
import math



router_method = {
    'threshold': {
        'method': 'threshold',
        'threshold': 0.5
    },
    'delta_threshold': {
        'method': 'delta_threshold',
        'threshold': 0.2
    },
    'svm': {
        'method': 'svm',
        'svm_router_train_file': 'logits.json',
        'trained_model_path': './svm_router_top1000.pkl',
        'topK': 1000
    },
    'normal': {
        'method': 'normal'
    },
    'none': {
        'method': 'none'
    }
}

collabrate_method = {
    'ContrastiveDecoding': {
        'method': 'ContrastiveDecoding',
        'alpha': 0.1,
        'beta': 0.5
    },
    'SpeculativeDecoding': {
        'method': 'SpeculativeDecoding',
        'K': 5
    },
    'EmulatorFineTuning': {
        'method': 'EmulatorFineTuning',
        'alpha': 1.0
    },
    'OracleDecoding': {
        'method': 'OracleDecoding'
    } 
}


def test_answer_gsm8k_(pred_str, ans_str):
    pattern = '\d*\.?\d+'
    pred = re.findall(pattern, pred_str)
    if(len(pred) >= 1):
        # print(pred_str)
        pred = pred[-1]
        gold = re.findall(pattern, ans_str)
        # print(ans_str)
        gold = gold[-1]
        return pred == gold
    else: return False

MMLU_TASKS = [
    'abstract_algebra',
    'anatomy',
    'astronomy',
    'business_ethics',
    'clinical_knowledge',
    'college_biology',
    'college_chemistry',
    'college_computer_science',
    'college_mathematics',
    'college_medicine',
    'college_physics',
    'computer_security',
    'conceptual_physics',
    'econometrics',
    'electrical_engineering',
    'elementary_mathematics',
    'formal_logic',
    'global_facts',
    'high_school_biology',
    'high_school_chemistry',
    'high_school_computer_science',
    'high_school_european_history',
    'high_school_geography',
    'high_school_government_and_politics',
    'high_school_macroeconomics',
    'high_school_mathematics',
    'high_school_microeconomics',
    'high_school_physics',
    'high_school_psychology',
    'high_school_statistics',
    'high_school_us_history',
    'high_school_world_history',
    'human_aging',
    'human_sexuality',
    'international_law',
    'jurisprudence',
    'logical_fallacies',
    'machine_learning',
    'management',
    'marketing',
    'medical_genetics',
    'miscellaneous',
    'moral_disputes',
    'moral_scenarios',
    'nutrition',
    'philosophy',
    'prehistory',
    'professional_accounting',
    'professional_law',
    'professional_medicine',
    'professional_psychology',
    'public_relations',
    'security_studies',
    'sociology',
    'us_foreign_policy',
    'virology',
    'world_religions'
]

def test_answer_mmlu_(pred_str, ans):
    pattern = 'answer is ('
    pred = pred_str.lower().split(pattern)
    if len(pred) == 1:
        pattern = 'answer is '
        pred = pred_str.lower().split(pattern)
        if len(pred) == 1:
            pattern = '('
            pred = pred_str.lower().split(pattern)

    if(len(pred) > 1):
        # print(pred)
        if len(pred[1]) == 0:
            pred = pred[1]
        else:
            pred = pred[1][0]
        # print(pred)
        gold = ans.lower()
        # print('debug 1, pred %s, gold %s' % (pred, gold))
        return pred == gold
    else:
        pred = 'C'
        # print(ans_str)
        gold = ans.lower()
        # print('debug 2, pred %s, gold %s' % (pred, gold))
        return pred == gold


BBH_MULTIPLE_CHOICE_TASKS = [
        'temporal_sequences', 'disambiguation_qa', 'date_understanding', 'tracking_shuffled_objects_three_objects', 'penguins_in_a_table', 
        'geometric_shapes', 'snarks', 'ruin_names', 'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_five_objects', 
        'logical_deduction_three_objects', 'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'movie_recommendation', 
        'salient_translation_error_detection', 'reasoning_about_colored_objects', 
]
BBH_FREE_FORM_TASKS = [
        'multistep_arithmetic_two', 'navigate', 'dyck_languages', 'word_sorting', 'sports_understanding', 
        'boolean_expressions', 'object_counting', 'formal_fallacies', 'causal_judgement', 'web_of_lies', 
]


def test_answer_bbh_(ans, mode, a):
    ans_line = ans.split('answer is ')
    # Expect to see 'answer is'. If not return whole string
    if len(ans_line) == 1:
        return ans == a
    else:
        ans = ans_line[-1].strip()

    if mode == 'multiple_choice':
        options = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', '(K)', '(L)', '(M)', '(N)', '(O)', '(P)', '(Q)', '(R)', '(S)', '(T)', '(U)', '(V)', '(W)', '(X)', '(Y)', '(Z)']
        for option in options:
            if option in ans:
                ans = option[1]
                break
        return ans == a
    elif mode == 'free_form':
        if ans[-1] == '.':
            ans = ans[:-1]
        return ans == a



# MATH-500
def find_answer_math_(s):
    assert('boxed' in s)
    ans = s.split('boxed')[-1]
    if(ans[0] == '{'):
        stack = 1
        a = ''
        for c in ans[1:]:
            if(c == '{'): 
                stack += 1
                a += c
            elif(c == '}'): 
                stack -= 1
                if(stack == 0): break
                a += c
            else: 
                a += c
    else:
        a = ans.split('$')[0].strip()
    return a


def test_answer_math_(pred_str, ans_str):
    if('The answer is: ' in pred_str):
        pred = pred_str.split('The answer is: ')[-1].strip()
    elif('The answer is ' in pred_str):
        pred = pred_str.split('The answer is ')[-1].strip()
    else:
        pattern = '\d*\.?\d+'
        pred = re.findall(pattern, pred_str)
        if(len(pred) >= 1):
            # print(pred_str)
            pred = pred[-1]
        else: pred = ''

    gold = find_answer_math_(ans_str)
    return pred == gold


# MBPP
MBPP_test_templates = """
{gen_code}
{test_codes}
"""

MBPP_prompt_templates = """
id: {task_id}
Question:
{text}

{test_codes}

Codes:
{code}

"""

MBPP_gen_templates = """
id: {task_id}
Question:
{text}

{test_codes}

Codes:
"""


def jsonl_to_dataframe(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            data_list.append(data)
    df = pd.DataFrame(data_list)
    return df


def generate_test_codes(test_list):
    return '\n'.join(test_list)


def test_answer_mbpp_(code_generate, test_list):
    code_generate = MBPP_test_templates.format(gen_code=code_generate, test_codes=generate_test_codes(test_list))
    try:
        if exec(code_generate, globals()) is None:
            return True
        else:
            return [None, False]
    except Exception as e:
        # print(str(e))
        return [str(e), False]
    
    
def load_judge_prompts_mtbench(prompt_file: str):
    """Load judge prompts.

    The return value is a python dict of type:
    Dict[judge_name: str -> dict]
    """
    prompts = {}
    with open(prompt_file) as fin:
        for line in fin:
            line = json.loads(line)
            prompts[line["name"]] = line
    return prompts



def replace_json_line(file_path, line_number, new_json_dict):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    if line_number < 1 or line_number > len(lines):
        raise IndexError("行号超出范围")
    lines[line_number - 1] = json.dumps(new_json_dict) + '\n'
    with open(file_path, 'w') as file:
        file.writelines(lines)


llm_models = [
    {
        'model': "gpt-4-turbo-2024-04-09", 
        'request_type': "openai", 
        'parameters': {"top_p": 0.7, "temperature": 0.9}
    },
    {
        'model': "gpt-4-turbo-preview", 
        'request_type': "openai", 
        'parameters': {"top_p": 0.7, "temperature": 0.9}
    },
]

def extract_rating_mtbench(text):
    import re
    pattern = r"\[\[(\d+)\]\]"
    match = re.findall(pattern, text)[-1]
    return int(match)

def test_answer_mtbench_(turns, reference, turns_output, model, types='single'):
    llm = LLMs(**model)
    judge_prompts = []
    with open("./lib_prompts/mtbench_judge_prompts.jsonl", "r") as json_file:
        for line in json_file:
            # print(line)
            judge_prompts.append(json.loads(line))
    if len(turns) > 1:
        turns_types = '-multi-turn'
    else:
        turns_types = ''
    try:
        if math.isnan(reference):
            reference_type = ''
        else:
            reference_type = '-math'
    except TypeError:
        reference_type = '-math'
    prompt_judge_name = f"{types}{reference_type}-v1{turns_types}"
    judge_prompt = None
    # print(prompt_judge_name)
    for i in range(len(judge_prompts)):
        if judge_prompts[i]['name'] == prompt_judge_name:
            judge_prompt = judge_prompts[i]
            break
    prompt_q = None
    if prompt_judge_name == 'single-v1-multi-turn':
        prompt_q = judge_prompt['prompt_template'].format(question_1=turns[0], answer_1=turns_output[0],
                                                          question_2=turns[1],  answer_2=turns_output[1])
    elif prompt_judge_name == 'single-v1':
        prompt_q = judge_prompt['prompt_template'].format(question=turns[0], answer=turns_output[0])
    elif prompt_judge_name == 'single-math-v1':
        prompt_q = judge_prompt['prompt_template'].format(question=turns[0], ref_answer_1=reference, answer=turns_output[0])
    elif prompt_judge_name == 'single-math-v1-multi-turn':
        prompt_q = judge_prompt['prompt_template'].format(question_1=turns[0], answer_1=turns_output[0],
                                                          question_2=turns[1], answer_2=turns_output[1],
                                                          ref_answer_1=reference[0], ref_answer_2=reference[1])

    sys_prompt = judge_prompt['system_prompt']
    # print(sys_prompt)
    # print(prompt_q)
    llm_judge = llm.request(prompt_q, sys_prompt)
    result = extract_rating_mtbench(llm_judge)
    return [result, llm_judge]