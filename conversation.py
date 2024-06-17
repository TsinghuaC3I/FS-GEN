from fastchat.conversation import Conversation, SeparatorStyle

# 找tokenizer_config.json

CONVS = {
    'Qwen1.5': Conversation(
        name="qwen",
        system_template="<|im_start|>system\n{system_message}",
        system_message="You are a helpful assistant.",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[
            151643,
            151644,
            151645,
            14582,
        ],  # "<|endoftext|>", "<|im_start|>", "<|im_end|>"
        stop_str="<|endoftext|>",
    ),
    'Qwen1.5-Chat': Conversation(
        name="qwen",
        system_template="<|im_start|>system\n{system_message}",
        system_message="You are a helpful assistant.",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[
            151643,
            151644,
            151645,
            14582,
        ],  # "<|endoftext|>", "<|im_start|>", "<|im_end|>"
        stop_str="<|endoftext|>",
    ),
    'Qwen': Conversation(
        name="qwen",
        system_template="<|im_start|>system\n{system_message}",
        system_message="You are a helpful assistant.",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[
            151643,
            151644,
            151645,
            14582,
        ],  # "<|endoftext|>", "<|im_start|>", "<|im_end|>"
        stop_str="<|endoftext|>",
    ),
    'Llama-2': Conversation(
        name="llama-2",
        system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
        roles=("[INST]", "[/INST]"),
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        sep2=" </s><s>",
    )
}


def generate_inputs(standard_conv, prompt_q, tokenizer):
    # print(standard_conv.name)
    if standard_conv.name == 'llama-2':
        conv = standard_conv.copy()
        conv.set_system_message("You will write beautiful compliments according to needs")
        # conv.append_message("<|user|>", prompt_q)
        # conv.append_message("<|assistant|>", None)
        conv.append_message("[INST]", prompt_q)
        conv.append_message("[/INST]", None)
        inputs = tokenizer(
            conv.get_prompt(),
            return_tensors='pt'
        )["input_ids"]
    elif standard_conv.name == 'qwen':
        conv = standard_conv.copy()
        conv.set_system_message("You will write beautiful compliments according to needs")
        conv.append_message("<|im_start|>user", prompt_q)
        conv.append_message("<|im_start|>assistant", None)
        inputs = tokenizer(
            conv.get_prompt(),
            return_tensors='pt'
        )["input_ids"]
    else:
        return None
    return inputs
