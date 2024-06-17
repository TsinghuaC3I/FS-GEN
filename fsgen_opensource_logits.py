from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.conversation import Conversation, SeparatorStyle
import numpy as np
import torch.nn.functional as F
import json
import torch
from sklearn import svm
from sklearn.metrics import accuracy_score
import joblib 
from router import ThresholdRouter, DeltaThresholdRouter, SVMRouter, NormalRouter, NoneRouter, StoppingCriteriaList, KeyWordsCriteria
# pip install transformers==4.38.2 --upgrade 

class FSGenOpenSource:
    def __init__(self, large_model, small_model, tokenizer, small_ft_model=None):
        self.large_model = large_model
        self.small_model = small_model
        self.small_ft_model = small_ft_model
        self.tokenizer = tokenizer

    def generate_text(self, input_ids, max_tokens=50, collabrate_method={'method': 'OrcleDecoding'}, router_method={'method':'normal'}, temperature=0.7, stop_word_ids=None):
        
        if router_method['method'] == 'threshold':
            method = ThresholdRouter(router_method['threshold'])
        elif router_method['method'] == 'delta_threshold':
            method = DeltaThresholdRouter(router_method['threshold'])
        elif router_method['method'] == 'svm':
            method = SVMRouter(router_method['svm_router_train_file'], router_method['trained_model_path'], router_method['topK'])
        elif router_method['method'] == 'normal':
            method = NormalRouter()
        elif router_method['method'] == 'none':
            method =  NoneRouter()
        else:
            method = router_method

        if collabrate_method['method'] == 'ContrastiveDecoding':
            return self.__contrastive_decode(input_ids, max_tokens, method, temperature, collabrate_method['alpha'], collabrate_method['beta'], stop_word_ids=stop_word_ids)
        elif collabrate_method['method'] == 'SpeculativeDecoding':
            return self.__speculative_decode(input_ids, max_tokens, method, temperature, collabrate_method['K'], stop_word_ids=stop_word_ids)
        elif collabrate_method['method'] == 'ProxyFineTuning':
            return self.__proxy_finetune(input_ids, max_tokens, method, temperature, collabrate_method['alpha'], stop_word_ids=stop_word_ids)
        else:
            return self.__oracle_decode(input_ids, max_tokens, method, temperature, stop_word_ids=stop_word_ids)

    def __oracle_decode(self, input_ids, max_tokens, method, temperature, **kwargs):
        slm_logits_prob = []
        llm_logits_prob = []
        if_match_now = []
        tokens = []
        topK = 100
        total_tokens_num = 0
        latent_tokens_num = 0
        mismatch_tokens_num = 0
        original_len = len(input_ids[0])
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        slm_kwargs = kwargs.copy()
        llm_kwargs = kwargs.copy()
        stop_word_ids = slm_kwargs.pop("stop_word_ids", None)
        generation_config = slm_kwargs.pop("generation_config", self.small_model.generation_config)
        if stop_word_ids is not None:
            stop_word_ids.append([generation_config.eos_token_id])
            stopping_criteria = StoppingCriteriaList([KeyWordsCriteria(stop_word_ids)]) if stop_word_ids else None
            pass
        else:
            stopping_criteria = None

        for i in range(max_tokens):
            total_tokens_num += 1
            slm_inputs = self.small_model.prepare_inputs_for_generation(input_ids, **slm_kwargs)
            slm_outputs = self.small_model(
                **slm_inputs, return_dict=True
            )
            slm_next_token_logits = slm_outputs.logits[:, -1, :].to(self.large_model.device)
            slm_next_token_probs = F.softmax(slm_next_token_logits, dim=-1).topk(topK)
            slm_indices = []
            for jt in slm_next_token_probs[1].tolist()[0]:
                slm_indices.append(jt)
            slm_logits_prob.append([[tensor.tolist() for tensor in slm_next_token_probs[0]], slm_indices])
            
            if method(slm_outputs):
                latent_tokens_num += 1
                llm_inputs = self.large_model.prepare_inputs_for_generation(input_ids, **llm_kwargs)
                llm_outputs = self.large_model(
                    **llm_inputs, return_dict=True
                )
                llm_next_token_logits = llm_outputs.logits[:, -1, :]

                next_tokens = torch.argmax(llm_next_token_logits, dim=-1)
                if next_tokens != torch.argmax(slm_next_token_logits, dim=-1):
                    if_match_now.append(1)
                    mismatch_tokens_num += 1
                else:
                    if_match_now.append(0)
            
                llm_next_token_probs = F.softmax(llm_next_token_logits, dim=-1).topk(topK)
                # print(slm_next_token_probs)
                llm_indices = []
                for jt in llm_next_token_probs[1].tolist()[0]:
                    llm_indices.append(jt)
                llm_logits_prob.append([[tensor.tolist() for tensor in llm_next_token_probs[0]], llm_indices])
                
                llm_kwargs = self.large_model._update_model_kwargs_for_generation(llm_outputs, llm_kwargs)
            else:
                next_tokens = torch.argmax(slm_next_token_logits, dim=-1)
            
            tokens.append(self.tokenizer.decode(next_tokens[0], skip_special_tokens=False))
            slm_kwargs = self.small_model._update_model_kwargs_for_generation(slm_outputs, slm_kwargs)
            # print(tokenizer.decode(input_ids[0][original_len:]))
            
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            # print(amateur_kwargs)
            if stopping_criteria and stopping_criteria(input_ids, None):
                break

            # if eos_token was found in one sentence, set sentence to finished    
            unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())
            if eos_token_id is not None:
                    assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
                    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                    
            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                break

        text = self.tokenizer.decode(input_ids[0][original_len:], skip_special_tokens=True)
        # print(text)

        SaveData = {
            'text': text, 
            'topK': topK,
            'tokens': tokens,
            'slm_logits_prob': slm_logits_prob,
            'llm_logits_prob': llm_logits_prob,
            'if_match_now': if_match_now,
            'total_tokens_num': total_tokens_num,
            'latent_tokens_num': latent_tokens_num,
            'mismatch_tokens_num': mismatch_tokens_num
        }
        return SaveData

    def __contrastive_decode(self, input_ids, max_tokens, method, temperature=0.7, alpha=0.1, beta=0.5, **kwargs):
        slm_logits_prob = []
        llm_logits_prob = []
        if_match_now = []
        tokens = []
        topK = 100
        total_tokens_num = 0
        latent_tokens_num = 0
        mismatch_tokens_num = 0
        original_len = len(input_ids[0])
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        slm_kwargs = kwargs.copy()
        llm_kwargs = kwargs.copy()
        stop_word_ids = slm_kwargs.pop("stop_word_ids", None)
        generation_config = slm_kwargs.pop("generation_config", self.small_model.generation_config)
        if stop_word_ids is not None:
            stop_word_ids.append([generation_config.eos_token_id])
            stopping_criteria = StoppingCriteriaList([KeyWordsCriteria(stop_word_ids)]) if stop_word_ids else None
            pass
        else:
            stopping_criteria = None
        # print(stop_word_ids)
        # print(self.tokenizer.decode(stop_word_ids, skip_special_tokens=False))
        for i in range(max_tokens):
            total_tokens_num += 1
            slm_inputs = self.small_model.prepare_inputs_for_generation(input_ids, **slm_kwargs)
            slm_outputs = self.small_model(
                **slm_inputs, return_dict=True
            )
            slm_next_token_logits = slm_outputs.logits[:, -1, :].to(self.large_model.device)
            slm_next_token_probs = F.softmax(slm_next_token_logits, dim=-1).topk(topK)
            slm_indices = []
            for jt in slm_next_token_probs[1].tolist()[0]:
                slm_indices.append(jt)
            slm_logits_prob.append([[tensor.tolist() for tensor in slm_next_token_probs[0]], slm_indices])
            
            if method(slm_outputs):
                latent_tokens_num += 1
                llm_inputs = self.large_model.prepare_inputs_for_generation(input_ids, **llm_kwargs)
                llm_outputs = self.large_model(
                    **llm_inputs, return_dict=True
                )
                llm_next_token_logits = llm_outputs.logits[:, -1, :]
                
                len_slm = slm_next_token_logits.shape[1]
                len_llm = llm_next_token_logits.shape[1]
                len_split = min(len_slm, len_llm)
                
                llm_next_token_logits = llm_next_token_logits[:, :len_split]
                slm_next_token_logits = slm_next_token_logits[:, :len_split]
                
                cutoff = np.log(alpha) + llm_next_token_logits.max(dim=-1, keepdim=True).values
                diffs = slm_next_token_logits + beta*(llm_next_token_logits - slm_next_token_logits)

                cd_logits = diffs.masked_fill(llm_next_token_logits < cutoff, -float('inf'))
                next_tokens = torch.argmax(cd_logits, dim=-1)
                # print(next_tokens[0])
                
                if next_tokens != torch.argmax(slm_next_token_logits, dim=-1):
                    if_match_now.append(1)
                    mismatch_tokens_num += 1
                else:
                    if_match_now.append(0)
                
                llm_next_token_probs = F.softmax(llm_next_token_logits, dim=-1).topk(topK)
                # print(slm_next_token_probs)
                llm_indices = []
                for jt in llm_next_token_probs[1].tolist()[0]:
                    llm_indices.append(jt)
                llm_logits_prob.append([[tensor.tolist() for tensor in llm_next_token_probs[0]], llm_indices])
                
                llm_kwargs = self.large_model._update_model_kwargs_for_generation(llm_outputs, llm_kwargs)
            else:
                next_tokens = torch.argmax(slm_next_token_logits, dim=-1)

            tokens.append(self.tokenizer.decode(next_tokens[0], skip_special_tokens=False))
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            slm_kwargs = self.small_model._update_model_kwargs_for_generation(slm_outputs, slm_kwargs)
            
            # print(amateur_kwargs)
            if stopping_criteria and stopping_criteria(input_ids, None):
                break

            # if eos_token was found in one sentence, set sentence to finished
            unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())
            if eos_token_id is not None:
                    assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
                    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                    
            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                break
        
        text = self.tokenizer.decode(input_ids[0][original_len:], skip_special_tokens=True)
        SaveData = {
            'text': text, 
            'topK': topK,
            'tokens': tokens,
            'slm_logits_prob': slm_logits_prob,
            'llm_logits_prob': llm_logits_prob,
            'if_match_now': if_match_now,
            'total_tokens_num': total_tokens_num,
            'latent_tokens_num': latent_tokens_num,
            'mismatch_tokens_num': mismatch_tokens_num
        }
        return SaveData

    def __speculative_decode(self, input_ids, max_tokens, method, temperature=0.7, K=5, **kwargs):
        total_tokens_num = 0
        latent_tokens_num = 0
        mismatch_tokens_num = 0
        original_len = len(input_ids[0])
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        slm_kwargs = kwargs.copy()
        llm_kwargs = kwargs.copy()
        stop_word_ids = slm_kwargs.pop("stop_word_ids", None)
        generation_config = slm_kwargs.pop("generation_config", self.small_model.generation_config)
        if stop_word_ids is not None:
            stop_word_ids.append([generation_config.eos_token_id])
            stopping_criteria = StoppingCriteriaList([KeyWordsCriteria(stop_word_ids)]) if stop_word_ids else None
            pass
        else:
            stopping_criteria = None
        n = len(input_ids[0])
        T = max_tokens + len(input_ids[0])
        ctn_slm_tokens_num = 0

        while n < T:
            # print(n)
            input_ids_draft = input_ids.clone()
            # print(self.tokenizer.decode(input_ids[0][original_len:]))
            import time
            
            slm_inputs = self.small_model.prepare_inputs_for_generation(input_ids, **slm_kwargs)
            slm_outputs = self.small_model(
                **slm_inputs, return_dict=True
            )
            if method(slm_outputs):
                
                latent_tokens_num += 1
                slm_outputs_lists = []
                slm_kwargs_draft = slm_kwargs.copy()
                for step_k in range(K):
                    slm_inputs = self.small_model.prepare_inputs_for_generation(input_ids_draft, **slm_kwargs_draft)
                    with torch.no_grad():
                        slm_outputs = self.small_model(
                            **slm_inputs, return_dict=True
                        )
                        slm_outputs_lists.append(slm_outputs)
                        slm_next_token_logits = slm_outputs.logits[:, -1, :]
                        predicted_token = torch.argmax(slm_next_token_logits, dim=-1)
                    input_ids_draft = torch.cat([input_ids_draft, predicted_token[:, None]], dim=-1)
                    slm_kwargs_draft = self.small_model._update_model_kwargs_for_generation(slm_outputs, slm_kwargs_draft)
                
                llm_inputs = self.large_model.prepare_inputs_for_generation(input_ids_draft, **llm_kwargs)
                with torch.no_grad():
                    # print("llm input ids", llm_inputs["input_ids"].shape)
                    llm_outputs = self.large_model(
                        **llm_inputs, return_dict=True
                    )
                    llm_next_token_logits = llm_outputs.logits[:, -1, :]
                
                all_accepted = True
                unaccepted_num = K 
                for index in range(K):
                    now_draft = n - 1
                    # print(llm_outputs.logits.shape)
                    if llm_outputs.logits.shape[1] >= original_len:
                        i = n - 1
                    else:
                        i = index + ctn_slm_tokens_num
                    # print(input_ids_draft)
                    j = torch.tensor([input_ids_draft[0][now_draft + 1].item()], device=input_ids_draft[0][now_draft + 1].device)
                    
                    if torch.argmax(llm_outputs.logits[:, i, :], dim=-1) == j:
                        total_tokens_num += 1
                        input_ids = torch.cat([input_ids, torch.argmax(llm_outputs.logits[:, i, :], dim=-1)[:, None]], dim=-1)
                        n += 1
                        unaccepted_num -= 1
                        if stopping_criteria and stopping_criteria(input_ids, None):
                            all_accepted = False
                            break
                    else:
                        mismatch_tokens_num += 1
                        total_tokens_num += 1
                        input_ids = torch.cat([input_ids, torch.argmax(llm_outputs.logits[:, i, :], dim=-1)[:, None]], dim=-1)
                        next_tokens = torch.argmax(llm_outputs.logits[:, i, :], dim=-1)
                        n += 1
                        unaccepted_num -= 1
                        all_accepted = False
                        break
                
                if all_accepted:
                    total_tokens_num += 1
                    input_ids = torch.cat([input_ids, torch.argmax(llm_outputs.logits[:, -1, :], dim=-1)[:, None]], dim=-1)
                    next_tokens = torch.argmax(llm_outputs.logits[:, -1, :], dim=-1)
                    n += 1
                # print(unaccepted_num)
                for slm_id in range(K-unaccepted_num):
                    slm_kwargs = self.small_model._update_model_kwargs_for_generation(slm_outputs_lists[slm_id], slm_kwargs)
                if all_accepted is False:
                    unaccepted_num += 1

                llm_outputs.logits = llm_outputs.logits[:, :-unaccepted_num, :]
                llm_outputs.past_key_values = [tuple(p[:, :, :-unaccepted_num, :] if len(p.shape) == 4 else p for p in layer) for layer in llm_outputs.past_key_values]
                llm_kwargs = self.large_model._update_model_kwargs_for_generation(llm_outputs, llm_kwargs)
                ctn_slm_tokens_num = 0
            else:
                total_tokens_num += 1
                input_ids = torch.cat([input_ids, torch.argmax(slm_outputs.logits[:, -1, :], dim=-1)[:, None]], dim=-1)
                next_tokens = torch.argmax(slm_outputs.logits[:, -1, :], dim=-1)
                slm_kwargs = self.small_model._update_model_kwargs_for_generation(slm_outputs, slm_kwargs)
                ctn_slm_tokens_num += 1
                n += 1
            
            if stopping_criteria and stopping_criteria(input_ids, None):
                break

            unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())
            if eos_token_id is not None:
                    assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
                    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                    
            if unfinished_sequences.max() == 0:
                break

        text = self.tokenizer.decode(input_ids[0][original_len:], skip_special_tokens=True)
        outcome = {
            'text': text, 
            'total_tokens_num': total_tokens_num,
            'latent_tokens_num': latent_tokens_num,
            'mismatch_tokens_num': mismatch_tokens_num
        }

        return outcome

    def __proxy_finetune(self, input_ids, max_tokens, method, temperature=0.7, alpha=1.0, **kwargs):
        assert self.small_ft_model is not None
        slm_logits_prob = []
        llm_logits_prob = []
        if_match_now = []
        tokens = []
        topK = 100
        total_tokens_num = 0
        latent_tokens_num = 0
        mismatch_tokens_num = 0
        original_len = len(input_ids[0])
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        slm_kwargs = kwargs.copy()
        slm_ft_kwargs = kwargs.copy()
        llm_kwargs = kwargs.copy()
        stop_word_ids = slm_kwargs.pop("stop_word_ids", None)
        generation_config = slm_kwargs.pop("generation_config", self.small_model.generation_config)
        if stop_word_ids is not None:
            stop_word_ids.append([generation_config.eos_token_id])
            stopping_criteria = StoppingCriteriaList([KeyWordsCriteria(stop_word_ids)]) if stop_word_ids else None
            pass
        else:
            stopping_criteria = None

        for i in range(max_tokens):
            total_tokens_num += 1
            slm_ft_inputs = self.small_ft_model.prepare_inputs_for_generation(input_ids, **slm_ft_kwargs)
            slm_ft_outputs = self.small_ft_model(
                **slm_ft_inputs, return_dict=True
            )
            slm_ft_next_token_logits = slm_ft_outputs.logits[:, -1, :].to(self.large_model.device)
            slm_ft_next_token_probs = F.softmax(slm_ft_next_token_logits, dim=-1).topk(topK)
            slm_indices = []
            for jt in slm_ft_next_token_probs[1].tolist()[0]:
                slm_indices.append(jt)
            slm_logits_prob.append([[tensor.tolist() for tensor in slm_ft_next_token_probs[0]], slm_indices])
            
            if method(slm_ft_outputs):
                latent_tokens_num += 1
                slm_inputs = self.small_model.prepare_inputs_for_generation(input_ids, **slm_kwargs)
                slm_outputs = self.small_model(
                    **slm_inputs, return_dict=True
                )
                slm_next_token_logits = slm_outputs.logits[:, -1, :].to(self.large_model.device)

                llm_inputs = self.large_model.prepare_inputs_for_generation(input_ids, **llm_kwargs)
                llm_outputs = self.large_model(
                    **llm_inputs, return_dict=True
                )
                llm_next_token_logits = llm_outputs.logits[:, -1, :]

                llm_len = llm_next_token_logits.shape[1]
                slm_len = slm_next_token_logits.shape[1]
                len_split = min(llm_len, slm_len)
                
                slm_next_token_logits = slm_next_token_logits[:, :len_split]
                llm_next_token_logits = llm_next_token_logits[:, :len_split]
                slm_ft_next_token_logits = slm_ft_next_token_logits[:, :len_split]
                
                ft_next_token_logits = slm_ft_next_token_logits - slm_next_token_logits
                llm_ft_next_token_logits = alpha * ft_next_token_logits + llm_next_token_logits

                # llm_ft_next_token_logits = slm_ft_next_token_logits * llm_next_token_logits / slm_next_token_logits
                next_tokens = torch.argmax(llm_ft_next_token_logits, dim=-1)
                if next_tokens != torch.argmax(slm_next_token_logits, dim=-1):
                    if_match_now.append(1)
                    mismatch_tokens_num += 1
                else:
                    if_match_now.append(0)
                # print(next_tokens[0])
                
                
                llm_next_token_probs = F.softmax(llm_next_token_logits, dim=-1).topk(topK)
                # print(slm_next_token_probs)
                llm_indices = []
                for jt in llm_next_token_probs[1].tolist()[0]:
                    llm_indices.append(jt)
                llm_logits_prob.append([[tensor.tolist() for tensor in llm_next_token_probs[0]], llm_indices])
                
                slm_kwargs = self.small_model._update_model_kwargs_for_generation(slm_outputs, slm_kwargs)
                llm_kwargs = self.large_model._update_model_kwargs_for_generation(llm_outputs, llm_kwargs)
            else:
                next_tokens = torch.argmax(slm_ft_next_token_logits, dim=-1)
            tokens.append(self.tokenizer.decode(next_tokens[0], skip_special_tokens=False))
            slm_ft_kwargs = self.small_ft_model._update_model_kwargs_for_generation(slm_ft_outputs, slm_ft_kwargs)
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            if stopping_criteria and stopping_criteria(input_ids, None):
                break

            unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())
            if eos_token_id is not None:
                assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                    
            if unfinished_sequences.max() == 0:
                break

        text = self.tokenizer.decode(input_ids[0][original_len:], skip_special_tokens=True)
        # print(text)
        SaveData = {
            'text': text, 
            'topK': topK,
            'tokens': tokens,
            'slm_logits_prob': slm_logits_prob,
            'llm_logits_prob': llm_logits_prob,
            'if_match_now': if_match_now,
            'total_tokens_num': total_tokens_num,
            'latent_tokens_num': latent_tokens_num,
            'mismatch_tokens_num': mismatch_tokens_num
        }
        return SaveData


