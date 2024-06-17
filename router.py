from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.conversation import Conversation, SeparatorStyle
import numpy as np
import torch.nn.functional as F
import json
import torch
from sklearn import svm
from sklearn.metrics import accuracy_score
import joblib 
from transformers.generation.utils import (
    ModelOutput,
    top_k_top_p_filtering,
    StoppingCriteria,
    StoppingCriteriaList,
    LogitsProcessorList
)



class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)




class ThresholdRouter:
    def __init__(self, threshold=0.5) -> None:
        self.t = threshold
        
    def __call__(self, slm_outputs):
        slm_next_token_logits = slm_outputs.logits[:, -1, :]
        probs = F.softmax(slm_next_token_logits, dim=-1)
        probs_max = torch.max(probs, dim=-1)[0]
        return probs_max < self.t



class DeltaThresholdRouter:
    def __init__(self, threshold=0.5) -> None:
        self.t = threshold
        
    def __call__(self, slm_outputs):
        slm_next_token_logits = slm_outputs.logits[:, -1, :]
        probs = F.softmax(slm_next_token_logits, dim=-1)
        slm_topk_probs, _ = probs.topk(10)
        diff_probs = (slm_topk_probs[0][0] - slm_topk_probs[0][1])/2
        return diff_probs < self.t



class SVMRouter(object):
    # "test_expert_amateur_model_complex_threshold_1_logits.json"
    def __init__(self, svm_router_train_file, trained_model_path='./svm_router_top1000.pkl', topK=10) -> None:
        super(SVMRouter, self).__init__()
        if trained_model_path == None:
            data_logits = pd.read_json(svm_router_train_file)
            correct_index = []
            for i, it in enumerate(data_logits['results']):
                if re.search(r"Collaboration \[on green\]True\[/on green\]", it):
                    correct_index.append(i)
            data_logits = data_logits.loc[correct_index]
            self.train_logit = data_logits['amateur_topK_logits']
            self.train_diff = data_logits['difference_or_not']
            self.model = self.svm_train()
        else:
            self.model = joblib.load(trained_model_path)
        self.PNsample_ratio = 1
        self.pred_diff = None
        self.topK = topK

    def __call__(self, slm_outputs):
        slm_next_token_logits = slm_outputs.logits[:, -1, :]
        slm_next_token_probs = F.softmax(slm_next_token_logits, dim=-1).topk(self.topK)
        input_logit = [tensor.tolist() for tensor in slm_next_token_probs[0]]
        self.pred_diff = self.model.predict(input_logit)
        return self.pred_diff[0]

    def svm_train(self):
        diff_lists = []
        amateur_logits_top10_probs = []
        for it in self.train_logit:
            amateur_logits_top10_probs.append([jt[0][0] for jt in it])
        for it in self.train_diff:
            diff_lists.append(it)
        train_index = len(amateur_logits_top10_probs)
        train_amateur_logits_top10_probs = []
        train_diff_lists = []
        for it in amateur_logits_top10_probs[0:train_index]:
            train_amateur_logits_top10_probs += it
        for it in diff_lists[0:train_index]:
            train_diff_lists += it
        positive_indices = [i for i, label in enumerate(train_diff_lists) if label == 1]
        negative_indices = [i for i, label in enumerate(train_diff_lists) if label == 0]
        positive_indices = random.sample(positive_indices, len(positive_indices))
        negative_indices = random.sample(negative_indices, int(len(positive_indices)*self.PNsample_ratio))

        x = [train_amateur_logits_top10_probs[it] for it in positive_indices] + [train_amateur_logits_top10_probs[it]
                                                                                 for it
                                                                                 in
                                                                                 negative_indices]
        y = [train_diff_lists[it] for it in positive_indices] + [train_diff_lists[it] for it in negative_indices]

        svm_model = svm.SVC(kernel='rbf')
        svm_model.fit(x, y)
        joblib.dump(svm_model, f'./svm_router_top{1000}.pkl')
        return svm_model



class NormalRouter:
    def __init__(self) -> None:
        self.t = True
        
    def __call__(self, *args):
        return self.t

class NoneRouter:
    def __init__(self) -> None:
        self.t = False
        
    def __call__(self, *args):
        return self.t