from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.get_LLM import get_LLM
from prompt_bank.get_prompt import get_prompt, get_new_state_prompt, get_evaluate_prompt
from prompt_bank.samples import SampleBank
import json
import logging
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

token_dic = {
    'spatial-then-temporal': 0,
    'spatial-temporal-parallely': 1,
    'temporal-then-spatial': 2,
}

reverse_token_dic = {
    0: 'spatial-then-temporal',
    1: 'spatial-temporal-parallely',
    2: 'temporal-then-spatial'
}

class LLM(nn.Module):

    def __init__(self, dataset_name, device, llm_model='LLAMA', llm_layers=3, opt_layers=6, choice=3, max_trial=5, option='DFS', rate=0.8):
        super(LLM, self).__init__()
        self.task = dataset_name
        self.opt_layers = opt_layers
        self.choice = choice
        self.device = device

        self.tokenizer, self.llm_model = get_LLM(llm_model, llm_layers)

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000

        self.max_trial = max_trial
        self.option = option
        self.rate = rate

        self.sample_bank = SampleBank()

    def get_response_from_LLM(self, prompt):
        logging.info('Prompt:{}'.format(prompt))

        messages = [{'role': 'system',
                     'content': 'You are a helpful assistant system and an expert in neural architecture search'},
                    {'role': 'user', 'content': prompt + '<| Answer Start |>'}]

        # 使用分词器的 apply_chat_template 方法将上面定义的消,息列表转护# tokenize=False 表示此时不进行令牌化，add_generation_promp
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 将处理后的文本令牌化并转换为模型输入张量，然后将这些张量移至之前
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        generated_ids = self.llm_model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # 对输出进行解码
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        response = self.get_answer(response[-1])
        return response
    def get_new_state(self, samples, current_layers, num_current_layer):
        prompt = get_new_state_prompt(samples, current_layers, num_current_layer)
        response = self.get_response_from_LLM(prompt)
        try:
            data = json.loads(response)
            logging.info('LLM response:{}'.format(data))
            current_layers['Layer_' + str(num_current_layer + 1)] = data['New layer']
            return current_layers
        except Exception as e:
            logging.info(e)
            return None

    def get_evaluate(self, samples, current_layers, num_current_layer):
        prompt = get_evaluate_prompt(samples, current_layers, num_current_layer)
        response = self.get_response_from_LLM(prompt)
        logging.info("LLM eval: {}".format(response))
        try:
            data = json.loads(response)
            if data['Judgement'] == 'possible':
                return True
            else:
                return False
        except Exception as e:
            return False

    def dfs_forward(self, samples, current_layers, num_current_layer):
        if num_current_layer >= self.opt_layers:
            return self.dict2matrix(current_layers)
        for i in range(0, self.max_trial - 1):
            new_current_layers = self.get_new_state(samples, current_layers, num_current_layer)
            if new_current_layers is not None and self.get_evaluate(samples, new_current_layers, num_current_layer + 1):
                return self.dfs_forward(samples, new_current_layers, num_current_layer + 1)
        new_current_layers = self.get_new_state(samples, current_layers, num_current_layer)
        return self.dfs_forward(samples, new_current_layers, num_current_layer + 1)

    def straight_forward(self, samples, current_epoch, total_epoch):
        prompt = get_prompt(self.task, samples, current_epoch, total_epoch, self.rate)
        response = self.get_response_from_LLM(prompt)
        matrix = self.prompt2matrix(response)
        return matrix

    def forward(self, matrix, metrics, current_epoch, total_epoch):

        # add new sample to memory bank
        sample = self.matrix2prompt(matrix, metrics)
        self.sample_bank.add_sample(sample)

        # get history results from memory bank
        samples = self.sample_bank.get_samples()
        explore_rounds = total_epoch * self.rate

        if self.option == 'straight' or current_epoch >= explore_rounds:
            matrix = self.straight_forward(samples, current_epoch, total_epoch)
        else:
            current_layers = {}
            matrix = self.dfs_forward(samples, current_layers, 0)

        return matrix

    def get_answer(self, text):
        index = text.find('<| Answer Start |>')

        # 获取特定子字符串之后的字符串
        result = text[index + len('<| Answer Start |>'):]

        index = result.find('assistant')
        result = result[index + len('assistant'):]
        result = result.strip()

        return result

    def matrix2prompt(self, matrix, metrics):
        values = metrics.get_value()
        ops = torch.argmax(matrix, dim=-1)
        layer_dic = {}
        for i in range(self.opt_layers):
            layer_dic['Layer_' + str(i+1)] = reverse_token_dic[ops[i].item()]
        values['Combination of modules'] = layer_dic
        return values

    def prompt2matrix(self, text):
        logging.info('LLM response:{}'.format(text))
        try:
            data = json.loads(text)
            pro = torch.zeros(self.opt_layers, self.choice).to(self.device)
            for i in range(1, self.opt_layers + 1):
                pro[i - 1][token_dic[data['Combination of modules']['Layer_' + str(i)]]] = 0.5
            return pro
        except Exception as e:
            logging.info(e)
            return None
    def dict2matrix(self, data):
        logging.info('LLM response:{}'.format(data))
        try:
            pro = torch.zeros(self.opt_layers, self.choice).to(self.device)
            for i in range(1, self.opt_layers + 1):
                pro[i - 1][token_dic[data['Layer_' + str(i)]]] = 0.5
            return pro
        except Exception as e:
            logging.info(e)
            return None

