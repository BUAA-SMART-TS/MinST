import random
import torch
import torch.nn as nn
import logging

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

class Sample():
    def __init__(self, dic, type='mae'):
        super(Sample, self).__init__()
        self.sample = dic
        self.type = type

    def __lt__(self, other):
        return self.sample[self.type] < other.sample[self.type]

    def mutate(self):
        pos = random.randint(0, 5)
        op = random.randint(0, 2)
        model = self.sample['model']
        model['Layer_' + str(pos+1)] = reverse_token_dic[op]
        return model
    def __str__(self):
        return f'Model({self.sample})'

class Evolution(nn.Module):

    def __init__(self, strategy='Normal', opt_layers=6, choice=3, device='cpu', popu_size=5):
        super(Evolution, self).__init__()
        self._samples = []
        self.opt_layers = opt_layers
        self.choice = choice
        self.device = device
        self.popu_size=popu_size
        self.strategy=strategy

    def add_sample(self, dic):
        sample = Sample(dic)
        self._samples.append(sample)
        logging.info('Add model to popu:{}'.format(sample))

    def elimit_sample(self):
        if self.strategy == 'Normal':
            logging.info('Eli model from popu (Normal):{}'.format(max(self._samples)))
            self._samples.remove(max(self._samples))
        elif self.strategy == 'Age':
            logging.info('Eli model from popu (Age):{}'.format(self._samples[0]))
            self._samples.remove(self._samples[0])
            

    def get_best(self):
        return min(self._samples)

    def reproduce(self):
        parent = self.get_best()
        return parent.mutate()

    def __len__(self):
        return len(self._samples)

    def forward(self, matrix, metrics):
        sample = self.matrix2prompt(matrix, metrics)
        self.add_sample(sample)
        if len(self._samples) > self.popu_size:
            self.elimit_sample()
        new_model = self.reproduce()
        return self.prompt2matrix(new_model)
        
    def matrix2prompt(self, matrix, metrics):
        values = metrics.get_value()
        ops = torch.argmax(matrix, dim=-1)
        layer_dic = {}
        for i in range(self.opt_layers):
            layer_dic['Layer_' + str(i+1)] = reverse_token_dic[ops[i].item()]
        values['model'] = layer_dic
        return values
        
    def prompt2matrix(self, model):
        logging.info('New model:{}'.format(model))
        pro = torch.zeros(self.opt_layers, self.choice).to(self.device)
        for i in range(1, self.opt_layers + 1):
            pro[i - 1][token_dic[model['Layer_' + str(i)]]] = 0.5
        return pro


        