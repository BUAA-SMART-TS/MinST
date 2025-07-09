import logging

class SampleBank():
    def __init__(self):
        self._samples = []

    def add_sample(self, sample):
        reduced_sample = {}
        reduced_sample['Combination of modules'] = sample['Combination of modules']
        for k, v in sample.items():
            if k in ['mae', 'mape', 'rmse']:
                reduced_sample[k] = v
        # if len(self._samples) > 0:
        #    last_sample = self._samples[-1]
        #    for k in ['mae', 'mape', 'rmse']:
        #        change = last_sample[k] - reduced_sample[k]
        #        reduced_sample[k + ' is reduced by'] = change
        self._samples.append(reduced_sample)

    def get_samples(self):
        out_str = ""
        self._samples = sorted(self._samples, key=lambda x: x["mae"])
        for i in range(len(self._samples)):
            out_str += 'Round ' + str(i) + ': ' + str(self._samples[i]) + '\n'
        return out_str