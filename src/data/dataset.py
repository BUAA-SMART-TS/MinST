import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, SequentialSampler, RandomSampler

from setting import data_path
from utils.helper import Scaler, asym_adj
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')


class Dataset_ST(Dataset):
    def __init__(self, path, train_prop, test_prop,
                 tag='train', seq_len=12, pred_len=12):
        self._path = path
        self._tag = tag
        
        self._train_prop = train_prop
        self._test_prop = test_prop
        
        self._seq_len = seq_len
        self._pred_len = pred_len
        
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(data_path, self._path, 'traffic.csv'))
        
        # timefeatures
        timestamps = time_features(df_raw[['date']], 't')
        
        # remove date
        df_raw = df_raw[df_raw.columns[1:]]
        
        # fill nan
        df_rec = df_raw.copy()
        df_rec = df_rec.replace(0, np.nan)
        df_rec = df_rec.bfill() #.fillna(method='pad')
        df_rec = df_rec.ffill() #.fillna(method='bfill')
        
        # data split
        num_samples = len(df_rec)
        num_train = round(num_samples * self._train_prop)
        num_test = round(num_samples * self._test_prop)
        num_val = num_samples-num_train-num_test
        
        # set scaler
        train_data = df_rec.values[:num_train]
        self.scaler = Scaler(train_data, missing_value=0.)
        
        borders = {
            'train':[0, num_train],
            'valid':[num_train,num_train+num_val],
            'test':[num_samples-num_test,num_samples]
        }
        border1, border2 = borders[self._tag][0], borders[self._tag][1]
        data = df_rec.values[border1:border2]
        data = self.scaler.transform(data)
        
        self.data_x = data
        self.data_y = df_rec.values[border1:border2] # df_raw.values[border1:border2]
        self.data_t = timestamps[border1:border2]
        
    def __getitem__(self, index):
        s_begin = index; s_end = s_begin + self._seq_len
        r_begin = s_end; r_end = r_begin + self._pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_t[s_begin:s_end]
        seq_y_mark = self.data_t[r_begin:r_end]
        
        nodes = seq_x.shape[-1]
        seq_x = np.expand_dims(seq_x, -1) # [seq_len, nodes, 1]
        seq_y = np.expand_dims(seq_y, -1) # [pred_len, nodes, 1]
        seq_x_mark = np.tile(np.expand_dims(seq_x_mark, -2), [1,nodes,1]) # [seq_len, nodes, timefeatures_dim]
        seq_y_mark = np.tile(np.expand_dims(seq_y_mark, -2), [1,nodes,1]) # [pred_len, nodes, timefeatures_dim]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self._seq_len - self._pred_len + 1


class TrafficDataset:
    def __init__(self, path, train_prop, test_prop,
                 num_sensors, normalized_k=0.1, adj_type='distance',
                 in_length=12, out_length=12, batch_size=32):
        logging.info('initialize %s DataWrapper', path)
        self._path = path
        self._train_prop = train_prop
        self._test_prop = test_prop
        self._num_sensors = num_sensors
        self._normalized_k = normalized_k
        self._adj_type = adj_type
        self._in_length = in_length
        self._out_length = out_length
        self._batch_size = batch_size

        self.build_graph()

    def build_graph(self):
        logging.info('initialize graph')

        self.adj_mats = self.read_adj_mat()
        
        for dim in range(self.adj_mats.shape[-1]):
            # normalize adj_matrix
            if dim%2 != 0:
                self.adj_mats[:, :, dim][self.adj_mats[:,:,dim]==np.inf] = 0.
            else:
                values = self.adj_mats[:, :, dim][self.adj_mats[:, :, dim] != np.inf].flatten()       
                self.adj_mats[:, :, dim] = np.exp(-np.square(self.adj_mats[:, :, dim] / (values.std() + 1e-8)))
                self.adj_mats[:, :, dim][self.adj_mats[:, :, dim] < self._normalized_k] = 0
            # transfer adj_matrix
            self.adj_mats[:, :, dim] = asym_adj(self.adj_mats[:, :, dim])    
        
        if self._adj_type=='distance':
            self.adj_mats = self.adj_mats[:,:,::2]
        elif self._adj_type=='connect':
            self.adj_mats = self.adj_mats[:,:,1::2]
        else:
            pass

        dataset = Dataset_ST(
            path = self._path,
            train_prop = self._train_prop,
            test_prop = self._test_prop,
            tag = 'train',
            seq_len = self._in_length,
            pred_len = self._out_length
        )
        self.scaler = dataset.scaler
        
    def get_dataloader(self, tag='train', batch_size=None, num_workers=None):
        logging.info('load %s inputs & labels [start]', tag)
        
        dataset = Dataset_ST(
            path = self._path,
            train_prop = self._train_prop,
            test_prop = self._test_prop,
            tag = tag,
            seq_len = self._in_length,
            pred_len = self._out_length
        )
        
        data_batch_size = batch_size or self._batch_size
        data_shuffle = True if tag!='test' else False
        data_num_workers = num_workers or 4
        
        dataloader = DataLoader(
            dataset, 
            batch_size = data_batch_size, 
            shuffle = data_shuffle,
            num_workers = data_num_workers, 
            drop_last=False
        )
        
        logging.info('load %s inputs & labels [ok]', tag)
        logging.info('input shape: ({},{},{},{})'.format(len(dataset), self._in_length, self._num_sensors, 1))
        logging.info('label shape: ({},{},{},{})'.format(len(dataset), self._out_length, self._num_sensors, 1))
        # print(dataset.data_x, dataset.data_y)
        np.savez(os.path.join(os.path.join(data_path, self._path), tag), x=dataset.data_x, y=dataset.data_y)
        
        return dataloader
        
    def read_adj_mat(self):
        cache_file = os.path.join(data_path, self._path, 'sensor_graph/adjacent_matrix_cached.npz')
        try:
            arrays = np.load(cache_file)
            g = arrays['g']
            logging.info('load adj_mat from the cached file [ok]')
        except:
            logging.info('load adj_mat from the cached file [fail]')
            logging.info('load adj_mat from scratch')
            
            # read idx
            with open(os.path.join(data_path, self._path, 'sensor_graph/graph_sensor_ids.txt')) as f:
                ids = f.read().strip().split(',')
            idx = {}
            for i, id in enumerate(ids):
                idx[id] = i
            
            # read graph
            graph_csv = pd.read_csv(os.path.join(data_path, self._path, 'sensor_graph/distances.csv'),
                                    dtype={'from': 'str', 'to': 'str'})
            g = np.zeros((self._num_sensors, self._num_sensors, 2))
            g[:] = np.inf

            for k in range(self._num_sensors): g[k, k] = 0
            for row in graph_csv.values:
                if row[0] in idx and row[1] in idx:
                    g[idx[row[0]], idx[row[1]], 0] = row[2]  # distance
                    g[idx[row[0]], idx[row[1]], 1] = 1  # hop

            g = np.concatenate([g, np.transpose(g, (1, 0, 2))], axis=-1)
            np.savez_compressed(cache_file, g=g)
            logging.info('save graph to the cached file [ok]')
        return g
    

if __name__ == '__main__':
    path = 'PEMS03'
    train_prop = 0.7
    test_prop = 0.2
    num_sensors = 358
    normalized_k = 0.1
    adj_type = 'distance'
    in_length = 12
    out_length = 12
    batch_size = 32

    dataset = TrafficDataset(
        path=path,
        train_prop=train_prop,
        test_prop=test_prop,
        num_sensors=num_sensors,
        normalized_k=normalized_k,
        adj_type=adj_type,
        in_length=in_length,
        out_length=out_length,
        batch_size=batch_size
    )

    train_loader = dataset.get_dataloader(tag='train')
    for i, (x,y,x_mark,y_mark) in enumerate(train_loader):
        print(i, x.shape, y.shape, x_mark.shape, y_mark.shape)
