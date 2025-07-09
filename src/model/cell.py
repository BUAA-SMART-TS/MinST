import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from model.mixed_op import MixedOp
from model.mixed_op_PC import MixedOpPCDARTS
from model.mode import Mode
import logging


class Cell(nn.Module):
    def __init__(self, channels, num_nodes, candidate_op_profiles, fixed_layers=None, opt='', weights=None):
        super(Cell, self).__init__()
        # create mixed operations
        self._channels = channels
        self._num_nodes = num_nodes
        self._mixed_ops = nn.ModuleDict() # nn.ModuleList()
        self.fixed_layers = fixed_layers
        
        for i in range(1, self._num_nodes):
            for j in range(i):
                node_str = '{:}<-{:}'.format(i,j)
                if 'PC' in opt:
                    self._mixed_ops[node_str] = MixedOpPCDARTS(candidate_op_profiles)
                else:
                    self._mixed_ops[node_str] = MixedOp(candidate_op_profiles)

        self._edge_keys = sorted(list(self._mixed_ops.keys()))
        self._edge2index = {key:i for i,key in enumerate(self._edge_keys)}
        self._index2edge = {i:key for i,key in enumerate(self._edge_keys)}
        self._num_edges = len(self._mixed_ops) # num_mixed_ops
        self._num_ops = len(candidate_op_profiles) # number of differnet op types

        # arch_weights
        if weights is None:
            self._candidate_alphas = nn.Parameter(1e-3*torch.randn(self._num_edges, self._num_ops), requires_grad=True) # [num_edges, num_ops]
        else:
            self._candidate_alphas = nn.Parameter(weights, requires_grad=True)  # [num_edges, num_ops]

        # pt_weights
        self._candidate_flags = nn.Parameter(torch.tensor(self._num_edges*[True], requires_grad=False, dtype=torch.bool), requires_grad=False) # # [num_edges,]
        self._project_weights = nn.Parameter(torch.zeros_like(self._candidate_alphas), requires_grad=False) # [num_edges, num_ops]

        self.set_mode(Mode.NONE)

    def set_mode(self, mode):
        self._mode = mode
        self.set_edge_mode(mode)

    def set_edge_mode(self, mode=None, allow_LLM=False):
        mode = mode or self._mode
        edge_sample_idx = {}
        for key, mix_op in self._mixed_ops.items(): # loop all edges, key: edge_name
            alpha = self._candidate_alphas[self._edge2index[key]].data
            if mode == Mode.PROJECT:
                if self._candidate_flags[self._edge2index[key]]==False:
                    op_id = torch.argmax(self._project_weights[self._edge2index[key]]).item() # choose the best op to set path
                    sample_idx = np.array([op_id], dtype=np.int32)
                else:
                    sample_idx = np.arange(self._num_ops)
            elif mode == Mode.NONE:
                sample_idx = None
            elif mode == Mode.ONE_PATH_FIXED:
                probs = F.softmax(alpha, dim=0)
                op = torch.argmax(probs).item()
                sample_idx = np.array([op], dtype=np.int32)
            elif mode == Mode.ONE_PATH_RANDOM:
                probs = F.softmax(alpha, dim=0)
                sample_idx = torch.multinomial(probs, 1, replacement=True).cpu().numpy()
            elif mode == Mode.TWO_PATHS:
                probs = F.softmax(alpha, dim=0)
                sample_idx = torch.multinomial(probs, 2, replacement=True).cpu().numpy()
            elif mode == Mode.ALL_PATHS:
                sample_idx = np.arange(self._num_ops)
            elif mode == Mode.NAS_BENCH:
                sample_idx = np.array([int(self.fixed_layers[self._edge2index[key]])])
            else:
                sample_idx = np.arange(self._num_ops)

            mix_op.set_mode(mode, sample_idx)
            edge_sample_idx[key] = sample_idx
        return edge_sample_idx

    def project_op(self, e_id, op_id):
        # set best op of the edge to 1
        self._project_weights[e_id][op_id] = 1 ## hard by default
        # if this edge had been choosed, set it to False
        self._candidate_flags[e_id] = False

    def project_parameters(self):
        # using arch weights get by DARTS search
        weights = F.softmax(self._candidate_alphas, dim=-1)
        for e_id in range(self._num_edges):
            if not self._candidate_flags[e_id]: # if the edge had been choosed, set new weights using hard code (1,0)
                weights[e_id].data.copy_(self._project_weights[e_id])

        return weights

    def arch_parameters(self):
        yield self._candidate_alphas

    def weight_parameters(self):
        for key, mix_op in self._mixed_ops.items():
            for p in mix_op.weight_parameters():
                yield p
    
    def proj_parameters(self):
        for pt_weight in [self._candidate_flags, self._project_weights]:
            yield pt_weight

    def num_weight_parameters(self):
        count = 0
        for key, mix_op in self._mixed_ops.items():
            count += mix_op.num_weight_parameters()
        return count

    def forward(self, x):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError


class ALLOTCell(Cell):
    def __init__(self, channels, num_mixed_ops, candidate_op_profiles, fixed_layers=None, opt='', weights=None):
        super(ALLOTCell, self).__init__(channels, num_mixed_ops, candidate_op_profiles, fixed_layers, opt=opt, weights=weights)

        # self.norm_layer1 = nn.LayerNorm(channels)
        # self.norm_layer2 = nn.LayerNorm(channels)

    def forward(self, x, x_mark, attn_mask=None, adj_mats=None, weights=None, **kwargs):
        # num_mixed_ops = 1
        nodes = [[x, kwargs['st']]]
        edge_sample_idx = self.set_edge_mode(self._mode)
        for i in range(1, self._num_nodes):
            inter_nodes_tt = []
            inter_nodes_st = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i,j)
                sample_idx = edge_sample_idx[node_str]
                edge_idx = self._edge2index[node_str]
                if weights is None:
                    op_weight = F.softmax(self._candidate_alphas[edge_idx][sample_idx], dim=-1)
                else:
                    op_weight = weights[edge_idx]
                
                inp = nodes[j][0]; kwargs['st'] = nodes[j][1]
                # r_graphs = kwargs['r_graphs']
                # print(r_graphs)
                tt_out, st_out = self._mixed_ops[node_str](inp, x_mark, attn_mask, adj_mats, op_weight, **kwargs)
                
                inter_nodes_tt.append(tt_out)
                inter_nodes_st.append(st_out)
            nodes.append([sum(inter_nodes_tt), sum(inter_nodes_st)])
        
        # print('nodes', len(nodes))
        node_out = kwargs['node_out']
        ret_tt = 0.; ret_st = 0.
        for n,node in enumerate(nodes):
            if node_out=='other' and n==0:  # do not need nodes[0]
                continue
            # print('add node', n, node[0].shape)
            ret_tt = ret_tt + node[0]
            ret_st = ret_st + node[1]
        
        return ret_tt, ret_st
        # return self.norm_layer1(ret_tt), self.norm_layer2(ret_st)

    def __repr__(self):
        edge_cnt = 0
        out_str = []
        for e_id in range(self._num_edges):
            node_str = self._index2edge[e_id]

            # probs = self._candidate_alphas[e_id]
            probs = F.softmax(self._candidate_alphas[e_id], dim=0) # [num_ops, ]

            projs = self._project_weights[e_id] # [num_ops, ]
            op_str = ['op:{}, prob:{:.3f}, proj:{}, info:{}'.format(i, prob, projs[i], self._mixed_ops[node_str]._candidate_ops[i]) for i,prob in enumerate(probs)]
            op_str = ',\n'.join(op_str)
            
            candidate_flag = self._candidate_flags[e_id] 
            out_str += ['mixed_op: {} {} candidate:{}\n{}\n{}'.format(e_id, node_str, candidate_flag,
                        self._mixed_ops[node_str], op_str)]

        out_str += ['candidate_flag: '+','.join(['{} {}'.format(e_id, self._candidate_flags[e_id]) for e_id in range(self._num_edges)])]
        out_str += ['proj_weights: '+';'.join(['e_id {}: '.format(e_id)+','.join(['{}'.format(p_w) for p_w in self._project_weights[e_id]]) for e_id in range(self._num_edges)])]

        from utils.helper import add_indent
        out_str = 'STCell {{\n{}\n}}'.format(add_indent('\n'.join(out_str), 4))
        return out_str
