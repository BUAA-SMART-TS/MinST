import torch
import torch.nn as nn
import torch.nn.functional as F

from model.cell import ALLOTCell
from model.mode import Mode
from model.embed import TemporalEncoding, TokenEncoding, PositionalEncoding
from utils.helper import resort_candidate_op


def create_layer(name, hidden_size, num_nodes, candidate_op_profiles, fixed_layers=None, opt='', weights=None):
    if name == 'ALLOTCell':
        return ALLOTCell(hidden_size, num_nodes, candidate_op_profiles, fixed_layers, opt=opt, weights=weights)
    if name == 'ConvPooling':
        return nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(1, 3), padding=(0, 1),
                         stride=(1, 2))
    if name == 'AvgPooling':
        return nn.AvgPool2d(kernel_size=(1,3), padding=(0,1), stride=(1,2))
    raise Exception('unknown layer name!')


class AdaptiveMatrixContainer(nn.Module):
    def __init__(self, adjinit, nodes):
        super(AdaptiveMatrixContainer, self).__init__()
        if adjinit is None:
            self._nodevec1 = nn.Parameter(torch.randn(nodes, 10).float(), requires_grad=True)
            self._nodevec2 = nn.Parameter(torch.randn(10, nodes).float(), requires_grad=True)
        else:
            adjinit = torch.from_numpy(adjinit)
            m, p, n = torch.svd(adjinit)
            initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
            initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
            self._nodevec1 = nn.Parameter(initemb1.float(), requires_grad=True)
            self._nodevec2 = nn.Parameter(initemb2.float(), requires_grad=True)

    def forward(self):
        adpadj = F.softmax(F.relu(torch.mm(self._nodevec1, self._nodevec2)), dim=1)

        return adpadj


class ALLOT(nn.Module):
    def __init__(self,
                 adjinit, nodes,
                 in_length, out_length,
                 in_size, out_size, 
                 hidden_size, skip_size, 
                 layer_names, skip_mode, node_out,
                 num_nodes, candidate_op_profiles,
                 dropout, fixed_layers=None, opt='', weights=None):
        super(ALLOT, self).__init__()

        self._cell_type = ALLOTCell
        self._skip_mode = skip_mode
        self._node_out = node_out

        candidate_op_profiles = resort_candidate_op(candidate_op_profiles)

        self.temporal_embedding = TemporalEncoding(hidden_size)
        self.position_embedding = PositionalEncoding(hidden_size)
        self.value_embedding = nn.Conv2d(in_size, hidden_size, kernel_size=(1, 1)) # TokenEncoding(c_in, d_model)
        self.spatil_embedding = nn.Conv2d(nodes, hidden_size, kernel_size=(1,1)) # nn.Linear(nodes, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)

        self._layers = nn.ModuleList()
        self._skip_convs = nn.ModuleList()
        self._vec_containers = nn.ModuleList()

        for name in layer_names:
            self._layers += [
                create_layer(name, hidden_size, num_nodes, candidate_op_profiles, fixed_layers, opt, weights)] # hidden_size == d_model
            self._skip_convs += [nn.Conv2d(hidden_size, skip_size, kernel_size=(1, 1))]
        
        for (op_name, profile) in candidate_op_profiles:
            if op_name not in ['Identity','Zero']:
                self._vec_containers.append(AdaptiveMatrixContainer(adjinit, nodes))

        self._end_conv1 = nn.Conv2d(in_length, out_length, kernel_size=(1, 1))
        self._end_conv2 = nn.Conv2d(skip_size, out_size, kernel_size=(1, 1))

        self.set_mode(Mode.NONE)

    def forward(self, x, x_mark, attn_mask, adj_mats, mode, weights, **kwargs):
        # x [B, L, N, D]
        self.set_mode(mode)

        adj_mats = torch.from_numpy(adj_mats).float().to(x.device)
        r_graphs = adj_mats.shape[-1]
        adpadjs = [vec_container().unsqueeze(-1).to(x.device) for vec_container in self._vec_containers]
        adj_mats = torch.cat([adj_mats] + adpadjs, dim=-1)
        
        st = 0. #adj_mats[:,:,0].unsqueeze(0).unsqueeze(0).expand(x.shape[0],x.shape[1],-1,-1) # [B, L, N, N]
        st_mask = attn_mask
        
        x = self.value_embedding(x.transpose(-1,1)).transpose(-1,1) + self.temporal_embedding(x_mark.transpose(-1,1)).transpose(-1,1) + self.position_embedding(x)
        x = self.embedding_dropout(x)
        # st = self.spatil_embedding(st.transpose(-1,1)).transpose(-1,1)

        skip = 0.
        # x [B, L, N, D]
        for idx, (layer, skip_conv) in enumerate(zip(self._layers, self._skip_convs)):
            if isinstance(layer, ALLOTCell):
                w_in =  None if weights is None else weights[idx]
                x, st = layer(x, x_mark, attn_mask, adj_mats, w_in, 
                              st=st, st_mask=st_mask, r_graphs=r_graphs, node_out=self._node_out,
                              **kwargs)
            else:
                x = layer(x)
            
            skip = skip + skip_conv(x.transpose(1,-1)).transpose(1,-1)
        # skip [B, L, N, D]
        if self._skip_mode=='skip':
            x = skip

        x = F.relu(self._end_conv1(x)) # [B, L, N, D]
        x = self._end_conv2(x.transpose(-1,1)).transpose(-1,1) # [B, L, N, D]

        # self.set_mode(Mode.NONE)
        return x.contiguous()

    def set_mode(self, mode):
        self._mode = mode
        for l in self._layers:
            if isinstance(l, self._cell_type):
                l.set_mode(mode)

    def weight_parameters(self):
        for m in [self.temporal_embedding, self.position_embedding, self.value_embedding, 
                  self.spatil_embedding]:
            for p in m.parameters():
                yield p
        for m in self._layers:
            if isinstance(m, self._cell_type):
                for p in m.weight_parameters():
                    yield p
            else:
                for p in m.parameters():
                    yield p
        for m in self._skip_convs:
            for p in m.parameters():
                yield p
        for m in self._vec_containers:
            for p in m.parameters():
                yield p
        for m in [self._end_conv1, self._end_conv2]:
            for p in m.parameters():
                yield p

    def arch_parameters(self):
        for m in self._layers:
            if isinstance(m, self._cell_type):
                for p in m.arch_parameters():
                    yield p

    def proj_parameters(self):
        for m in self._layers:
            if isinstance(m, self._cell_type):
                for p in m.proj_parameters():
                    yield p

    def num_weight_parameters(self):
        from utils.helper import num_parameters
        current_mode = self._mode
        self.set_mode(Mode.ONE_PATH_FIXED)
        count = 0
        for m in [self.temporal_embedding, self.position_embedding, self.value_embedding, 
                  self.spatil_embedding, self._end_conv1, self._end_conv2]:
            count += num_parameters(m)
        for m in self._layers:
            if isinstance(m, self._cell_type):
                count += m.num_weight_parameters()
            else:
                count += num_parameters(m)
        for m in self._skip_convs:
            count += num_parameters(m)
        for m in self._vec_containers:
            count += num_parameters(m)

        self.set_mode(current_mode)
        return count

    def num_cells(self):
        count = 0
        for layer in self._layers:
            if isinstance(layer, self._cell_type):
                count += 1
        return count
    
    def num_ops(self, cell_id=None):
        cell_idx = 0
        for layer in self._layers:
            if isinstance(layer, self._cell_type):
                if cell_id is not None:
                    if cell_idx==cell_id:
                        return layer._num_ops
                    cell_idx += 1
                else:
                    return layer._num_ops
    
    def num_edges(self, cell_id=None):
        cell_idx = 0
        for layer in self._layers:
            if isinstance(layer, self._cell_type):
                if cell_id is not None:
                    if cell_idx==cell_id:
                        return layer._num_edges
                    cell_idx += 1
                else:
                    return layer._num_edges

    def __repr__(self):
        out_str = []
        for l in self._layers:
            out_str += [str(l)]

        from utils.helper import add_indent
        out_str = 'Model {{\n{}\n}}\n'.format(add_indent('\n'.join(out_str), 4))
        return out_str

