import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attn import S2TAttentionLayer, STSAttentionLayer, T2SAttentionLayer, STSGAttentionLayer, T2SGAttentionLayer, \
        STLinearAttention, STSLinearAttention, STFullAttention
from model.encoder import Encoder, EncoderLayer
from model.decoder import DecoderLayer, Decoder
from model.gcn import GraphConv, NGraphConv


def create_op(op_name, setting):
    name2op = {
        'S2TLayer': lambda: S2TLayer(**setting),
        'T2SLayer': lambda: T2SLayer(**setting),
        'STSLayer': lambda: STSLayer(**setting),
        'STSGLayer': lambda: STSGLayer(**setting),
        'T2SGLayer': lambda: T2SGLayer(**setting),
        'T2STLayer': lambda: T2STLayer(**setting),
        'Identity': lambda: Identity(),
        'Zero': lambda: Zero()
    }
    op = name2op[op_name]()
    return op


class BasicOp(nn.Module):
    def __init__(self, **kwargs):
        super(BasicOp, self).__init__()

    def forward(self, inputs, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        cfg = []
        for (key, value) in self.setting:
            cfg += [str(key) + ': ' + str(value)]
        return str(self.type) + '(' + ', '.join(cfg) + ')'

    @property
    def type(self):
        raise NotImplementedError

    @property
    def setting(self):
        raise NotImplementedError


class Identity(BasicOp):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x, x_mark, attn_mask=None, adj_mats=None, **kwargs):
        if isinstance(x, list):
            out = 0.
            for i in x: out += i
            return out, kwargs['st']
        else:
            return x, kwargs['st']

    @property
    def type(self):
        return 'Identity'

    @property
    def setting(self):
        return []


class Zero(BasicOp):
    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x, x_mark, attn_mask=None, adj_mats=None, **kwargs):
        return torch.zeros_like(x[0] if isinstance(x, list) else x), kwargs['st']

    @property
    def type(self):
        return 'Zero'

    @property
    def setting(self):
        return []


class S2TLayer(BasicOp):
    def __init__(self, seq_len, out_len, d_model=512, n_heads=8, d_ff=8, nodes=207, 
                n_graphs=3, order=2, use_bn=True, dropout=0.1, activation='gelu'):
        super(S2TLayer, self).__init__()
        self._d_model = d_model
        self._n_heads = n_heads
        self._d_ff = d_ff
        
        self.s2t_layer = EncoderLayer(
            S2TAttentionLayer(STLinearAttention(mask_flag=False), d_model, n_heads, 
                                n_graphs=n_graphs, order=order, use_bn=use_bn, dropout=dropout),
            d_model,
            d_ff,
            dropout=dropout,
            activation=activation
        ) 
        
    def forward(self, x, x_mark, attn_mask=None, adj_mats=None, **kwargs):
        # x [B, L, N, D]
        out = self.s2t_layer(x, attn_mask, adj_mats, **kwargs) # [B, L, N, D]
        
        return out, kwargs['st']
    
    @property
    def type(self):
        return 'S2TLayer'

    @property
    def setting(self):
        return [
            ('d_model', self._d_model),
            ('n_heads', self._n_heads),
            ('d_ff', self._d_ff),
        ]


class STSLayer(BasicOp):
    def __init__(self, seq_len, out_len, d_model=512, n_heads=8, d_ff=8, nodes=207, 
                n_graphs=3, order=2, use_bn=True, dropout=0.1, activation='gelu'):
        super(STSLayer, self).__init__()
        self._d_model = d_model
        self._n_heads = n_heads
        self._d_ff = d_ff
        
        self.sts_layer = EncoderLayer(
            STSAttentionLayer(STSLinearAttention(mask_flag=False), d_model, n_heads, 
                                n_graphs=n_graphs, order=order, use_bn=use_bn, dropout=dropout),
            d_model,
            d_ff,
            dropout=dropout,
            activation=activation
        )
        
    def forward(self, x, x_mark, attn_mask=None, adj_mats=None, **kwargs):
        # x [B, L, N, D]
        out = self.sts_layer(x, attn_mask, adj_mats, **kwargs) # [B, L, N, D]
        
        return out, kwargs['st']
    
    @property
    def type(self):
        return 'STSLayer'

    @property
    def setting(self):
        return [
            ('d_model', self._d_model),
            ('n_heads', self._n_heads),
            ('d_ff', self._d_ff),
        ]


class STSGLayer(BasicOp):
    def __init__(self, seq_len, out_len, d_model=512, n_heads=8, d_ff=8, nodes=207, 
                n_graphs=3, order=2, use_bn=True, dropout=0.1, activation='gelu'):
        super(STSGLayer, self).__init__()
        self._d_model = d_model
        self._n_heads = n_heads
        self._d_ff = d_ff
        
        self.stsg_layer = EncoderLayer(
            STSGAttentionLayer(STSLinearAttention(mask_flag=False), d_model, n_heads, 
                                n_graphs=n_graphs, order=order, use_bn=use_bn, dropout=dropout),
            d_model,
            d_ff,
            dropout=dropout,
            activation=activation
        )
        
    def forward(self, x, x_mark, attn_mask=None, adj_mats=None, **kwargs):
        # x [B, L, N, D]
        out = self.stsg_layer(x, attn_mask, adj_mats, **kwargs) # [B, L, N, D]
        
        return out, kwargs['st']
    
    @property
    def type(self):
        return 'STSGLayer'

    @property
    def setting(self):
        return [
            ('d_model', self._d_model),
            ('n_heads', self._n_heads),
            ('d_ff', self._d_ff),
        ]


class T2SGLayer(BasicOp):
    def __init__(self, seq_len, out_len, d_model=512, n_heads=8, d_ff=8, nodes=207, 
                n_graphs=3, order=2, use_bn=True, dropout=0.1, activation='gelu'):
        super(T2SGLayer, self).__init__()
        self._d_model = d_model
        self._n_heads = n_heads
        self._d_ff = d_ff
        
        self.t2sg_layer = EncoderLayer(
            T2SGAttentionLayer(STLinearAttention(mask_flag=False), d_model, n_heads, 
                                n_graphs=n_graphs, order=order, use_bn=use_bn, dropout=dropout),
            d_model,
            d_ff,
            dropout=dropout,
            activation=activation
        ) 
        
    def forward(self, x, x_mark, attn_mask=None, adj_mats=None, **kwargs):
        # x [B, L, N, D]
        out = self.t2sg_layer(x, attn_mask, adj_mats, **kwargs) # [B, L, N, D]
        
        return out, kwargs['st']
    
    @property
    def type(self):
        return 'T2SGLayer'

    @property
    def setting(self):
        return [
            ('d_model', self._d_model),
            ('n_heads', self._n_heads),
            ('d_ff', self._d_ff),
        ]


class T2STLayer(BasicOp):
    def __init__(self, seq_len, out_len, d_model=512, n_heads=8, d_ff=8, nodes=207, 
                n_graphs=3, order=2, use_bn=True, dropout=0.1, activation='gelu'):
        super(T2STLayer, self).__init__()
        self._d_model = d_model
        self._n_heads = n_heads
        self._d_ff = d_ff
        
        self.t2s_layer1 = EncoderLayer(
            S2TAttentionLayer(STLinearAttention(mask_flag=False), d_model, n_heads, 
                                n_graphs=n_graphs, order=order, use_bn=use_bn, dropout=dropout),
            d_model,
            d_ff,
            dropout=dropout,
            activation=activation
        ) 
        self.t2s_layer2 = EncoderLayer(
            S2TAttentionLayer(STLinearAttention(mask_flag=False), d_model, n_heads, 
                                n_graphs=n_graphs, order=order, use_bn=use_bn, dropout=dropout),
            d_model,
            d_ff,
            dropout=dropout,
            activation=activation
        ) 
        self.gcn = GraphConv(d_model*2, d_model, n_graphs, order, use_bn, dropout)
        
    def forward(self, x, x_mark, attn_mask=None, adj_mats=None, **kwargs):
        # x [B, L, N, D]
        out1 = self.t2s_layer1(x, attn_mask, adj_mats, **kwargs) # [B, L, N, D]
        out2 = self.t2s_layer2(x, attn_mask, adj_mats, **kwargs) # [B, L, N, D]

        out = torch.cat([out1, out2], dim=-1) # [B, L, N, D*2]
        out = self.gcn(out.transpose(-1, 1), adj_mats, **kwargs) # [B, D, N, L]
        out = out.transpose(-1, 1)

        return out, kwargs['st']
    
    @property
    def type(self):
        return 'T2STLayer'

    @property
    def setting(self):
        return [
            ('d_model', self._d_model),
            ('n_heads', self._n_heads),
            ('d_ff', self._d_ff),
        ]


class T2SModule(nn.Module):
    def __init__(self, seq_len, out_len, d_model=512, n_heads=8, d_ff=8, nodes=207, 
                n_graphs=3, order=2, use_bn=True, dropout=0.1, activation='gelu'):
        super(T2SModule, self).__init__()

        self.temporal_layer = EncoderLayer(
            T2SAttentionLayer(STLinearAttention(mask_flag=False), d_model, n_heads, 
                            n_graphs=n_graphs, order=order, use_bn=use_bn, dropout=dropout),
            d_model,
            d_ff,
            dropout=dropout,
            activation=activation
        )
        self.spatial_layer = DecoderLayer(
            T2SAttentionLayer(STLinearAttention(mask_flag=False), d_model, n_heads, 
                            n_graphs=n_graphs, order=order, use_bn=use_bn, dropout=dropout),
            T2SAttentionLayer(STLinearAttention(mask_flag=False), d_model, n_heads, 
                            n_graphs=n_graphs, order=order, use_bn=use_bn, dropout=dropout),
            d_model=d_model,
            d_ff=d_ff, 
            dropout=dropout,
            activation=activation
        )
        self.gcn_layers = nn.ModuleList(
            [
                NGraphConv(
                    d_model, d_model, n_graphs+1, order, use_bn, dropout
                ) for t in range(out_len)
            ]
        )
        self.trans_layer_t = nn.Conv2d(in_channels=seq_len, out_channels=out_len, kernel_size=(1,1), bias=True)
        self.trans_layer_s = nn.Conv2d(in_channels=d_model, out_channels=nodes, kernel_size=(1,1), bias=True)
        # self.norm_layer = nn.LayerNorm(d_model)

        self.nodes = nodes
        self.out_len = out_len

    def forward(self, x, st_i, x_mask, st_i_mask, adj_mats, **kwargs):
        # x [B,L,N,D]; st_i [B,L,N,D]; adj_mats [N,N,n_graphs]
        t_out = self.temporal_layer(x, x_mask, adj_mats, **kwargs)
        s_out = self.spatial_layer(st_i, t_out, st_i_mask, x_mask, adj_mats, **kwargs)

        # t_out [B,L,N,D]; s_out [B,L,N,D]
        t_out2 = self.trans_layer_t(t_out)
        s_out2 = self.trans_layer_s(s_out.transpose(1,-1)).transpose(1,-1)
        # t_out2 [B,OL,N,D]; s_out2 [B,L,N,N]
        dts = []
        for l in range(self.out_len):
            adj_mats_n = torch.cat([
                adj_mats.unsqueeze(0).expand(x.shape[0],-1,-1,-1), # [B,N,N,n_grpahs]
                s_out2[:,l,:,:].unsqueeze(-1), # [B,N,N,1]    
            ], dim=-1)
            dt = self.gcn_layers[l](t_out2[:,l,:,:].permute(0,2,1), adj_mats_n, **kwargs) # [B, D, N]
            dts.append(dt.transpose(2,1)) # [B, N, D]
        dts = torch.stack(dts, dim=1) # [B, L, N, D]

        return t_out, s_out, dts


class T2SLayer(BasicOp):
    def __init__(self, seq_len, out_len, d_model=512, n_heads=8, d_ff=8, nodes=207, 
                n_graphs=3, order=2, use_bn=True, dropout=0.1, activation='gelu',):
        super(T2SLayer, self).__init__()
        self._d_model = d_model
        self._n_heads = n_heads
        self._d_ff = d_ff
        
        self.t2s_layer = T2SModule(
            seq_len, out_len, d_model, n_heads, d_ff, nodes, 
            n_graphs, order, use_bn, dropout, activation
        )
        
    def forward(self, x, x_mark, attn_mask=None, adj_mats=None, **kwargs):
        # x [B, L, N, D]
        st = kwargs['st']; st_mask = kwargs['st_mask']
        out, st, dts = self.t2s_layer(x, st, attn_mask, st_mask, adj_mats, **kwargs) # [B, L, N, D]
        
        return dts, st
    
    @property
    def type(self):
        return 'T2SLayer'

    @property
    def setting(self):
        return [
            ('d_model', self._d_model),
            ('n_heads', self._n_heads),
            ('d_ff', self._d_ff),
        ]