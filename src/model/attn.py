import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask
from model.gcn import GraphConv


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class STLinearAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1):
        super(STLinearAttention, self).__init__()
        self.feature_map = elu_feature_map # feature_map or elu_feature_map
        self.eps = scale or 1e-6
        
    def forward(self, queries, keys, values, attn_mask):
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        KV = torch.einsum("bnshd,bnshm->bnhmd", K, values)
        Z = 1/(torch.einsum("bnlhd,bnhd->bnlh", Q, K.sum(dim=2))+self.eps)
        V = torch.einsum("bnlhd,bnhmd,bnlh->bnlhm", Q, KV, Z)
        
        return V.contiguous()


class STFullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1):
        super(STFullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, N, L, H, E = queries.shape
        _, _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("bnlhd,bnshd->bnhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # Compute the attention and the weighted average
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bnhls,bnshd->bnlhd", A, values)

        # Make sure that what we return is contiguous
        return V.contiguous()


class STSLinearAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1):
        super(STSLinearAttention, self).__init__()
        self.feature_map = elu_feature_map # feature_map or elu_feature_map
        self.eps = scale or 1e-6
        
    def forward(self, queries, keys, values, attn_mask):
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        KV = torch.einsum("bnhd,bnhm->bhmd", K, values)
        Z = 1/(torch.einsum("bnhd,bhd->bnh", Q, K.sum(dim=1))+self.eps)
        V = torch.einsum("bnhd,bhmd,bnh->bnhm", Q, KV, Z)
        
        return V.contiguous()


class S2TAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None,
                 n_graphs=3, order=2, use_bn=True, dropout=0.1):
        super(S2TAttentionLayer, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        
        self.gcn1 = GraphConv(d_model, d_model, n_graphs, order, use_bn, dropout)
        self.gcn2 = GraphConv(d_model, d_model, n_graphs, order, use_bn, dropout)

    def forward(self, queries, keys, values, attn_mask, adj_mats, **kwargs):
        # Extract the dimensions into local variables
        B, L, N1, _ = queries.shape
        _, S, N2, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(B, L, N1, H, -1)
        gcn_keys = self.gcn1(keys.transpose(-1, 1), adj_mats, **kwargs) # [B, D, N, T]
        keys = self.key_projection(gcn_keys.transpose(-1, 1)).view(B, S, N2, H, -1) # [B, S, N, H, d]
        gcn_values = self.gcn2(values.transpose(-1, 1), adj_mats, **kwargs) # [B, D, N, T]
        values = self.value_projection(gcn_values.transpose(-1, 1)).view(B, S, N2, H, -1) # [B, S, N, H, d]
        
        queries = queries.transpose(2, 1) # [B, N, L, H, d]
        keys = keys.transpose(2, 1) # [B, N, S, H, d]
        values = values.transpose(2, 1)
        
        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        ).view(B, N1, L, -1)
        
        new_values = new_values.transpose(2, 1) # [B, L, N1, D]
        
        # Project the output and return
        return self.out_projection(new_values)


class STSAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None,
                 n_graphs=3, order=2, use_bn=True, dropout=0.1):
        super(STSAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, adj_mats, **kwargs):
        # Extract the dimensions into local variables
        B, L, N1, _ = queries.shape
        _, S, N2, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(B, L*N1, H, -1) # [B, L*N, H, d]
        keys = self.key_projection(keys).view(B, S*N2, H, -1)
        values = self.value_projection(values).view(B, S*N2, H, -1)
        
        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        ).view(B, L, N1, -1)

        # Project the output and return
        return self.out_projection(new_values)


class STSGAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None,
                 n_graphs=3, order=2, use_bn=True, dropout=0.1):
        super(STSGAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads * 2, d_model)
        self.n_heads = n_heads

        self.gcn = GraphConv(d_model, d_model, n_graphs, order, use_bn, dropout)

    def forward(self, queries, keys, values, attn_mask, adj_mats, **kwargs):
        # Extract the dimensions into local variables
        B, L, N1, _ = queries.shape
        _, S, N2, _ = keys.shape
        H = self.n_heads

        gcn_values = self.gcn(values.transpose(-1, 1), adj_mats, **kwargs) # [B, D, N, T]
        gcn_values = gcn_values.transpose(-1, 1) # [B, T, N, D]

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(B, L*N1, H, -1) # [B, L*N, H, d]
        keys = self.key_projection(keys).view(B, S*N2, H, -1)
        values = self.value_projection(values).view(B, S*N2, H, -1)

        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        ).view(B, L, N1, -1)

        new_values = torch.cat([new_values, gcn_values], dim=-1)

        # Project the output and return
        return self.out_projection(new_values)


class T2SAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None,
                 n_graphs=3, order=2, use_bn=True, dropout=0.1):
        super(T2SAttentionLayer, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, adj_mats, **kwargs):
        # Extract the dimensions into local variables
        B, L, N1, _ = queries.shape
        _, S, N2, _ = keys.shape # [B, S, N, D]
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(B, L, N1, H, -1)
        keys = self.key_projection(keys).view(B, S, N2, H, -1)
        values = self.value_projection(values).view(B, S, N2, H, -1)

        queries = queries.transpose(2,1) # [B, N, L, H, d]
        keys = keys.transpose(2,1) # [B, N, S, H, d]
        values = values.transpose(2,1)
        
        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
        ).view(B, N1, L, -1)

        new_values = new_values.transpose(2,1) # [B, L, N1, D]
        
        # Project the output and return
        return self.out_projection(new_values)


class T2SGAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None,
                 n_graphs=3, order=2, use_bn=True, dropout=0.1):
        super(T2SGAttentionLayer, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

        self.gcn = GraphConv(d_model, d_model, n_graphs, order, use_bn, dropout)

    def forward(self, queries, keys, values, attn_mask, adj_mats, **kwargs):
        # Extract the dimensions into local variables
        B, L, N1, _ = queries.shape
        _, S, N2, _ = keys.shape # [B, S, N, D]
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(B, L, N1, H, -1)
        keys = self.key_projection(keys).view(B, S, N2, H, -1)
        values = self.value_projection(values).view(B, S, N2, H, -1)

        queries = queries.transpose(2,1) # [B, N, L, H, d]
        keys = keys.transpose(2,1) # [B, N, S, H, d]
        values = values.transpose(2,1)
        
        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
        ).view(B, N1, L, -1)

        new_values = self.gcn(new_values.permute(0, 3, 1, 2), adj_mats, **kwargs) # [B, D, N, L]
        new_values = new_values.transpose(-1, 1) # [B, L, N1, D]
        
        # Project the output and return
        return self.out_projection(new_values)