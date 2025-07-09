import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv2d(in_channels=d_model, out_channels=d_ff, kernel_size=(1,1)) # nn.Linear(d_model, d_ff)
        self.conv2 = nn.Conv2d(in_channels=d_ff, out_channels=d_model, kernel_size=(1,1)) # nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask, adj_mats, **kwargs):
        # x [B, L, N, D]
        x = x + self.dropout(self.attention(
            x, x, x,
            attn_mask,
            adj_mats, 
            **kwargs
        ))

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1)))) # [B, D, N, L]
        y = self.dropout(self.conv2(y)).transpose(-1, 1) # [B, L, N, D]

        return self.norm2(x+y)

class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask, adj_mats, **kwargs):
        # x [B, L, N, D]
        for layer in self.layers:
            x = layer(x, attn_mask, adj_mats, **kwargs)

        # Apply the normalization if needed
        if self.norm is not None:
            x = self.norm(x)

        return x