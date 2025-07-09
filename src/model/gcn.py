import torch
import torch.nn as nn

class gconv(nn.Module):
    def __init__(self):
        super(gconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()

class GraphConv(nn.Module):
    def __init__(self, c_in, c_out, n_graphs, order, use_bn=True, dropout=0.1):
        super(GraphConv,self).__init__()
        c_in = (order * n_graphs + 1) * c_in

        self.n_graphs = n_graphs
        self.order = order
        self.use_bn = use_bn

        self.gconv = gconv()
        self.linear = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), 
                                    padding=(0,0), stride=(1,1), bias=True)
        self.dropout = nn.Dropout(dropout)
        if use_bn:
            self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x, adj_mats, **kwargs):
        # x: [B, D, N, T]
        out = [x]

        for i in range(self.n_graphs):
            y = x
            for j in range(self.order):
                y = self.gconv(y, adj_mats[:, :, i].squeeze())
                out += [y]
            
        x = torch.cat(out, dim=1)
        x = self.linear(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.dropout(x)
        return x # [B, D, N, T]

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('bdn,bnm->bdm', (x, A))
        return x.contiguous()

class NGraphConv(nn.Module):
    def __init__(self, c_in, c_out, n_graphs, order, use_bn=True, dropout=0.1):
        super(NGraphConv,self).__init__()
        c_in = (order * n_graphs + 1) * c_in

        self.n_graphs = n_graphs
        self.order = order
        self.use_bn = use_bn

        self.nconv = nconv()
        self.linear = torch.nn.Conv1d(c_in, c_out, kernel_size=1, 
                                    padding=0, stride=1, bias=True)
        self.dropout = nn.Dropout(dropout)
        if use_bn:
            self.bn = nn.BatchNorm1d(c_out)

    def forward(self, x, adj_mats, **kwargs):
        # x: [B, D, N]
        out = [x]

        for i in range(self.n_graphs):
            y = x
            for j in range(self.order):
                y = self.nconv(y, adj_mats[:, :, :, i].squeeze())
                out += [y]
            
        x = torch.cat(out, dim=1)
        x = self.linear(x)
        
        if self.use_bn:
            x = self.bn(x)
        x = self.dropout(x)
        return x # [B, D, N]