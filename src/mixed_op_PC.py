import torch
import torch.nn as nn
from model.mixed_op import MixedOp
from model.candidate_op import BasicOp, create_op

# PC-DARTS

def channel_shuffle(x, groups):
    """
    https://github.com/yuhuixu1993/PC-DARTS/blob/86446d1b6bbbd5f752cc60396be13d2d5737a081/model_search.py#L9
    """

    batchsize, height, width, num_channels = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, height, width, groups,
               channels_per_group)

    x = torch.transpose(x, 3, 4).contiguous()

    # flatten
    x = x.view(batchsize, height, width, -1)
    return x


class MixedOpPCDARTS(MixedOp):
    """
    Adapted from PCDARTS:
    https://github.com/yuhuixu1993/PC-DARTS/blob/86446d1b6bbbd5f752cc60396be13d2d5737a081/model_search.py#L25
    """

    def __init__(self, candidate_op_profiles):
        super(MixedOp, self).__init__()

        self._num_ops = len(candidate_op_profiles)
        self._candidate_op_profiles = candidate_op_profiles
        self._candidate_ops = nn.ModuleList()

        for (op_name, profile) in self._candidate_op_profiles:
            # profile['d_model'] = profile['d_model'] // 4
            print(profile)
            self._candidate_ops += [create_op(op_name, profile)]

    def forward(self, x, x_mark, attn_mask=None, adj_mats=None, weights=None, **kwargs):
        probs = weights
        t_out = 0.;
        s_out = 0.

        dim_2 = x.shape[3]

        x_remain = x[:, :, :, dim_2 // 4:]
        
        for i, idx in enumerate(self._sample_idx):
            if adj_mats is not None:  # [N,N,r_graphs+num_ops]
                # idx_count = len(self._candidate_op_profiles)
                if self._candidate_op_profiles[idx][0] in ['Identity', 'Zero']:
                    op_adj_mats = None
                else:
                    r_graphs = kwargs['r_graphs']
                    op_adj_mats = torch.cat([adj_mats[:, :, :r_graphs], adj_mats[:, :, r_graphs + idx].unsqueeze(-1)],
                                            dim=-1)
            out, st = self._candidate_ops[idx](x, x_mark, attn_mask, op_adj_mats, **kwargs)

            t_out += probs[i] * out
            s_out += probs[i] * st

        x_new = t_out[:, :, :, : dim_2 // 4]
        ans = torch.cat([x_new, x_remain], dim=3)
        ans = channel_shuffle(ans, 4)
        return ans, s_out

'''
    def __init__(self, C, stride):
        super(MixedOpPCDARTS, self).__init__(C, stride)
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C // 4, stride, False)
            self._ops.append(op)

    def forward(self, x, weights):
        # channel proportion k=4
        dim_2 = x.shape[3]
        xtemp = x[:, :, :, :  dim_2 // 4]
        xtemp2 = x[:, :, :, dim_2 // 4:]
        temp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops))
        ans = torch.cat([temp1, xtemp2], dim=3)
        ans = channel_shuffle(ans, 4)
        return ans
'''