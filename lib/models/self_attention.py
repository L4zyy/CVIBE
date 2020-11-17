import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class SelfAttention(nn.Module):

    def __init__(self, input_size=2048, nhead=8,
         num_layers=6, dim_feedforward=2048, dropout=0.1,
         activation="relu", normalize_before=False,
         ):
        super().__init__()


        self.num_layers = num_layers
        layer = TransformerEncoderLayer(input_size, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.layers = _get_clones(layer, num_layers)

        self.norm = nn.LayerNorm(input_size) if normalize_before else None

    def forward(self, src, pos_embed):

        seq_len, bs, f = src.shape

        output = src

        for layer in self.layers:
            output = layer(output, pos=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(nn.Module):

    def __init__(self, input_size, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(input_size, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(input_size, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, input_size)

        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(input_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, pos)
        return self.forward_post(src, pos)