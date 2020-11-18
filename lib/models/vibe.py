# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F

from lib.core.config import VIBE_DATA_DIR
from lib.models.spin import Regressor, hmr

import math
from lib.models.self_attention import SelfAttention


class TemporalEncoder(nn.Module):
    def __init__(
            self,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True,
            temporal_type='gru',
            seqlen=16,
            n_head=0,
    ):
        super(TemporalEncoder, self).__init__()

        self.temporal_type = temporal_type

        self.input_size = 2048

        if temporal_type == 'gru':
            self.gru = nn.GRU(
                input_size=2048,
                hidden_size=hidden_size,
                bidirectional=bidirectional,
                num_layers=n_layers
            )

            self.linear = None
            if bidirectional:
                self.linear = nn.Linear(hidden_size*2, 2048)
            elif add_linear:
                self.linear = nn.Linear(hidden_size, 2048)

        elif temporal_type == 'self_attn':
            self.self_attn = SelfAttention(self.input_size, nhead=n_head, num_layers=n_layers)

            self.pos_embed = torch.zeros(seqlen, self.input_size).to('cuda')
            for pos in range(seqlen):
                for i in range(0, self.input_size, 2):
                    self.pos_embed[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i)/self.input_size)))
                    self.pos_embed[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1))/self.input_size)))

            self.linear = None
            if add_linear:
                self.linear = nn.Linear(self.input_size, 2048)

        self.use_residual = use_residual

    def forward(self, x):
        n,t,f = x.shape
        x = x.permute(1,0,2) # NTF -> TNF
        if self.temporal_type == 'gru':
            y, _ = self.gru(x)
        elif self.temporal_type == 'self_attn':
            y = self.self_attn(x, self.pos_embed.unsqueeze(1).repeat(1, n, 1))
        if self.linear:
            y = F.relu(y)
            y = self.linear(y.view(-1, y.size(-1)))
            y = y.view(t,n,f)
        if self.use_residual and y.shape[-1] == 2048:
            y = y + x
        y = y.permute(1,0,2) # TNF -> NTF
        return y


class VIBE(nn.Module):
    def __init__(
            self,
            seqlen,
            batch_size=64,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True,
            pretrained=osp.join(VIBE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),
            temporal_type='gru',
            n_head=0,
    ):

        super(VIBE, self).__init__()

        self.seqlen = seqlen
        self.batch_size = batch_size

        self.encoder = TemporalEncoder(
            n_layers=n_layers,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            add_linear=add_linear,
            use_residual=use_residual,
            temporal_type=temporal_type,
            seqlen=self.seqlen,
            n_head=n_head,
        )

        # regressor can predict cam, pose and shape params in an iterative way
        self.regressor = Regressor()

        if pretrained and os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)['model']

            self.regressor.load_state_dict(pretrained_dict, strict=False)
            print(f'=> loaded pretrained model from \'{pretrained}\'')


    def forward(self, input, J_regressor=None):
        # input size NTF
        batch_size, seqlen = input.shape[:2]

        feature = self.encoder(input)
        feature = feature.reshape(-1, feature.size(-1))

        smpl_output = self.regressor(feature, J_regressor=J_regressor)
        for s in smpl_output:
            s['theta'] = s['theta'].reshape(batch_size, seqlen, -1)
            s['verts'] = s['verts'].reshape(batch_size, seqlen, -1, 3)
            s['kp_2d'] = s['kp_2d'].reshape(batch_size, seqlen, -1, 2)
            s['kp_3d'] = s['kp_3d'].reshape(batch_size, seqlen, -1, 3)
            s['rotmat'] = s['rotmat'].reshape(batch_size, seqlen, -1, 3, 3)

        return smpl_output


class VIBE_Demo(nn.Module):
    def __init__(
            self,
            seqlen,
            batch_size=64,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True,
            pretrained=osp.join(VIBE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),
    ):

        super(VIBE_Demo, self).__init__()

        self.seqlen = seqlen
        self.batch_size = batch_size

        self.encoder = TemporalEncoder(
            n_layers=n_layers,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            add_linear=add_linear,
            use_residual=use_residual,
        )

        self.hmr = hmr()
        checkpoint = torch.load(pretrained)
        self.hmr.load_state_dict(checkpoint['model'], strict=False)

        # regressor can predict cam, pose and shape params in an iterative way
        self.regressor = Regressor()

        if pretrained and os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)['model']

            self.regressor.load_state_dict(pretrained_dict, strict=False)
            print(f'=> loaded pretrained model from \'{pretrained}\'')


    def forward(self, input, J_regressor=None):
        # input size NTF
        batch_size, seqlen, nc, h, w = input.shape

        feature = self.hmr.feature_extractor(input.reshape(-1, nc, h, w))

        feature = feature.reshape(batch_size, seqlen, -1)
        feature = self.encoder(feature)
        feature = feature.reshape(-1, feature.size(-1))

        smpl_output = self.regressor(feature, J_regressor=J_regressor)

        for s in smpl_output:
            s['theta'] = s['theta'].reshape(batch_size, seqlen, -1)
            s['verts'] = s['verts'].reshape(batch_size, seqlen, -1, 3)
            s['kp_2d'] = s['kp_2d'].reshape(batch_size, seqlen, -1, 2)
            s['kp_3d'] = s['kp_3d'].reshape(batch_size, seqlen, -1, 3)
            s['rotmat'] = s['rotmat'].reshape(batch_size, seqlen, -1, 3, 3)

        return smpl_output
