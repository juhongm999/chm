r""" Conovlutional Hough matching layers """

import torch.nn as nn
import torch

from .base.correlation import Correlation
from .base.geometry import Geometry
from .base.chm import CHM4d, CHM6d


class CHMLearner(nn.Module):

    def __init__(self, ktype, feat_dim):
        super(CHMLearner, self).__init__()

        # Scale-wise feature transformation
        self.scales = [0.5, 1, 2]
        self.conv2ds = nn.ModuleList([nn.Conv2d(feat_dim, feat_dim // 4, kernel_size=3, padding=1, bias=False) for _ in self.scales])

        # CHM layers
        ksz_translation = 5
        ksz_scale = 3
        self.chm6d = CHM6d(1, 1, ksz_scale, ksz_translation, ktype)
        self.chm4d = CHM4d(1, 1, ksz_translation, ktype, bias=True)

        # Activations
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self, src_feat, trg_feat):

        corr = Correlation.build_correlation6d(src_feat, trg_feat, self.scales, self.conv2ds).unsqueeze(1)
        bsz, ch, s, s, h, w, h, w = corr.size()

        # CHM layer (6D)
        corr = self.chm6d(corr)
        corr = self.sigmoid(corr)

        # Scale-space maxpool
        corr = corr.view(bsz, -1, h, w, h, w).max(dim=1)[0]
        corr = Geometry.interpolate4d(corr, [h * 2, w * 2]).unsqueeze(1)

        # CHM layer (4D)
        corr = self.chm4d(corr).squeeze(1)

        # To ensure non-negative vote scores & soft cyclic constraints
        corr = self.softplus(corr)
        corr = Correlation.mutual_nn_filter(corr.view(bsz, corr.size(-1) ** 2, corr.size(-1) ** 2).contiguous())

        return corr

