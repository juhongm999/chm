r""" Provides functions that creates/manipulates correlation matrices """

import math

from torch.nn.functional import interpolate as resize
import torch

from .geometry import Geometry


class Correlation:

    @classmethod
    def mutual_nn_filter(cls, correlation_matrix, eps=1e-30):
        r""" Mutual nearest neighbor filtering (Rocco et al. NeurIPS'18 )"""
        corr_src_max = torch.max(correlation_matrix, dim=2, keepdim=True)[0]
        corr_trg_max = torch.max(correlation_matrix, dim=1, keepdim=True)[0]
        corr_src_max[corr_src_max == 0] += eps
        corr_trg_max[corr_trg_max == 0] += eps

        corr_src = correlation_matrix / corr_src_max
        corr_trg = correlation_matrix / corr_trg_max

        return correlation_matrix * (corr_src * corr_trg)

    @classmethod
    def build_correlation6d(self, src_feat, trg_feat, scales, conv2ds):
        r""" Build 6-dimensional correlation tensor """

        bsz, _, side, side = src_feat.size()

        # Construct feature pairs with multiple scales
        _src_feats = []
        _trg_feats = []
        for scale, conv in zip(scales, conv2ds):
            s = (round(side * math.sqrt(scale)),) * 2
            _src_feat = conv(resize(src_feat, s, mode='bilinear', align_corners=True))
            _trg_feat = conv(resize(trg_feat, s, mode='bilinear', align_corners=True))
            _src_feats.append(_src_feat)
            _trg_feats.append(_trg_feat)

        # Build multiple 4-dimensional correlation tensor
        corr6d = []
        for src_feat in _src_feats:
            ch = src_feat.size(1)

            src_side = src_feat.size(-1)
            src_feat = src_feat.view(bsz, ch, -1).transpose(1, 2)
            src_norm = src_feat.norm(p=2, dim=2, keepdim=True)

            for trg_feat in _trg_feats:
                trg_side = trg_feat.size(-1)
                trg_feat = trg_feat.view(bsz, ch, -1)
                trg_norm = trg_feat.norm(p=2, dim=1, keepdim=True)

                correlation = torch.bmm(src_feat, trg_feat) / torch.bmm(src_norm, trg_norm)
                correlation = correlation.view(bsz, src_side, src_side, trg_side, trg_side).contiguous()
                corr6d.append(correlation)

        # Resize the spatial sizes of the 4D tensors to the same size
        for idx, correlation in enumerate(corr6d):
            corr6d[idx] = Geometry.interpolate4d(correlation, [side, side])

        # Build 6-dimensional correlation tensor
        corr6d = torch.stack(corr6d).view(len(scales), len(scales),
                             bsz, side, side, side, side).permute(2, 0, 1, 3, 4, 5, 6)
        return corr6d.clamp(min=0)

