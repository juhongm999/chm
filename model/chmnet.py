r""" Convolutional Hough Matching Networks """

import torch.nn as nn
import torch

from . import chmlearner as chmlearner
from .base import backbone


class CHMNet(nn.Module):
    def __init__(self, ktype):
        super(CHMNet, self).__init__()

        self.backbone = backbone.resnet101(pretrained=True)
        self.learner = chmlearner.CHMLearner(ktype, feat_dim=1024)

    def forward(self, src_img, trg_img):
        src_feat, trg_feat = self.extract_features(src_img, trg_img)
        correlation  = self.learner(src_feat, trg_feat)
        return correlation

    def extract_features(self, src_img, trg_img):
        feat = self.backbone.conv1.forward(torch.cat([src_img, trg_img], dim=1))
        feat = self.backbone.bn1.forward(feat)
        feat = self.backbone.relu.forward(feat)
        feat = self.backbone.maxpool.forward(feat)

        for idx in range(1, 5):
            feat = self.backbone.__getattr__('layer%d' % idx)(feat)

            if idx == 3:
                src_feat = feat.narrow(1, 0, feat.size(1) // 2).clone()
                trg_feat = feat.narrow(1, feat.size(1) // 2, feat.size(1) // 2).clone()
                return src_feat, trg_feat

    def training_objective(cls, prd_kps, trg_kps, npts):
        l2dist = (prd_kps - trg_kps).pow(2).sum(dim=1)
        loss = []
        for dist, npt in zip(l2dist, npts):
            loss.append(dist[:npt].mean())
        return torch.stack(loss).mean()

