r""" Evaluates CHMNet with PCK """

import torch


class Evaluator:
    r""" Computes evaluation metrics of PCK """
    @classmethod
    def initialize(cls, alpha):
        cls.alpha = torch.tensor(alpha).unsqueeze(1).cuda()

    @classmethod
    def evaluate(cls, prd_kps, batch):
        r""" Compute percentage of correct key-points (PCK) with multiple alpha {0.05, 0.1, 0.15 }"""

        pcks = []
        for idx, (pk, tk) in enumerate(zip(prd_kps, batch['trg_kps'])):
            pckthres = batch['pckthres'][idx].cuda()
            npt = batch['n_pts'][idx]
            prd_kps = pk[:, :npt].cuda()
            trg_kps = tk[:, :npt].cuda()

            l2dist = (prd_kps - trg_kps).pow(2).sum(dim=0).pow(0.5).unsqueeze(0).repeat(len(cls.alpha), 1)
            thres = pckthres.expand_as(l2dist).float() * cls.alpha
            pck = torch.le(l2dist, thres).sum(dim=1) / float(npt)
            if len(pck) == 1: pck = pck[0]
            pcks.append(pck)

        eval_result = {'pck': pcks}

        return eval_result

