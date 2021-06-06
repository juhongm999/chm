r""" CHM 4D kernel (psi, iso, and full) generator """

import torch

from .geometry import Geometry


class KernelGenerator:
    def __init__(self, ksz, ktype):
        self.ksz = ksz
        self.idx4d = Geometry.init_idx4d(ksz)
        self.kernel = torch.zeros((ksz, ksz, ksz, ksz)).cuda()
        self.center = (ksz // 2, ksz // 2)
        self.ktype = ktype

    def quadrant(self, crd):
        if crd[0] < self.center[0]:
            horz_quad = -1
        elif crd[0] < self.center[0]:
            horz_quad = 1
        else:
            horz_quad = 0

        if crd[1] < self.center[1]:
            vert_quad = -1
        elif crd[1] < self.center[1]:
            vert_quad = 1
        else:
            vert_quad = 0

        return horz_quad, vert_quad

    def generate(self):
        return None if self.ktype == 'full' else self.generate_chm_kernel()

    def generate_chm_kernel(self):
        param_dict = {}
        for idx in self.idx4d:
            src_i, src_j, trg_i, trg_j = idx
            d_tail = Geometry.get_distance((src_i, src_j), self.center)
            d_head = Geometry.get_distance((trg_i, trg_j), self.center)
            d_off = Geometry.get_distance((src_i, src_j), (trg_i, trg_j))
            horz_quad, vert_quad = self.quadrant((src_j, src_i))

            src_crd = (src_i, src_j)
            trg_crd = (trg_i, trg_j)

            key = self.build_key(horz_quad, vert_quad, d_head, d_tail, src_crd, trg_crd, d_off)
            coord1d = Geometry.get_coord1d((src_i, src_j, trg_i, trg_j), self.ksz)

            if param_dict.get(key) is None: param_dict[key] = []
            param_dict[key].append(coord1d)

        return param_dict

    def build_key(self, horz_quad, vert_quad, d_head, d_tail, src_crd, trg_crd, d_off):

        if self.ktype == 'iso':
            return '%d' % d_off
        elif self.ktype == 'psi':
            d_max = max(d_head, d_tail)
            d_min = min(d_head, d_tail)
            return '%d_%d_%d' % (d_max, d_min, d_off)
        else:
            raise Exception('not implemented.')

