r""" SPair-71k dataset """

import json
import glob
import os

import torch.nn.functional as F
import torch
from PIL import Image
import numpy as np

from .dataset import CorrespondenceDataset


class SPairDataset(CorrespondenceDataset):

    def __init__(self, benchmark, datapath, thres, split):
        r""" SPair-71k dataset constructor """
        super(SPairDataset, self).__init__(benchmark, datapath, thres, split)

        self.train_data = open(self.spt_path).read().split('\n')
        self.train_data = self.train_data[:len(self.train_data) - 1]
        self.src_imnames = list(map(lambda x: x.split('-')[1] + '.jpg', self.train_data))
        self.trg_imnames = list(map(lambda x: x.split('-')[2].split(':')[0] + '.jpg', self.train_data))
        self.seg_path = os.path.abspath(os.path.join(self.img_path, os.pardir, 'Segmentation'))
        self.cls = os.listdir(self.img_path)
        self.cls.sort()

        anntn_files = []
        for data_name in self.train_data:
            anntn_files.append(glob.glob('%s/%s.json' % (self.ann_path, data_name))[0])
        anntn_files = list(map(lambda x: json.load(open(x)), anntn_files))
        self.src_kps = list(map(lambda x: torch.tensor(x['src_kps']).t().float(), anntn_files))
        self.trg_kps = list(map(lambda x: torch.tensor(x['trg_kps']).t().float(), anntn_files))
        self.src_bbox = list(map(lambda x: torch.tensor(x['src_bndbox']).float(), anntn_files))
        self.trg_bbox = list(map(lambda x: torch.tensor(x['trg_bndbox']).float(), anntn_files))
        self.cls_ids = list(map(lambda x: self.cls.index(x['category']), anntn_files))

        self.vpvar = list(map(lambda x: torch.tensor(x['viewpoint_variation']), anntn_files))
        self.scvar = list(map(lambda x: torch.tensor(x['scale_variation']), anntn_files))
        self.trncn = list(map(lambda x: torch.tensor(x['truncation']), anntn_files))
        self.occln = list(map(lambda x: torch.tensor(x['occlusion']), anntn_files))

    def __getitem__(self, idx):
        r""" Construct and return a batch for SPair-71k dataset """
        sample = super(SPairDataset, self).__getitem__(idx)

        sample['src_mask'] = self.get_mask(sample, sample['src_imname'])
        sample['trg_mask'] = self.get_mask(sample, sample['trg_imname'])

        sample['src_bbox'] = self.get_bbox(self.src_bbox, idx, sample['src_imsize'])
        sample['trg_bbox'] = self.get_bbox(self.trg_bbox, idx, sample['trg_imsize'])
        sample['pckthres'] = self.get_pckthres(sample,  sample['trg_imsize'])

        sample['vpvar'] = self.vpvar[idx]
        sample['scvar'] = self.scvar[idx]
        sample['trncn'] = self.trncn[idx]
        sample['occln'] = self.occln[idx]

        return sample

    def get_mask(self, sample, imname):
        mask_path = os.path.join(self.seg_path, sample['category'], imname.split('.')[0] + '.png')

        tensor_mask = torch.tensor(np.array(Image.open(mask_path)))

        class_dict = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
                      'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9,
                      'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
                      'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}

        class_id = class_dict[sample['category']] + 1
        tensor_mask[tensor_mask != class_id] = 0
        tensor_mask[tensor_mask == class_id] = 255

        tensor_mask = F.interpolate(tensor_mask.unsqueeze(0).unsqueeze(0).float(),
                                    size=(self.img_size, self.img_size),
                                    mode='bilinear', align_corners=True).int().squeeze()

        return tensor_mask

    def get_image(self, img_names, idx):
        r""" Return image tensor """
        path = os.path.join(self.img_path, self.cls[self.cls_ids[idx]], img_names[idx])

        return Image.open(path).convert('RGB')

    def get_pckthres(self, sample, imsize):
        r""" Compute PCK threshold """
        return super(SPairDataset, self).get_pckthres(sample, imsize)

    def get_points(self, pts_list, idx, imsize):
        r""" Return key-points of an image """
        return super(SPairDataset, self).get_points(pts_list, idx, imsize)

    def match_idx(self, kps, n_pts):
        r""" Sample the nearst feature (receptive field) indices """
        return super(SPairDataset, self).match_idx(kps, n_pts)

    def get_bbox(self, bbox_list, idx, imsize):
        r""" Return object bounding-box """
        bbox = bbox_list[idx].clone()
        bbox[0::2] *= (self.img_size / imsize[0])
        bbox[1::2] *= (self.img_size / imsize[1])
        return bbox
