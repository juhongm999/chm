r""" Convolutional Hough Matching Networks testing code """

import argparse

from torch.utils.data import DataLoader
import torch

from model.base.geometry import Geometry
from common.evaluation import Evaluator
from common.logger import AverageMeter
from common.logger import Logger
from data import download
from model import chmnet


def test(model, dataloader):
    average_meter = AverageMeter(dataloader.dataset.benchmark)

    model.eval()
    for idx, batch in enumerate(dataloader):

        # 1. CHMNet forward pass
        corr_matrix = model(batch['src_img'].cuda(), batch['trg_img'].cuda())

        # 2. Transfer key-points
        prd_kps = Geometry.transfer_kps(corr_matrix, batch['src_kps'].cuda(), batch['n_pts'].cuda(), normalized=False)

        # 3. Evaluate predictions
        eval_result = Evaluator.evaluate(Geometry.unnormalize_kps(prd_kps), batch)
        average_meter.update(eval_result)
        average_meter.write_test_process(idx, len(dataloader))

    return average_meter.get_test_result()


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Convolutional Hough Matching Network Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='../Datasets_CHM')
    parser.add_argument('--benchmark', type=str, default='', choices=['pfpascal', 'pfwillow', 'spair'])
    parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
    parser.add_argument('--bsz', type=int, default=100)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--ktype', type=str, default='psi', choices=['psi', 'iso', 'full'])
    parser.add_argument('--alpha', nargs='+', type=float, default=[0.05, 0.1])
    parser.add_argument('--img_size', type=int, default=240)
    args = parser.parse_args()
    Logger.initialize(args, training=False)

    # Model initialization
    model = chmnet.CHMNet(args.ktype).cuda()
    model.load_state_dict(torch.load(args.load))
    Evaluator.initialize(args.alpha)
    Geometry.initialize(img_size=args.img_size)

    # Dataset initialization
    download.download_dataset(args.datapath, args.benchmark)
    test_ds = download.load_dataset(args.benchmark, args.datapath, args.thres, 'test')
    test_dl = DataLoader(test_ds, batch_size=args.bsz, shuffle=False)

    # Test CHMNet
    Logger.info('Evaluating %s...' % args.benchmark)
    with torch.no_grad(): result = test(model, test_dl)
    for alpha, pck in zip(args.alpha, result['pck']):
        Logger.info('PCK at alpha=%.2f: %.2f' % (alpha, pck))
    Logger.info('==================== Finished testing ====================')

