r""" Convolutional Hough Matching Networks training & validation code """

import argparse

from torch.utils.data import DataLoader
import torch.optim as optim
import torch

from model.base.geometry import Geometry
from common.evaluation import Evaluator
from common.logger import AverageMeter
from common.logger import Logger
from data import download
from model import chmnet


def train(epoch, model, dataloader, optimizer, training):

    model.train() if training else model.eval()
    average_meter = AverageMeter(dataloader.dataset.benchmark)

    for idx, batch in enumerate(dataloader):

        # 1. CHMNet forward pass
        corr_matrix = model(batch['src_img'].cuda(), batch['trg_img'].cuda())
        prd_trg_kps = Geometry.transfer_kps(corr_matrix, batch['src_kps'].cuda(), batch['n_pts'].cuda(), normalized=False)

        # 2. Evaluate predictions
        eval_result = Evaluator.evaluate(Geometry.unnormalize_kps(prd_trg_kps), batch)

        # 3. CHMNet backward pass
        loss = model.training_objective(prd_trg_kps, Geometry.normalize_kps(batch['trg_kps'].cuda()), batch['n_pts'].cuda())
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        average_meter.update(eval_result, loss.item())
        average_meter.write_process(idx, len(dataloader), epoch)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)

    mean = lambda x: sum(x) / len(x)
    avg_loss = mean(average_meter.loss_buffer)
    avg_pck = mean(average_meter.buffer['pck'])
    return avg_loss, avg_pck


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Convolutional Hough Matching Network Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='../Datasets_CHM')
    parser.add_argument('--benchmark', type=str, default='pfpascal', choices=['pfpascal', 'spair'])
    parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--backbone_lr', type=float, default=1e-5)
    parser.add_argument('--bsz', type=int, default=20)
    parser.add_argument('--nworker', type=int, default=8)
    parser.add_argument('--ktype', type=str, default='psi', choices=['psi', 'iso', 'full'])
    parser.add_argument('--img_size', type=int, default=240)
    args = parser.parse_args()
    Logger.initialize(args, training=True)

    # Model initialization
    model = chmnet.CHMNet(args.ktype).cuda()
    optimizer = optim.Adam([{"params": model.learner.parameters(), "lr": args.lr},
                            {"params": model.backbone.parameters(), "lr": args.backbone_lr}])
    Evaluator.initialize([args.alpha])
    Geometry.initialize(img_size=args.img_size)

    # Dataset initialization
    download.download_dataset(args.datapath, args.benchmark)
    trn_ds = download.load_dataset(args.benchmark, args.datapath, args.thres, 'trn')
    val_ds = download.load_dataset(args.benchmark, args.datapath, args.thres, 'val')
    trn_dl = DataLoader(trn_ds, batch_size=args.bsz, num_workers=args.nworker, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.bsz, num_workers=0, shuffle=False)

    # Train CHMNet
    best_val_pck = float('-inf')
    best_val_loss = float('inf')
    for epoch in range(1000):

        trn_loss, trn_pck = train(epoch, model, trn_dl, optimizer, training=True)
        with torch.no_grad():
            val_loss, val_pck = train(epoch, model, val_dl, optimizer, training=False)

        # Save the best model
        if val_pck > best_val_pck:
            best_val_pck = val_pck
            Logger.save_model(model, epoch, val_pck)
        Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
        Logger.tbd_writer.add_scalars('data/pck', {'trn_pck': trn_pck, 'val_pck': val_pck}, epoch)
        Logger.tbd_writer.flush()

    Logger.tbd_writer.close()
    Logger.info('==================== Finished training ====================')
