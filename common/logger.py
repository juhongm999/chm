r""" Logging """

import datetime
import logging
import os

from tensorboardX import SummaryWriter
import torch


class Logger:
    r""" Writes results of training/testing """
    @classmethod
    def initialize(cls, args, training):
        logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
        logpath = args.logpath if training else '_TEST_' + args.load.split('/')[-1].split('.')[0] + logtime
        if logpath == '': logpath = logtime

        cls.logpath = os.path.join('logs', logpath + '.log')
        cls.benchmark = args.benchmark
        os.makedirs(cls.logpath)

        logging.basicConfig(filemode='w',
                            filename=os.path.join(cls.logpath, 'log.txt'),
                            level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M:%S')

        # Console log config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # Tensorboard writer
        cls.tbd_writer = SummaryWriter(os.path.join(cls.logpath, 'tbd/runs'))

        # Log arguments
        if training:
            logging.info(':======== Convolutional Hough Matching Networks =========')
            for arg_key in args.__dict__:
                logging.info('| %20s: %-24s' % (arg_key, str(args.__dict__[arg_key])))
            logging.info(':========================================================\n')

    @classmethod
    def info(cls, msg):
        r""" Writes message to .txt """
        logging.info(msg)

    @classmethod
    def save_model(cls, model, epoch, val_pck):
        torch.save(model.state_dict(), os.path.join(cls.logpath, 'pck_best_model.pt'))
        cls.info('Model saved @%d w/ val. PCK: %5.2f.\n' % (epoch, val_pck))


class AverageMeter:
    r""" Stores loss, evaluation results, selected layers """
    def __init__(self, benchamrk):
        r""" Constructor of AverageMeter """
        self.buffer_keys = ['pck']
        self.buffer = {}
        for key in self.buffer_keys:
            self.buffer[key] = []

        self.loss_buffer = []

    def update(self, eval_result, loss=None):
        for key in self.buffer_keys:
            self.buffer[key] += eval_result[key]

        if loss is not None:
            self.loss_buffer.append(loss)

    def write_result(self, split, epoch):
        msg = '\n*** %s ' % split
        msg += '[@Epoch %02d] ' % epoch

        if len(self.loss_buffer) > 0:
            msg += 'Loss: %5.2f  ' % (sum(self.loss_buffer) / len(self.loss_buffer))

        for key in self.buffer_keys:
            msg += '%s: %6.2f  ' % (key.upper(), sum(self.buffer[key]) / len(self.buffer[key]))
        msg += '***\n'
        Logger.info(msg)

    def write_process(self, batch_idx, datalen, epoch):
        msg = '[Epoch: %02d] ' % epoch
        msg += '[Batch: %04d/%04d] ' % (batch_idx+1, datalen)
        if len(self.loss_buffer) > 0:
            msg += 'Loss: %5.2f  ' % self.loss_buffer[-1]
            msg += 'Avg Loss: %5.5f  ' % (sum(self.loss_buffer) / len(self.loss_buffer))

        for key in self.buffer_keys:
            msg += 'Avg %s: %5.2f  ' % (key.upper(), sum(self.buffer[key]) / len(self.buffer[key]) * 100)
        Logger.info(msg)

    def write_test_process(self, batch_idx, datalen):
        msg = '[Batch: %04d/%04d] ' % (batch_idx+1, datalen)

        for key in self.buffer_keys:
            if key == 'pck':
                pcks = torch.stack(self.buffer[key]).mean(dim=0) * 100
                val = ''
                for p in pcks:
                    val += '%5.2f   ' % p.item()
                msg += 'Avg %s: %s   ' % (key.upper(), val)
            else:
                msg += 'Avg %s: %5.2f  ' % (key.upper(), sum(self.buffer[key]) / len(self.buffer[key]))
        Logger.info(msg)

    def get_test_result(self):
        result = {}
        for key in self.buffer_keys:
            result[key] = torch.stack(self.buffer[key]).mean(dim=0) * 100

        return result
