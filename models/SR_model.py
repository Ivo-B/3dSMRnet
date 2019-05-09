import os
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import models.networks as networks
from models.modules import loss
from .base_model import BaseModel
from utils.util import AverageMeter

logger = logging.getLogger('base')




class SRModel(BaseModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)
        train_opt = opt['optim_config']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        self.load()

        if self.is_train:
            self.is_LRx2x4 = "LRx2x4" in opt['data_config']['train']['mode']
            self.netG.train()

            # loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l1charbonnier':
                self.cri_pix = loss.L1CharbonnierLoss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            beta1_G = train_opt['beta1_G'] if train_opt['beta1_G'] else 0.9
            beta2_G = train_opt['beta2_G'] if train_opt['beta2_G'] else 0.999
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(
                optim_params, lr=train_opt['lr_G'], weight_decay=wd_G, betas=(beta1_G, beta2_G))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(
                        optimizer, train_opt['lr_steps'], train_opt['lr_gamma'])
                    )
            elif train_opt['lr_scheme'] == 'StepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.StepLR(
                        optimizer, train_opt['lr_steps'], train_opt['lr_gamma'])
                    )
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()
            if self.is_LRx2x4:
                self.log_dict['l_sum_pix'] = AverageMeter()
                self.log_dict['lx2_pix'] = AverageMeter()
            self.log_dict['l_pix'] = AverageMeter()

        elif 'val' in opt['data_config']:
            self.is_LRx2x4 = "LRx2x4" in opt['data_config']['val']['mode']
        else:
            self.is_LRx2x4 = "LRx2x4" in opt['data_config']['test_2']['mode']
        # print network
        self.print_network()

    def feed_data(self, data, need_HR=True):
        if self.is_LRx2x4:
            self.var_Lx4 = data['LRx4'].to(self.device)  # LR
            self.var_Lx2 = data['LRx2'].to(self.device)  # LR
            if need_HR:
                self.real_H = data['HR'].to(self.device)  # HR
        else:
            self.var_L = data['LR'].to(self.device)  # LR
            if need_HR:
                self.real_H = data['HR'].to(self.device)  # HR
        self.hz = data['hz']

    def optimize_parameters(self, step):
        if self.is_LRx2x4:
            self.optimizer_G.zero_grad()
            self.fake_Hx2, self.fake_H = self.netG(self.var_Lx4)
            lx2_pix = self.l_pix_w * self.cri_pix(self.fake_Hx2, self.var_Lx2)
            l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
            l_sum_pix = lx2_pix + l_pix
            l_sum_pix.backward()
            self.optimizer_G.step()
            num = self.var_Lx4.size(0)
            self.log_dict['lx2_pix'].update(lx2_pix.item(), num)
            self.log_dict['l_sum_pix'].update(l_sum_pix.item(), num)
        else:
            self.optimizer_G.zero_grad()
            self.fake_H = self.netG(self.var_L)
            l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
            l_pix.backward()
            self.optimizer_G.step()
            num = self.var_L.size(0)

        # set log
        self.log_dict['l_pix'].update(l_pix.item(), num)

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            if self.is_LRx2x4:
                self.fake_Hx2, self.fake_H = self.netG(self.var_Lx4)
            else:
                self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def test_x8(self):
        # from https://github.com/thstkdgus35/EDSR-PyTorch
        self.netG.eval()
        for k, v in self.netG.named_parameters():
            v.requires_grad = False

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [self.var_L]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])
        sr_list = [self.netG(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        self.fake_H = output_cat.mean(dim=0, keepdim=True)

        for k, v in self.netG.named_parameters():
            v.requires_grad = True
        self.netG.train()

    def reset_log(self):
        for k in self.log_dict.keys():
            self.log_dict[k] = AverageMeter()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        if self.is_LRx2x4:
            out_dict['LRx4'] = self.var_Lx4.detach()[0].float().cpu()
            out_dict['LRx2'] = self.var_Lx2.detach()[0].float().cpu()
            out_dict['SRx2'] = self.fake_Hx2.detach()[0].float().cpu()
            out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        else:
            out_dict['LR'] = self.var_L.detach()[0].float().cpu()
            out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.real_H.detach()[0].float().cpu()
        out_dict['hz'] = self.hz
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def load(self):
        load_path_G = self.opt['run_config']['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
