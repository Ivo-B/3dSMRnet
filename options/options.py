import os
import os.path as osp
import logging
from collections import OrderedDict
import json

import re


def parse(opt_path, is_train=True, is_tensorboard_available=True):
    # remove comments starting with '//'
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    if not is_tensorboard_available:
        opt['run_config']['use_tb_logger'] = False
        print('TensorboardX support not available')
    opt['run_config']['id'] = opt['run_config']['id'] + '_' + opt['run_config']['num']
    opt['run_config']['is_train'] = is_train

    # datasets
    for phase, dataset in opt['data_config'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        dataset['scale'] = opt['model_config']['scale']
        dataset['dataroot_HR'] = os.path.expanduser(dataset['dataroot_HR']) if dataset['dataroot_HR'] else None

        if re.search(r"x\d", dataset['mode']):
            scale_data = re.findall(r"x\d", dataset['mode'])
            for scaling in scale_data:
                key = 'dataroot_LR'+scaling
                dataset[key] = os.path.expanduser(dataset[key])
        else:
            dataset['dataroot_LR'] = os.path.expanduser(dataset['dataroot_LR'])
        dataset['data_type'] = 'hdf5'

    if is_train:
        experiments_root = os.path.join(opt['run_config']['path']['root'], 'experiments', opt['run_config']['id'])
        opt['run_config']['path']['experiments_root'] = experiments_root
        opt['run_config']['path']['models'] = os.path.join(experiments_root, 'models')
        opt['run_config']['path']['training_state'] = os.path.join(experiments_root, 'training_state')
        opt['run_config']['path']['log'] = experiments_root
        opt['run_config']['path']['val_images'] = os.path.join(experiments_root, 'val_images')

        # change some options for debug mode
        if 'debug' in opt['run_config']['id']:
            opt['run_config']['logger']['print_freq'] = 2
            opt['run_config']['logger']['save_checkpoint_freq'] = 8
            opt['optim_config']["val_freq"] = 5
    else:  # test
        results_root = os.path.join(opt['run_config']['path']['root'], 'results', opt['run_config']['id'])
        opt['run_config']['path']['results_root'] = results_root
        opt['run_config']['path']['log'] = results_root

    #num_channels = int(opt['model_config']['network_G']['in_nc'])
    #LR_size = int(opt['data_config']['train']['HR_size'] / opt['model_config']['scale'])
    #if 'LapSRN' in opt['model_config']['model']:
    #    assert opt['data_config']['train']['HR_size'] % opt['model_config']['scale'] == 0
    #opt['model_config']['input_shape'] = (num_channels, LR_size, LR_size, LR_size)

    #opt['optim_config']['max_iters'] = opt['optim_config']["epoch_iters"] * opt['optim_config']["epochs"]
    #opt['optim_config']['running_lr'] = opt['optim_config']['base_lr']

    # export CUDA_VISIBLE_DEVICES
    gpu_list = ','.join(str(x) for x in opt['run_config']['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
    return opt


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


def check_resume(opt):
    '''Check resume states and pretrain_model paths'''
    logger = logging.getLogger('base')
    if opt['path']['resume_state']:
        if opt['path']['pretrain_model_G'] or opt['path']['pretrain_model_D']:
            logger.warning('pretrain_model path will be ignored when resuming training.')

        state_idx = osp.basename(opt['path']['resume_state']).split('.')[0]
        opt['path']['pretrain_model_G'] = osp.join(opt['path']['models'],
                                                   '{}_G.pth'.format(state_idx))
        logger.info('Set [pretrain_model_G] to ' + opt['path']['pretrain_model_G'])
        if 'gan' in opt['model']:
            opt['path']['pretrain_model_D'] = osp.join(opt['path']['models'],
                                                       '{}_D.pth'.format(state_idx))
            logger.info('Set [pretrain_model_D] to ' + opt['path']['pretrain_model_D'])
