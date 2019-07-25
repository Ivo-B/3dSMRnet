# System libs
import os
from collections import OrderedDict
import logging
import argparse
import random
import math
from engfmt import Quantity

# Numerical libs
import numpy as np

import torch
import torch.utils.data
import torch.backends.cudnn

from data import create_dataloader, create_dataset
import options.options as option
from utils import util
from models import create_model

try:
    from tensorboardX import SummaryWriter
    is_tensorboard_available = True
    print("TensorBoardX summary writer Available")
except Exception:
    is_tensorboard_available = False


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="PyTorch LapSRN")

    opt_p = 'experiments/001_Train_SR-RRDB-3d_SynomagD_scale4.json'
    parser.add_argument('-opt', default=opt_p, type=str, required=False, help='Path to option JSON file.')

    config = option.parse(parser.parse_args().opt, True, is_tensorboard_available)
    config = option.dict_to_nonedict(config)

    run_config = config['run_config']
    optim_config = config['optim_config']
    data_config = config['data_config']

    # train from scratch OR resume training
    if run_config['path']['+']:  # resuming training
        resume_state = torch.load(run_config['path']['resume_state'])
    else:  # training from scratch
        resume_state = None
        util.mkdir_and_rename(run_config['path']['experiments_root'])  # rename old folder if exists
        util.mkdirs((path for key, path in run_config['path'].items() if not key == 'experiments_root'
                     and 'pretrain_model' not in key and 'resume' not in key))

    # config loggers. Before it, the log will not work
    util.setup_logger(None, run_config['path']['log'], 'train', level=logging.INFO, screen=True)
    util.setup_logger('val', run_config['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(config))

    if resume_state:
        # TODO: not implemented just copied, update check_resume
        # logger.info('Resuming training from epoch: {}, iter: {}.'.format(
        #     resume_state['epoch'], resume_state['iter']))
        # option.check_resume(config)  # check resume options
        raise NotImplementedError

    # tensorboard logger
    if run_config['use_tb_logger'] and 'debug' not in run_config['id']:
        util.mkdir_and_rename(
            os.path.join(run_config['path']['root'], 'tb_logger', run_config['id']))  # rename old folder if exists
        tb_logger = SummaryWriter(log_dir=os.path.join(run_config['path']['root'], 'tb_logger', run_config['id']))

    # set random seed
    logger.info("===> Set seed")
    seed = run_config['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
        logger.info("=> Random seed: {}".format(seed))
    else:
        seed = int(seed, 16)
        logger.info("=> Manual seed: {}".format(seed))
    seed = int(run_config['manual_seed'], 16)
    util.set_random_seed(seed)

    torch.backends.cudnn.benckmark = True

    logger.info("===> Loading datasets")
    for phase, dataset_opt in data_config.items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                len(train_set), train_size))
            total_iters = int(optim_config['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if 'debug' in run_config['id']:
                total_epochs = 10
            logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                total_epochs, total_iters))
            train_loader = create_dataloader(train_set, dataset_opt)
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            logger.info('Number of val images in [{:s}]: {:d}'.format(dataset_opt['name'],
                                                                      len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    logger.info("===> Building model")
    # create model
    model = create_model(config)

    if is_tensorboard_available and 'debug' not in run_config['id']:
        # TODO: fix problem
        # Save graph to tensorboard
        # dummy_input = Variable(torch.rand((10,) + config['model_config']['input_shape']))
        # tb_logger.add_graph(model.netG, (dummy_input,))
        pass

    # resume training
    if resume_state:
        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    best_psnr = OrderedDict([])
    is_newBest = True
    for epoch in range(start_epoch, total_epochs):
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            # update learning rate
            model.update_learning_rate()

            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            # log
            if current_step % run_config['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v.val) if v.val is not None else ''
                    # tensorboard logger
                    if run_config['use_tb_logger'] and 'debug' not in run_config['id']:
                        tb_logger.add_scalar('Train/running/{}'.format(k), v.val, current_step)
                logger.info(message)

            # validation
            if current_step % optim_config['val_freq'] == 0:
                avg_metric = OrderedDict([])

                idx = 0
                total_images = 0
                img_dir = os.path.join(run_config['path']['val_images'])
                util.mkdir(img_dir)
                for val_data in val_loader:
                    idx += 1

                    model.feed_data(val_data)
                    model.test()

                    visuals = model.get_current_visuals()
                    visuals['hz'] = visuals['hz'].numpy()
                    sr_imgs = OrderedDict([])
                    lr_imgs = OrderedDict([])

                    for k in visuals.keys():
                        if 'SR' in k:
                            sr_imgs[k] = (util.tensor2img(visuals[k], min_max=None, out_type=np.float32,
                                                          as_grid=False, data_format=data_config['val']['data_format']))  # float32
                            if sr_imgs[k].ndim == 4:
                                sr_imgs[k] = sr_imgs[k][np.newaxis, :, :, :, :]
                        if 'LR' in k:
                            lr_imgs[k] = (util.tensor2img(visuals[k], min_max=None, out_type=np.float32,
                                                          as_grid=False, data_format=data_config['val']['data_format']))  # float32
                            if lr_imgs[k].ndim == 4:
                                lr_imgs[k] = lr_imgs[k][np.newaxis, :, :, :, :]
                    gt_img = util.tensor2img(visuals['HR'], min_max=None, out_type=np.float32,
                                             as_grid=False, data_format=data_config['val']['data_format'])
                    if gt_img.ndim == 4:
                        gt_img = gt_img[np.newaxis, :, :, :, :]

                    # calculate PSNR
                    for sr_k in sr_imgs.keys():
                        if 'x' in sr_k:  # find correct key
                            for lr_k in lr_imgs.keys():
                                if sr_k.replace('SR', '') in lr_k:
                                    tmp_hr = lr_imgs[lr_k]
                                    break
                        else:
                            tmp_hr = gt_img
                        for sr_vol, lr_vol in zip(sr_imgs[sr_k], tmp_hr):
                            mse, rmse, psnr = util.calculate_mse_rmse_psnr(sr_vol, lr_vol)
                            if sr_k in avg_metric:
                                avg_metric[sr_k]['mse'] += mse
                                avg_metric[sr_k]['rmse'] += rmse
                                avg_metric[sr_k]['psnr'] += psnr
                            else:
                                avg_metric[sr_k] = OrderedDict([])
                                avg_metric[sr_k]['mse'] = mse
                                avg_metric[sr_k]['rmse'] = rmse
                                avg_metric[sr_k]['psnr'] = psnr

                    # Save SR images for reference
                    for img_num in range(len(visuals['hz'])):
                        if total_images % 40 == 0:
                            img_name = "{0:d}_{1:s}_{2:d}.png".format(total_images,
                                                                      str(Quantity(visuals['hz'][img_num], 'hz')),
                                                                      current_step)

                            save_img_path = os.path.join(img_dir, img_name)
                            util.showAndSaveSlice(sr_imgs, lr_imgs, gt_img, save_img_path,
                                                  scale=config['model_config']['scale'], index=img_num,
                                                  data_format=data_config['val']['data_format'],
                                                  data_mean=data_config['val']['data_mean'],
                                                  data_std=data_config['val']['data_std'])
                        total_images += 1
                log_str = '# Validation #'
                log_str2 = '<epoch:{:3d}, iter:{:8,d}>'.format(epoch, current_step)
                for k in avg_metric.keys():
                    for metric_k in avg_metric[k]:
                        avg_metric[k][metric_k] = avg_metric[k][metric_k] / idx
                        if 'rmse' in metric_k:
                            if k not in best_psnr:
                                best_psnr[k] = 10e6
                            if avg_metric[k][metric_k] < best_psnr[k]:
                                is_newBest = True
                                best_psnr[k] = avg_metric[k][metric_k]
                                log_str += '\tBEST'
                        log_str += ' {}-{}: {:.4e} * {}'.format(k, metric_k, avg_metric[k][metric_k], idx)
                        log_str2 += ' {}-{}: {:.4e} * {}'.format(k, metric_k, avg_metric[k][metric_k], idx)

                        # tensorboard logger
                        if run_config['use_tb_logger'] and 'debug' not in run_config['id']:
                            tb_logger.add_scalar('val/{}_{}'.format(k, metric_k), avg_metric[k][metric_k], current_step)
                # log
                logger.info(log_str)
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info(log_str2)

            # save models and training states
            if current_step % run_config['logger']['save_checkpoint_freq'] == 0 or is_newBest:
                logger.info('Saving models and training states.')
                model.save(current_step)
                model.save_training_state(epoch, current_step)
                is_newBest = False

        # log
        logs = model.get_current_log()
        for k, v in logs.items():
            # tensorboard logger
            if run_config['use_tb_logger'] and 'debug' not in run_config['id']:
                if v.avg is not None:
                    tb_logger.add_scalar('Train/{}'.format(k), v.avg, current_step)
        model.reset_log()

    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')


if __name__ == "__main__":
    main()
