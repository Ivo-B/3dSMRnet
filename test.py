import os
import sys
import logging
import time
import argparse
import numpy as np
from collections import OrderedDict

import options.options as option
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model
from engfmt import Quantity
from timeit import default_timer as timer

try:
    from tensorboardX import SummaryWriter
    is_tensorboard_available = True
except Exception:
    is_tensorboard_available = False


def main():
    # options
    parser = argparse.ArgumentParser()
    # TODO Remove
    opt_p = 'E:\\repos\\3dSMRnet\\experiments\\001_Test_SR-RRDB-3d_scale4.json'    # JUST FOR TESTIN!!!!!!!
    parser.add_argument('-opt', default=opt_p, type=str, required=False, help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=False, is_tensorboard_available=is_tensorboard_available)
    opt = option.dict_to_nonedict(opt)

    run_config = opt['run_config']
    data_config = opt['data_config']
    util.mkdirs((path for key, path in run_config['path'].items() if not key == 'pretrain_model_G'))

    # config loggers. Before it, the log will not work
    util.setup_logger(None, run_config['path']['log'], 'test.log', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))

    # Create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(data_config.items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
        test_loaders.append([test_loader, len(test_set)])

    # Create model
    model = create_model(opt)

    for test_loader, total_images in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info('\nTesting [{:s}]...'.format(test_set_name))
        dataset_dir = os.path.join(run_config['path']['results_root'], test_set_name)
        util.mkdir(dataset_dir)

        test_results = OrderedDict()

        data_size = test_loader.dataset.opt['LRSize'] * opt['model_config']['scale']
        data_format = test_loader.dataset.opt['data_format']
        if data_format == 'RGB':
            data_size = (data_size, data_size, data_size, 3)
        elif data_format == 'Complex':
            data_size = (data_size, data_size, data_size, 2)
        else:
            raise NotImplementedError('DataFormat [{:s}] not recognized.'.format(data_format))

        test_result_dataset = util.HDF5Store(os.path.join(dataset_dir, test_set_name + '_CNNPredict.h5'), data_size)
        test_results['time'] = 0
        test_results['total_time'] = 0

        need_HR = False
        num_images = 0
        for idx, data in enumerate(test_loader):
            need_HR = False if test_loader.dataset.opt['dataroot_HR'] is None else True
            start = timer()
            model.feed_data(data, need_HR=need_HR)
            model.test()  # test
            end = timer()
            test_results['time'] += (end - start)
            visuals = model.get_current_visuals(need_HR=need_HR)
            visuals['hz'] = visuals['hz'].numpy()
            sr_imgs = OrderedDict([])
            lr_imgs = OrderedDict([])
            num_images += len(visuals['hz'])

            for k in visuals.keys():
                if 'SR' in k:
                    sr_imgs[k] = (util.tensor2img(visuals[k], min_max=None, out_type=np.float32, as_grid=False, data_format=data_format))# float32
                    if sr_imgs[k].ndim == 4:
                        sr_imgs[k] = sr_imgs[k][np.newaxis, :, :, :, :]
                if 'LR' in k:
                    lr_imgs[k] = (util.tensor2img(visuals[k], min_max=None, out_type=np.float32, as_grid=False, data_format=data_format))# float32
                    if lr_imgs[k].ndim == 4:
                        lr_imgs[k] = lr_imgs[k][np.newaxis, :, :, :, :]
            if need_HR:
                gt_img = util.tensor2img(visuals['HR'], min_max=None, out_type=np.float32, as_grid=False, data_format=data_format)  # float32
                if gt_img.ndim == 4:
                    gt_img = gt_img[np.newaxis, :, :, :, :]
            else:
                gt_img = None

            # save images
            for k in sr_imgs.keys():
                for img_num in range(len(visuals['hz'])):
                    img_name = "{0:d}_{1:s}".format(idx * lr_imgs['LR'].shape[0] + img_num, str(Quantity(visuals['hz'][img_num], 'hz')))
                    suffix = opt['suffix']
                    if suffix:
                        save_img_path = os.path.join(dataset_dir, img_name + suffix + '.nii.gz')
                    else:
                        save_img_path = os.path.join(dataset_dir, img_name + '.nii.gz')

                    test_result_dataset.append(sr_imgs[k][img_num], visuals['hz'][img_num])
                    if run_config['visual_examples']:
                        util.showAndSaveSlice(sr_imgs, lr_imgs, gt_img, save_img_path.replace('.nii.gz', '.png'),
                                              slice=test_loader.dataset.opt['LRSize'] // 2,
                                              scale=opt['model_config']['scale'], is_train=False, index=img_num)

            # calculate MSE/RMSE
            if need_HR:
                log_str = ""
                for sr_k in sr_imgs.keys():
                    if 'x' in sr_k:  # find correct key
                        for lr_k in lr_imgs.keys():
                            if sr_k.replace('SR', '') in lr_k:
                                tmp_hr = lr_imgs[lr_k]
                                break
                    else:
                        tmp_hr = gt_img
                    for sr_vol, lr_vol in zip(sr_imgs[sr_k], tmp_hr):
                        mse = np.square(sr_vol - lr_vol).mean()
                        rmse = np.sqrt(mse)

                        if sr_k in test_results:
                            test_results[sr_k]['mse'].append(mse)
                            test_results[sr_k]['rmse'].append(rmse)
                        else:
                            test_results[sr_k] = OrderedDict([])
                            test_results[sr_k]['mse'] = [mse]
                            test_results[sr_k]['rmse'] = [rmse]
                    log_str += "\n\t\t{}  MSE: {:.6f}; RMSE: {:.6f}".format(
                        sr_k, np.mean(test_results[sr_k]['mse']), np.mean(test_results[sr_k]['rmse']))

            logger.info("System matrix {}/{}: {}".format(num_images, total_images, log_str))
            end = timer()
            test_results['total_time'] += (end - start)

        logger.info("Raw processing time: {} s".format(test_results['time']))
        logger.info("Total time : {} s".format(test_results['total_time']))
        if need_HR:  # metrics
            for tr_k in test_results.keys():
                if 'time' not in tr_k:
                    # Average results
                    ave_mse = sum(test_results[tr_k]['mse']) / len(test_results[tr_k]['mse'])
                    ave_rmse = sum(test_results[tr_k]['rmse']) / len(test_results[tr_k]['rmse'])
                    logger.info('----Average MSE/RMSE results for {} {}----\n\tMSE: {:.6f}; RMSE: {:.6f}.\n'.format(test_set_name, tr_k, ave_mse, ave_rmse))

if __name__ == "__main__":
    main()
