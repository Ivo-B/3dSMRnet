import os
import math
from datetime import datetime
import numpy as np
import cv2
from torchvision.utils import make_grid
import random
import torch
import logging
from collections import OrderedDict
import matplotlib.pyplot as plt
import re
from matplotlib import colors
import h5py

####################
# miscellaneous
####################


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('base')
        logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
    '''set up logger'''
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)


####################
# image convert
####################


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1), as_grid=False, data_format='RGB'):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    if min_max:
        tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
        tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    else:
        tensor = tensor.squeeze().float().cpu()
    n_dim = tensor.dim()
    if n_dim == 5:
        img_np = tensor.numpy()
        if data_format == 'RGB':
            img_np = img_np[:, [2, 1, 0], :, :, :] # to BGR
        img_np = np.transpose(img_np, (0, 2, 3, 4, 1)) # to HWDC
    elif n_dim == 4:
        if as_grid:
            n_img = len(tensor)
            img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
            img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
        else:
            img_np = tensor.numpy()
            if data_format == 'RGB':
                img_np = img_np[[2, 1, 0], :, :, :]  # to BGR
            img_np = np.transpose(img_np, (1, 2, 3, 0)) # to HWDC
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


import nibabel as nib
def save_img(vol, img_path, mode='RGB', spacing=1.0):
    #TODO: update save for MIPSystemMatrix vol
    #img = nib.Nifti1Image(data, np.eye(4))
    if mode == 'RGB':
        rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
        vol = (vol).astype(np.uint8).copy().view(dtype=rgb_dtype).reshape(vol.shape)
    nii_img = nib.Nifti1Image(vol, np.eye(4) * spacing)
    nib.save(nii_img, img_path)
    #cv2.imwrite(img_path, img)


def complex_array_to_rgb(X, theme='dark', rmax=None):
    '''Takes an array of complex number and converts it to an array of [r, g, b],
    where phase gives hue and saturaton/value are given by the absolute value.
    Especially for use with imshow for complex plots.'''

    absmax = rmax or np.abs(X).max()
    Y = np.zeros(X.shape + (3,), dtype='float')
    Y[..., 0] = np.angle(X) / (2 * np.pi) % 1
    if theme == 'light':
        Y[..., 1] = np.clip(np.abs(X) / absmax, 0, 1)
        Y[..., 2] = 1
    elif theme == 'dark':
        Y[..., 1] = 1
        Y[..., 2] = np.clip(np.abs(X) / absmax, 0, 1)
    Y = colors.hsv_to_rgb(Y)
    return Y, absmax


def showAndSaveSlice(sr_bVols, lr_bVols, gt_bVol, save_img_path, slice=20, index=0, scale=4, is_train=True, data_format='RGB', data_mean=None, data_std=None):
    def norm_color(slice, vmin, vmax):
        slice = (slice - vmin) / (vmax - vmin)
        slice[slice < 0.0] = 0.
        slice[slice > 1.0] = 1.
        return slice

    normVal = None
    if gt_bVol is not None:
        if data_format == 'RGB':
            gt_slice = gt_bVol[index, :, :, slice, :][:, :, [2, 1, 0]]  # BGR->RGB
            normVal = [gt_slice.min(), gt_slice.max()]
            gt_slice = norm_color(gt_slice, normVal[0], normVal[1])  # color sync
        else:
            # convert imag real to complex
            gt_slice = gt_bVol[index, :, :, slice, :]
            gt_slice = gt_slice[..., 1] + 1j * gt_slice[..., 0]
            gt_slice, normVal = complex_array_to_rgb(gt_slice)

    lr_slices = OrderedDict([])
    sr_slices = OrderedDict([])
    for i, k in enumerate(lr_bVols.keys()):
        if is_train:
            slice_tmp = slice // int(re.findall(r'\d+', k)[0]) if re.findall(r'\d+', k) else slice // scale
        else:
            slice_tmp = (slice * scale) // int(re.findall(r'\d+', k)[0]) if re.findall(r'\d+', k) else slice

        lr_slices[k] = lr_bVols[k][index, :, :, slice_tmp, :]
        if i == 0:
            # invert input norm
            if data_std:
                if type(data_std) is list:
                    # channel std
                    for i, cStd in enumerate(data_std):
                        lr_slices[k][:, :, i] = lr_slices[k][:, :, i] * cStd
                else:
                    lr_slices[k] = lr_slices[k] * data_std

            if data_mean:
                if type(data_mean) is list:
                    # channel mean
                    for i, cMean in enumerate(data_mean):
                        lr_slices[k][:, :, i] = lr_slices[k][:, :, i] + cMean
                else:
                    lr_slices[k] = lr_slices[k] + data_mean


            if data_format == 'RGB':
                lr_slices[k] += 236.17393  # invert input norm
                if normVal is None:
                    normVal = [lr_slices[k].min(), lr_slices[k].max()]
            elif data_format == 'Complex':
                lr_slices[k][:, :, 0] = (lr_slices[k][:, :, 0] * 3396.1528) - 73.54369  # invert input norm
                lr_slices[k][:, :, 1] = (lr_slices[k][:, :, 1] * 3714.9639) - 6.0050526  # invert input norm
                # convert imag real to complex
                if normVal is None:
                    _, normVal = complex_array_to_rgb(lr_slices[k][..., 1] + 1j * lr_slices[k][..., 0])
            else:
                raise NotImplementedError('DataFormat [{:s}] not recognized.'.format(data_format))
        if data_format == 'RGB':
            lr_slices[k] = lr_slices[k][:, :, [2, 1, 0]]  # BGR->RGB
            lr_slices[k] = norm_color(lr_slices[k], normVal[0], normVal[1])  # color sync
        elif data_format == 'Complex':
            # convert imag real to complex
            lr_slices[k] = lr_slices[k][..., 1] + 1j * lr_slices[k][..., 0]
            lr_slices[k], _ = complex_array_to_rgb(lr_slices[k], rmax=normVal)
        else:
            raise NotImplementedError('DataFormat [{:s}] not recognized.'.format(data_format))

    for i, k in enumerate(sr_bVols.keys()):
        if is_train:
            slice_tmp = slice // int(re.findall(r'\d+', k)[0]) if re.findall(r'\d+', k) else slice
        else:
            slice_tmp = slice * int(re.findall(r'\d+', k)[0]) if re.findall(r'\d+', k) else slice * scale
        sr_slices[k] = sr_bVols[k][index, :, :, slice_tmp, :]

        if data_format == 'RGB':
            sr_slices[k] = sr_slices[k][:, :, [2, 1, 0]]  # BGR->RGB
            sr_slices[k] = norm_color(sr_slices[k], normVal[0], normVal[1])  # color sync
        elif data_format == 'Complex':
            # convert imag real to complex
            sr_slices[k] = sr_slices[k][..., 1] + 1j * sr_slices[k][..., 0]
            sr_slices[k], _ = complex_array_to_rgb(sr_slices[k], rmax=normVal)
        else:
            raise NotImplementedError('DataFormat [{:s}] not recognized.'.format(data_format))

    num_row = 2
    if gt_bVol is not None:
        num_col = len(lr_slices) + 1
        fig, axes = plt.subplots(num_row, num_col)
        fig.delaxes(axes[1][0])
        for i, k in enumerate(lr_slices.keys()):
            axes[0, i].imshow(lr_slices[k], vmin=0, vmax=1)
            axes[0, i].set_title(k)
        axes[0, -1].imshow(gt_slice)
        axes[0, -1].set_title('GT')

        for i, k in enumerate(sr_slices.keys()):
            axes[1, i+1].imshow(sr_slices[k], vmin=0, vmax=1)
            axes[1, i+1].set_title(k)
    else:
        num_col = len(sr_slices) + 1
        fig, axes = plt.subplots(1, num_col)
        fig.delaxes(axes[1][0])
        for i, k in enumerate(lr_slices.keys()):
            axes[i].imshow(lr_slices[k], vmin=0, vmax=1)
            axes[i].set_title(k)
        for i, k in enumerate(sr_slices.keys()):
            axes[i + 1].imshow(sr_slices[k], vmin=0, vmax=1)
            axes[i + 1].set_title(k)
    fig.tight_layout()
    fig.savefig(save_img_path)
    plt.close()


class HDF5Store(object):
    """
    Simple class to append value to a hdf5 file on disc (usefull for building keras datasets)

    Params:
        datapath: filepath of h5 file
        dataset: dataset name within the file
        shape: dataset shape (not counting main/batch axis)
        dtype: numpy dtype

    Usage:
        hdf5_store = HDF5Store('/tmp/hdf5_store.h5','X', shape=(20,20,3))
        x = np.random.random(hdf5_store.shape)
        hdf5_store.append(x)
        hdf5_store.append(x)

    From https://gist.github.com/wassname/a0a75f133831eed1113d052c67cf8633
    """

    def __init__(self, datapath, shape, dtype=np.float32, compression="gzip", chunk_len=1):
        self.datapath = datapath
        self.dataset = 'data'
        self.shape = shape
        self.i = 0

        with h5py.File(self.datapath, mode='w') as h5f:
            self.dset = h5f.create_dataset(
                'data',
                shape=(0,) + shape,
                maxshape=(None,) + shape,
                dtype=dtype,
                compression=compression,
                chunks=(chunk_len,) + shape)

            self.dset = h5f.create_dataset(
                'hz',
                shape=(0,) + (1,),
                maxshape=(None,) + (1,),
                dtype=dtype,
                compression=compression,
                chunks=(chunk_len,) + (1,))

    def append(self, values, freq):
        with h5py.File(self.datapath, mode='a') as h5f:
            dset = h5f[self.dataset]
            dset.resize((self.i + 1,) + self.shape)
            dset[self.i] = [values]

            dset = h5f['hz']
            dset.resize((self.i + 1,) + (1,))
            dset[self.i] = [freq]

            self.i += 1
            h5f.flush()


####################
# metric
####################

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def calculate_mse_rmse_psnr(vol_predict, vol_true):
    # calculate volumen metrics
    mse = np.square(vol_true - vol_predict).mean()

    rmse = np.sqrt(mse)
    if rmse == 0:
        psnr = 100.0
    else:
        psnr = 20.0 * np.log10(vol_true.max() / rmse)
    return mse, rmse, psnr


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
