from skimage import measure
import h5py
import numpy as np
import logging
logger = logging.getLogger('base')


gtImg = {}
for phantom in ['concentrationPhantom', 'resolutionPhantom', 'shapePhantom']:
    path = "C:\\Users\\310276216\\Downloads\\{}_GTPeri.h5".format(phantom)
    with h5py.File(path) as hf:
        print('Read hdf5: {}'.format(path))
        for key, value in hf.items():
            gtImg[phantom] = np.squeeze(np.array(value))

root_path = "C:\\Users\\310276216\\Downloads\\"
for method in ['CS8x', 'CS27x', 'CS64x', 'SRRRDB-SynomagEqui2xTrain', 'SRRRDB-SynomagEqui3xTrain_Sub',
               'SRRRDB-SynomagEqui4xTrain']:
    print('\t{}'.format(method))
    str = ""
    mSSIM = 0.
    mPSNR = 0.
    mNRMSE = 0.
    for phantom in ['shapePhantom', 'resolutionPhantom', 'concentrationPhantom',]:
        print('{}'.format(phantom))
        path = root_path + "{}_{}.h5".format(phantom, method)
        with h5py.File(path) as hf:
            print('\t\tRead hdf5: {}'.format(path))
            for key, value in hf.items():
                img = np.squeeze(np.array(value))
        ssim = np.round(measure.compare_ssim(gtImg[phantom], img, data_range=1 ), 4)
        mSSIM += ssim
        psnr = np.round(measure.compare_psnr(gtImg[phantom], img, data_range=1 ), 2)
        mPSNR += psnr
        nrmse = np.round(measure.compare_nrmse(gtImg[phantom], img, norm_type="min-max"), 4)
        mNRMSE += nrmse
        str += "& {} & {} & {}".format(nrmse, ssim, psnr)
    print ("\t\t"+str)
    print ("\t\t {:.4f} & {:.4f} & {:.2f}".format(mNRMSE/3., mSSIM/3., mPSNR/3.))
            #print('\t\t(.{:d}/.{:d}/{})'.format(int(nrmse*10000), int(ssim*10000),psnr))
            #print('\t\tSSIM {}'.format(np.round(measure.compare_ssim(gtImg[phantom], img, data_range=1 ), 4)))
            #print('\t\tPSNR {}'.format(np.round(measure.compare_psnr(gtImg[phantom], img, data_range=1), 2)))
            #print('\t\tNRMSE {}'.format(np.round(measure.compare_nrmse(gtImg[phantom], img, norm_type="min-max"),4)))
