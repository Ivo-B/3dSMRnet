import numpy as np
import h5py


def load_poissondisc_sampling(path):
    points = np.loadtxt(path, delimiter=',')
    return points


def zero_padding(dataset, out_shape):
    # zero padding
    input_size = tuple(dataset.shape[2:])
    pad_size = [(0, 0), (0, 0), 0, 0, 0]
    for vI, vO, i in zip(input_size, out_shape, range(2, 6)):
        diff = (vO - vI)
        if diff % 2 != 0:
            pad_size[i] = (2 * (diff // 2), diff % 2)
        else:
            pad = (diff // 2)
            pad_size[i] = (pad, pad)

    return np.pad(dataset, tuple(pad_size), mode='constant', constant_values=0)


def load_dataset(file_path):
    with h5py.File(file_path, 'r') as hf:
        for key in hf.keys():
            # print(key) #Names of the groups in HDF5 file.
            if key.lower() == 'data':
                dset_rgb = np.array(hf[key]).astype(np.float32)
            if 'hz' in key.lower():
                dset_hz = np.array(hf[key]).astype(np.float32)

    return dset_rgb, dset_hz


def transform_to_BGR_FHWDC(dataset):
    # RGB -> BGR; # F CHWD -> F HWDC
    return np.transpose(dataset[:, [2, 1, 0], :, :, :], [0, 2, 3, 4, 1])


def transform_to_RGB_FCHWD(dataset):
    # BGR -> RGB; # F HWDC -> F CHWD
    return np.transpose(dataset[:, :, :, :, [2, 1, 0]], [0, 4, 1, 2, 3])


def save_dataset(dataset, dataset_hz, file_path):
    print('Saving dataset to ', file_path)
    with h5py.File(file_path, 'w') as hf:
        hf.create_dataset("data", data=dataset)
        hf.create_dataset("hz", data=dataset_hz)

def main():
    # file_name = 'Data9x9x9x3x8391Perimag_SNR1.h5'
    # file_name = 'Data37x37x37x3x3175Perimag_SNR3.h5'
    file_names = ['Data37x37x37x3x3175Perimag_SNR3.h5']
    file_names += ['Data37x37x37x3x3929SynomagPEG_SNR3.h5']
    file_name_outs = ["Perimag"]
    file_name_outs += ["SynomagPEG"]
    for file_name, file_name_out in zip(file_names, file_name_outs):

        file_path = 'E:\\datasets\\MPISystemMatrix\\{}'.format(file_name)
        file_path_out = 'E:\\datasets\\MPISystemMatrix\\pre_processed'

        ouput_size = (40, 40, 40)
        print('Loading dataset: {}'.format(file_path))
        dset_rgb, dset_hz = load_dataset(file_path)
        print(dset_rgb.shape)
        print('done')

        print('Zero pad to {}'.format(ouput_size))
        dset_rgb_zeroPad = zero_padding(dset_rgb, ouput_size)
        print('done')

        np.random.seed(0)
        idxs = np.arange(len(dset_rgb))
        np.random.shuffle(idxs)
        t_size = int(len(dset_rgb) * 0.9)

        phase_indxs = [idxs[:t_size], idxs[t_size:], idxs]
        # Test ist the full dataset
        for i, phase in enumerate(['Train', 'Val', 'Test']):
            out_freq = transform_to_BGR_FHWDC(dset_rgb_zeroPad[phase_indxs[i]])
            shape_str = str(out_freq.shape).replace(", ", "x").replace("(", "").replace(")", "")
            file_path = file_path_out + '//{}Test_SNR3_RGB_HR_{}.h5'.format(file_name_out, shape_str)
            save_dataset(out_freq, dset_hz[phase_indxs[i]], file_path)
            for equi in [2, 4]:
                out_freq = transform_to_BGR_FHWDC(dset_rgb_zeroPad[phase_indxs[i]][:, :, ::equi, ::equi, ::equi])
                shape_str = str(out_freq.shape).replace(", ", "x").replace("(", "").replace(")", "")
                file_path = file_path_out + '//{0}{1}_SNR3_RGB_Equi{2}x_{3}.h5'.format(file_name_out, phase, equi, shape_str)
                save_dataset(out_freq, dset_hz[phase_indxs[i]], file_path)


if __name__ == "__main__":
    main()
