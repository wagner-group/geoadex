'''
Code to load ANN datasets.
Reference:
https://github.com/erikbern/ann-benchmarks/blob/master/ann_benchmarks/datasets.py
'''
import os

import h5py

DATA_DIR = '/home/chawin/data/'


def download(src, dst):
    if not os.path.exists(dst):
        # TODO: should be atomic
        print('downloading %s -> %s...' % (src, dst))
        urlretrieve(src, dst)


def get_dataset_fn(dataset):
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    return os.path.join(DATA_DIR, '%s.hdf5' % dataset)


def get_dataset(which):
    hdf5_fn = get_dataset_fn(which)
    try:
        url = 'http://ann-benchmarks.com/%s.hdf5' % which
        download(url, hdf5_fn)
    except:
        raise NotImplementedError('dataset does not exist.')
    hdf5_f = h5py.File(hdf5_fn, 'r')
    return hdf5_f


def to_numpy_array(dataset):
    x_train = dataset['train'][:]
