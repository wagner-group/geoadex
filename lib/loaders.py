'''
Load datasets. Taken from:
https://github.com/wangwllu/knn_robustness/blob/master/knn_robustness/utils/loaders.py
'''

import os
from abc import ABC, abstractmethod
from typing import List

import numpy as np
from sklearn.datasets import load_svmlight_files, make_moons
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def initialize_data(params):
    loader = LoaderFactory().create(
        name=params['dataset'],
        root=params['dataset_dir'],
        random=params['random'],
        seed=params['seed'],
        partial=params['partial'],
        label_domain=params['label_domain'],
        gaussian_params=params['gaussian'])

    return loader.load()


def encode_labels(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    return y_train, y_test


def scale_features(X_train, X_test):
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


class Loader(ABC):

    def __init__(self, root, random, seed):
        self._root = root
        self._random = random
        self._seed = seed

    def load(self):
        X_train, y_train, X_test, y_test = self._load_preprocessed_data()
        return X_train, y_train, X_test, y_test

    @abstractmethod
    def _load_preprocessed_data(self):
        pass


class LibsvmLoader(Loader):

    def _load_preprocessed_data(self):
        if self._path_of_test is None:
            X, y = load_svmlight_files([self._path_of_train])
            X = X.toarray()
            ind = np.arange(len(X))
            # Set random seed to 0 to match Yang et al. implementation
            np.random.seed(self._seed)
            np.random.shuffle(ind)
            X_train, y_train = X[ind[200:]], y[ind[200:]]
            X_test, y_test = X[ind[:200]], y[ind[:200]]
        else:
            X_train, y_train, X_test, y_test = load_svmlight_files(
                [self._path_of_train, self._path_of_test])
            X_train = X_train.toarray()
            X_test = X_test.toarray()
            # DEBUG: shuffle train-test split
            if self._random:
                np.random.seed(self._seed)
                X = np.concatenate([X_train, X_test])
                y = np.concatenate([y_train, y_test])
                test_ratio = len(X_test) / len(X)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, random_state=self._seed, test_size=test_ratio)

        y_train, y_test = encode_labels(y_train, y_test)
        X_train, X_test = scale_features(X_train, X_test)

        return X_train, y_train, X_test, y_test

    @ property
    def _path_of_train(self):
        return os.path.join(self._root, self._file_name_of_train)

    @ property
    def _path_of_test(self):
        if self._file_name_of_test is None:
            return None
        return os.path.join(self._root, self._file_name_of_test)

    @ property
    @ abstractmethod
    def _file_name_of_train(self):
        pass

    @ property
    @ abstractmethod
    def _file_name_of_test(self):
        pass


class GisetteLoader(LibsvmLoader):

    @ property
    def _file_name_of_train(self):
        return 'gisette_scale'

    @ property
    def _file_name_of_test(self):
        return 'gisette_scale.t'


class LetterLoader(LibsvmLoader):

    @ property
    def _file_name_of_train(self):
        return 'letter.scale'

    @ property
    def _file_name_of_test(self):
        return 'letter.scale.t'


class PendigitsLoader(LibsvmLoader):

    @ property
    def _file_name_of_train(self):
        return 'pendigits'

    @ property
    def _file_name_of_test(self):
        return 'pendigits.t'


class UspsLoader(LibsvmLoader):

    @ property
    def _file_name_of_train(self):
        return 'usps'

    @ property
    def _file_name_of_test(self):
        return 'usps.t'


class SatimageLoader(LibsvmLoader):

    @ property
    def _file_name_of_train(self):
        return 'satimage.scale'

    @ property
    def _file_name_of_test(self):
        return 'satimage.scale.t'


class GaussianLoader(Loader):

    def __init__(self, num_points, dim, dist, sd, seed=1, test_ratio=0.2):
        self.seed = seed
        self.num_points = num_points
        self.dim = dim
        self.dist = dist
        self.sd = sd
        self.test_ratio = test_ratio

    def load(self):
        np.random.seed(self.seed)
        data = np.random.randn(self.num_points, self.dim) * self.sd
        data[:self.num_points // 2, 0] -= self.dist
        data[self.num_points // 2:, 0] += self.dist
        label = np.zeros(self.num_points, dtype=np.int64)
        label[self.num_points // 2:] = 1
        X_train, X_test, y_train, y_test = train_test_split(
            data, label, random_state=self.seed, test_size=self.test_ratio)
        X_train, X_test = scale_features(X_train, X_test)
        return X_train, y_train, X_test, y_test

    def _load_preprocessed_data(self):
        return self.load()


class CovtypeLoader(Loader):

    def __init__(self, root, seed, num_points):
        self.root = root
        self.seed = seed
        self.num_points = num_points

    def load(self):
        np.random.seed(self.seed)
        data = np.genfromtxt(os.path.join(self.root, 'covtype.data'), delimiter=',')
        ty = data[:, -1].astype(int) - 1
        tX = data[:, :-1].astype(float)
        idx = np.random.choice(np.arange(len(tX)), self.num_points, replace=False)
        X, y = tX[idx], ty[idx]

        ind = np.arange(len(X))
        np.random.seed(self.seed - 1)
        np.random.shuffle(ind)
        X_train, y_train = X[ind[200:]], y[ind[200:]]
        X_test, y_test = X[ind[:200]], y[ind[:200]]

        y_train, y_test = encode_labels(y_train, y_test)
        X_train, X_test = scale_features(X_train, X_test)

        return X_train, y_train, X_test, y_test

    def _load_preprocessed_data(self):
        return self.load()


class DiabetesLoader(LibsvmLoader):

    @ property
    def _file_name_of_train(self):
        return 'diabetes'

    @ property
    def _file_name_of_test(self):
        return None


class FourclassLoader(LibsvmLoader):

    @ property
    def _file_name_of_train(self):
        return 'fourclass'

    @ property
    def _file_name_of_test(self):
        return None


class AustralianLoader(LibsvmLoader):

    @ property
    def _file_name_of_train(self):
        return 'australian'

    @ property
    def _file_name_of_test(self):
        return None


class CancerLoader(LibsvmLoader):

    @ property
    def _file_name_of_train(self):
        return 'breast-cancer'

    @ property
    def _file_name_of_test(self):
        return None


class YangMNISTLoader(Loader):

    def __init__(self, n_dims, seed):
        self.n_dims = n_dims
        self.seed = seed

    def load(self):
        from keras.datasets import mnist

        (X, y), (tX, ty) = mnist.load_data()

        def process(X, y):
            X = X.reshape(len(X), -1).astype(np.float) / 255.
            y = np.copy(y)
            y.setflags(write=1)
            idx1 = np.where(y == 1)[0]
            idx2 = np.where(y == 7)[0]
            y[idx1] = 0
            y[idx2] = 1
            X = np.vstack((X[idx1], X[idx2])).astype(np.float) / 255.
            y = np.concatenate((y[idx1], y[idx2]))
            return X, y
        X, y_train = process(X, y)
        tX, y_test = process(tX, ty)

        if self.n_dims:
            pca = PCA(n_components=self.n_dims, random_state=0)
            ttX = pca.fit_transform(np.vstack((X, tX)))
            X_train, X_test = ttX[:len(X)], ttX[len(X):]

        # Set seed to match Yang's code
        idxs = np.arange(len(X_test))
        np.random.seed(self.seed - 1)
        np.random.shuffle(idxs)
        X_test, y_test = X_test[idxs[:200]], y_test[idxs[:200]]
        idxs = np.arange(len(X_train))
        np.random.seed(self.seed)
        np.random.shuffle(idxs)
        X_train, y_train = X_train[idxs], y_train[idxs]
        X_train = X_train.reshape((len(X_train), -1))
        X_test = X_test.reshape((len(X_test), -1))

        y_train, y_test = encode_labels(y_train, y_test)
        X_train, X_test = scale_features(X_train, X_test)

        return X_train, y_train, X_test, y_test

    def _load_preprocessed_data(self):
        return self.load()


class YangFMNISTLoader(Loader):

    def __init__(self, n_dims, seed):
        self.n_dims = n_dims
        self.seed = seed

    def load(self):
        from keras.datasets import fashion_mnist

        (X, y), (tX, ty) = fashion_mnist.load_data()

        def process(X, y):
            X = X.reshape(len(X), -1).astype(np.float) / 255.
            y = np.copy(y)
            y.setflags(write=1)
            idx1 = np.where(y == 0)[0]
            idx2 = np.where(y == 6)[0]
            y[idx1] = 0
            y[idx2] = 1
            X = np.vstack((X[idx1], X[idx2])).astype(np.float) / 255.
            y = np.concatenate((y[idx1], y[idx2]))
            return X, y
        X, y_train = process(X, y)
        tX, y_test = process(tX, ty)

        if self.n_dims:
            pca = PCA(n_components=self.n_dims, random_state=0)
            ttX = pca.fit_transform(np.vstack((X, tX)))
            X_train, X_test = ttX[:len(X)], ttX[len(X):]

        # Set seed to match Yang's code
        idxs = np.arange(len(X_test))
        np.random.seed(self.seed - 1)
        np.random.shuffle(idxs)
        X_test, y_test = X_test[idxs[:200]], y_test[idxs[:200]]
        idxs = np.arange(len(X_train))
        np.random.seed(self.seed)
        np.random.shuffle(idxs)
        X_train, y_train = X_train[idxs], y_train[idxs]
        X_train = X_train.reshape((len(X_train), -1))
        X_test = X_test.reshape((len(X_test), -1))

        y_train, y_test = encode_labels(y_train, y_test)
        X_train, X_test = scale_features(X_train, X_test)

        return X_train, y_train, X_test, y_test

    def _load_preprocessed_data(self):
        return self.load()


class HalfmoonLoader(Loader):

    def load(self):
        X, y = make_moons(n_samples=2200, noise=0.25, random_state=0)

        ind = np.arange(len(X))
        np.random.seed(0)
        np.random.shuffle(ind)
        X_train, y_train = X[ind[200:]], y[ind[200:]]
        X_test, y_test = X[ind[:200]], y[ind[:200]]

        y_train, y_test = encode_labels(y_train, y_test)
        X_train, X_test = scale_features(X_train, X_test)

        return X_train, y_train, X_test, y_test

    def _load_preprocessed_data(self):
        return self.load()


class IJCNNLoader(Loader):

    def load(self):
        n_samples = 2200
        X, y = load_svmlight_files(['/home/chawin/data/ijcnn1.tr'])
        X = X.todense()

        idx1 = np.random.choice(
            np.where(y == 1)[0], n_samples//2, replace=False)
        idx2 = np.random.choice(
            np.where(y == -1)[0], n_samples//2, replace=False)
        y[idx1] = 0
        y[idx2] = 1
        X = np.vstack((X[idx1], X[idx2])).astype(np.float)
        y = np.concatenate((y[idx1], y[idx2]))
        y = y.astype(int)

        ind = np.arange(len(X))
        np.random.seed(0)
        np.random.shuffle(ind)
        X_train, y_train = X[ind[200:]], y[ind[200:]]
        X_test, y_test = X[ind[:200]], y[ind[:200]]

        y_train, y_test = encode_labels(y_train, y_test)
        X_train, X_test = scale_features(X_train, X_test)

        return X_train, y_train, X_test, y_test

    def _load_preprocessed_data(self):
        return self.load()


class LoaderDecorator(Loader):

    def __init__(self, decorated_loader: Loader):
        self._decorated_loader = decorated_loader

    def load(self):
        return self._decorated_loader.load()

    def _load_preprocessed_data(self):
        return self._decorated_loader._load_preprocessed_data()


class PartialLoaderDecorator(LoaderDecorator):

    def __init__(self, decorated_loader: Loader, label_domain: List[int]):
        super().__init__(decorated_loader)
        self._label_domain = label_domain

    def load(self):
        X_train, y_train, X_test, y_test = super().load()

        X_train, y_train = self._select(X_train, y_train)
        X_test, y_test = self._select(X_test, y_test)

        y_train, y_test = encode_labels(y_train, y_test)
        return X_train, y_train, X_test, y_test

    def _select(self, X, y):
        mask = np.isin(y, self._label_domain)
        return X[mask], y[mask]


class LoaderFactory:

    def create(self, name, root, random=True, seed=None, partial=False,
               label_domain=None, gaussian_params=None):

        if name == 'gisette':
            loader = GisetteLoader(root, random, seed)
        elif name == 'letter':
            loader = LetterLoader(root, random, seed)
        elif name == 'pendigits':
            loader = PendigitsLoader(root, random, seed)
        elif name == 'usps':
            loader = UspsLoader(root, random, seed)
        elif name == 'satimage':
            loader = SatimageLoader(root, random, seed)
        elif name == 'gaussian':
            loader = GaussianLoader(seed=seed, **gaussian_params)
        elif name == 'cancer':
            loader = CancerLoader(root, random, seed)
        elif name == 'australian':
            loader = AustralianLoader(root, random, seed)
        elif name == 'diabetes':
            loader = DiabetesLoader(root, random, seed)
        elif name == 'fourclass':
            loader = FourclassLoader(root, random, seed)
        elif name == 'covtype':
            loader = CovtypeLoader(root, seed, 2200)
        elif name == 'halfmoon':
            loader = HalfmoonLoader(root, random, seed)
        elif name == 'ijcnn':
            loader = IJCNNLoader(root, random, seed)
        elif name == 'yang-mnist':
            loader = YangMNISTLoader(25, seed)
        elif name == 'yang-fmnist':
            loader = YangFMNISTLoader(25, seed)
        else:
            raise Exception('unsupported dataset')

        if partial:
            assert label_domain is not None
            loader = PartialLoaderDecorator(loader, label_domain)

        return loader
