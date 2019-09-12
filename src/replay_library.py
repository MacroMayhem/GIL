__author__ = "Aditya Singh"
__version__ = "0.1"

import tensorflow as tf
import numpy as np
import sklearn.mixture
import copy
np.random.seed(42)


class Data:
    def __init__(self):
        self.selected_samples = None
        self.mean = None
        self.cov = None
        self.inv_cov = None
        self.gmm = None

    def set_samples(self, samples):
        self.selected_samples = copy.deepcopy(samples)

    def set_mean(self, mu):
        self.mean = mu

    def set_cov(self, cov):
        self.cov = cov.astype('float32')
        sign, logdet = np.linalg.slogdet(self.cov)
        if sign == 0:
            raise ArithmeticError('Covariance matrix is not invertible!')
        self.inv_cov = tf.linalg.inv(self.cov)
        if not self.is_pos_semidef(self.inv_cov):
            raise ArithmeticError('Inv Cov is not Positive semi definite')

    def is_pos_semidef(self, x):
        return np.all(np.linalg.eigvals(x) >= 0)

    def set_gmm(self, gmm):
        self.gmm = gmm


class ReplayManager:
    def __init__(self, **kwargs):
        self.replay_buffer_size = kwargs.get('replay_buffer_size', 20)
        self.library = {}

    def init_replay_library(self, y):
        """
        :param x: List of images
        :param y: Integer class label
        :return:
        """
        self.library[y] = Data()

    def populate_replay_library(self, x, y):
        sample_indices = np.random.randint(low=0, high=x.shape[0], size=self.replay_buffer_size)
        selected_samples = x[sample_indices]
        self.library[y].set_samples(selected_samples)

    def fit_gaussian(self, features_x, label):
        self.library[label].set_mean(np.mean(features_x, axis=0))
        self.library[label].set_cov(np.cov(features_x.numpy().T))

    def get_mean(self, cls):
        return self.library[cls].mean

    def get_inv_cov(self, cls):
        return self.library[cls].inv_cov

    def get_data(self, cls_label):
        return self.library[cls_label]

    def fitGMM(self, features_x, label, n_components=1):
        gmm = sklearn.mixture.GaussianMixture(n_components=n_components, covariance_type='full', n_init=3)
        gmm.fit(features_x.numpy())
        print('The fit converged: {}'.format(gmm.converged_))
        self.library[label].set_gmm(gmm)

    def get_mhdistance(self, features_x, labels):
        mh_distance = []
        mh_labels = []
        for g in sorted(np.unique(labels)):
            if g not in self.library:
                continue
            indices = np.where(np.asarray(labels) == g)
            cls_x = features_x[indices][:]
            batch_size, dims = cls_x.shape
            cls_mean = self.library[g].mean
            cls_inv_cov = self.library[g].inv_cov
            cls_mean_shift = tf.reshape(tf.subtract(cls_x, cls_mean), [batch_size, -1, dims])
            left_mult = tf.reshape(tf.linalg.matmul(cls_mean_shift, cls_inv_cov), [batch_size, -1])
            d = tf.math.sqrt(tf.keras.backend.batch_dot(left_mult, tf.reshape(cls_mean_shift, [batch_size, -1]), axes=[1, 1]))
            mh_distance.append(d.numpy())
            mh_labels.append([g for i in range(batch_size)])
        return mh_distance, mh_labels

    def get_mean_distance(self, features_x, labels):
        mean_distance = []
        mean_labels = []
        for g in sorted(np.unique(labels)):
            if g not in self.library:
                continue
            indices = np.where(np.asarray(labels) == g)
            cls_x = features_x[indices][:]
            batch_size, dims = cls_x.shape
            cls_mean = self.library[g].mean
            d = np.linalg.norm(np.subtract(cls_x, cls_mean), axis=1)
            mean_distance.append(d)
            mean_labels.append([g for i in range(batch_size)])
        return mean_distance, labels