__author__ = "Aditya Singh"
__version__ = "0.1"

import tensorflow as tf
import numpy as np
np.random.seed(42)


class Data:
    def __init__(self, **kwargs):
        self.selected_samples = kwargs.get('selected_samples')
        self.population_mu = None
        self.cov = None
        self.inv_cov = None

    def set_population_mu(self, mu):
        self.population_mu = mu

    def set_cov(self, cov):
        self.cov = cov
        det = np.linalg.det(cov)
        if det == 0:
            raise ZeroDivisionError('Covariance matrix is not invertible!')
        self.inv_cov = tf.linalg.inv(cov)


class ReplayManager:
    def __init__(self, **kwargs):
        self.replay_buffer_size = kwargs.get('replay_buffer_size', 20)
        self.library = {}

    def populate_replay_library(self, x, y):
        """
        :param x: List of images
        :param y: Integer class label
        :return:
        """
        sample_indices = np.random.randint(low=0, high=x.shape[0], size=self.replay_buffer_size)
        selected_samples = x[sample_indices]
        self.library[y] = Data(selected_samples=selected_samples)

    def fit_gaussian(self, features_x, label):
        self.library[label].set_mean(np.mean(features_x, axis=0))
        self.library[label].set_cov(np.cov(features_x.T))

    def get_data(self, cls_label):
        return self.library[cls_label]