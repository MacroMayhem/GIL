__author__ = "Aditya Singh"
__version__ = "0.1"

from enum import Enum
import tensorflow as tf
import numpy as np
np.random.seed(42)


class Strategy(Enum):
    CLF = 0
    CLU = 1
    DIST = 2
    MIX_REPLAY = 3
    DIST_OLD = 4
    NORM = 5
    DEEP_APPROX = 6

class DataLoader:
    @staticmethod
    def load_data(dataset_name='mnist', n_classes_per_step=2, initial_learning_size=0, shuffle_order=False):
        if dataset_name == 'mnist':
            (train_images, y_train), (test_images, y_test) = tf.keras.datasets.mnist.load_data()
        elif dataset_name == 'cifar100':
            (train_images, y_train), (test_images, y_test) = tf.keras.datasets.cifar100.load_data()
        elif dataset_name == 'cifar-10':
            (train_images, y_train), (test_images, y_test) = tf.keras.datasets.cifar10.load_data()
        else:
            raise NotImplementedError('{} currently not implemented!', dataset_name)
        training_order = np.unique(y_test)

        if shuffle_order:
            np.random.shuffle(training_order)
        if initial_learning_size == 0:
            grouped_ordering = [training_order[i:i + n_classes_per_step] for i in range(0, len(training_order), n_classes_per_step)]
        else:
            grouped_ordering = []
            grouped_ordering.append([training_order[i]for i in range(initial_learning_size)])
            grouped_ordering.append(training_order[i:i + n_classes_per_step] for i in range(initial_learning_size, len(training_order), n_classes_per_step))

        return train_images, y_train, test_images, y_test, grouped_ordering, training_order

    @staticmethod
    def get_step_data(x, y, classes_to_extract, replay=False):
        step_x = []
        step_y = []
        if not replay:
            for cls in classes_to_extract:
                indices = np.argwhere(y==cls)
                indices = [i[0] for i in indices]
                step_x.extend(x[indices])
                step_y.extend(y[indices])

        perm = np.arange(0, len(step_y), 1)
        np.random.shuffle(perm)
        step_x = np.asarray(step_x[:])
        step_y = np.asarray(step_y[:])

        step_x = step_x[perm]
        step_y = step_y[perm]

        return step_x, step_y
