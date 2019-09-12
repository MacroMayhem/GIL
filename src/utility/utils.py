__author__ = "Aditya Singh"
__version__ = "0.1"

import numpy as np
import matplotlib.pyplot as plt
import umap
import os
import seaborn as sns
from scipy import stats
import pandas as pd


class Metric:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.average_accuracies = []

    def compute_confusion_matrix(self, test_y, prediction_label, old_classes, filename):
        cf = self._gen_cfm(test_y, prediction_label, old_classes)
        np.savetxt(os.path.join(self.base_dir, filename), cf, delimiter=',')

    def update_average_accuracy(self, acc):
        self.average_accuracies.append(acc)

    def embedding_plot(self, prediction_features, labels, filename):
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(prediction_features.numpy())
        plt.subplots(figsize=(15, 15))
        sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels, legend='full', palette=sns.color_palette("bright", len(np.unique(labels))))
        plt.savefig(os.path.join(self.base_dir, filename))
        plt.close()

    def _gen_cfm(self, y, pred_y, classes):
        confusion_matrix = np.zeros(shape=(len(classes) + 2, len(classes) + 2))
        for i in range(1, len(classes) + 1):
            confusion_matrix[0, i] = classes[i - 1]
            confusion_matrix[i, 0] = classes[i - 1]

        for y_true, y_pred in zip(y, pred_y):
            confusion_matrix[classes.index(y_true) + 1, classes.index(y_pred) + 1] += 1

        for idx, cls in enumerate(classes):
            confusion_matrix[idx + 1, -1] = (confusion_matrix[idx + 1, idx + 1] + 1e-6) / (
                        np.sum(confusion_matrix[idx + 1, 1:-1]) + 1e-6)
            confusion_matrix[-1, idx + 1] = (confusion_matrix[idx + 1, idx + 1] + 1e-6) / (
                        np.sum(confusion_matrix[1:-1, idx + 1]) + 1e-6)

        with np.printoptions(precision=4, suppress=True):
            print(np.array_repr(confusion_matrix, max_line_width=250))
        return confusion_matrix

    def gen_QQplot(self, x, labels, file_prefix):
        data = []
        for g in np.unique(labels):
            indices = np.where(np.asarray(labels) == g)
            data.append(x[indices])
        stats.probplot(data, plot=sns.mpl.pyplot)
        plt.savefig(os.path.join(self.base_dir, '{}_QQ-{}.png'.format(file_prefix, g)))
        plt.close()

    def gen_box_plots(self, vals, labels, filename):
        # data = []
        # for g in np.unique(labels):
        #     indices = np.where(np.asarray(labels) == g)
        #     data.append(vals[indices][:])
        plt.subplots(figsize=(15, 15))
        box_plot = sns.boxplot(data=vals)
        box_plot.get_figure()
        plt.savefig(os.path.join(self.base_dir, filename))
        plt.close()


def modified_to_categorical(y, classes, dtype='float32'):
    categorical_mapping = {}
    id = 0
    for cls in classes:
        categorical_mapping[cls] = id
        id += 1

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    num_classes = len(classes)
    n = y.shape[0]

    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), [categorical_mapping[sample] for sample in y]] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

