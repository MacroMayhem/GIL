__author__ = "Aditya Singh"
__version__ = "0.1"

import numpy as np

def custom_confusion_metrics(y, pred_y, classes):

    confusion_matrix = np.zeros(shape=(len(classes)+2, len(classes)+2))
    for i in range(1, len(classes)+1):
        confusion_matrix[0, i] = classes[i-1]
        confusion_matrix[i, 0] = classes[i-1]

    for y_true, y_pred in zip(y, pred_y):
        confusion_matrix[classes.index(y_true)+1, classes.index(y_pred)+1] += 1

    for idx, cls in enumerate(classes):
        confusion_matrix[idx+1, -1] = (confusion_matrix[idx+1, idx+1]+1e-6) /(np.sum(confusion_matrix[idx+1, 1:-1])+1e-6)
        confusion_matrix[-1, idx+1] = (confusion_matrix[idx+1, idx+1]+1e-6) / (np.sum(confusion_matrix[1:-1, idx+1]) + 1e-6)

    print(np.array_repr(confusion_matrix, max_line_width=50, precision=4))
    return confusion_matrix


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