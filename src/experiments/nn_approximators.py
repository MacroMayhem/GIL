__author__ = "Aditya Singh"
__version__ = "0.1"
import tensorflow as tf
from src.utility.dataloader import DataLoader, Strategy
from src.utility.Models import resnet18
from src.replay_library import ReplayManager
from src.utility.Models import DNet, MappingEstimator
import numpy as np
from src.utility.utils import Metric
import os
from sklearn.model_selection import train_test_split
###### Training Parameters ######
dataset = 'mnist'
per_class_samples_retained = 20
feature_dims = 128
classes_per_addition_step = 2
learning_epochs = 20
model_arch = 'resnet-18'
batch_size = 128
shuffle_buffer_size = 10000
replay_buffer_size = 2*classes_per_addition_step*per_class_samples_retained
# one of clf-clu, kld, ws
strategy = [Strategy.CLF, Strategy.CLU]
#################################

#Creating output dir
input_dir = os.path.join('../output', dataset, 'StepClasses:{}'.format(classes_per_addition_step)
                               , '_'.join(strat.name for strat in strategy), str(feature_dims), model_arch)
output_dir = os.path.join(input_dir, 'incr_distance')
os.makedirs(output_dir, exist_ok=True)


train_images, y_train, test_images, y_test, training_order, sequence = DataLoader.load_data(dataset, n_classes_per_step=classes_per_addition_step)
# REQUIRED FOR MNIST
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype("float32") / 255.0
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype("float32") / 255.0

#np.random is seeded by dataloader
train_images, val_images, y_train, y_val = train_test_split(train_images, y_train, test_size=0.2)

metric_manager = Metric(output_dir)

print('The training order for files is {}'.format(training_order))

training_order = [list(i) for i in training_order]
base_exp_classes = training_order[0]
updated_ordering = [training_order[0]]
for idx in range(1, len(training_order)):
    updated_ordering.append(updated_ordering[idx-1]+training_order[idx])

num_classes = len(np.unique(y_train))
resnet_model = resnet18(input_dims=(28, 28, 1), feature_dims=feature_dims, num_classes=num_classes, strategy=strategy)
model = DNet(model=resnet_model, strategy=strategy)

deep_approximators = {}
if Strategy.DEEP_APPROX in strategy:
    for cls in sequence:
        deep_approximators[cls] = MappingEstimator(cls_label=cls, is_input_normalised=Strategy.NORM in strategy, ndims=feature_dims)