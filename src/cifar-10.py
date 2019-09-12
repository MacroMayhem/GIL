__author__ = "Aditya Singh"
__version__ = "0.1"

"""
Addition step defines the stage when new classes are added for training
"""
import tensorflow as tf
from src.utility.dataloader import DataLoader, Strategy
from src.utility.Models import resnet18
from src.replay_library import ReplayManager
from src.utility.Models import DNet
import numpy as np
from src.utility.utils import modified_to_categorical, Metric
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import copy
import operator

###### Training Parameters ######
dataset = 'cifar-10'
per_class_samples_retained = 20
feature_dims = 128
classes_per_addition_step = 2
learning_epochs = 1
model_arch = 'resnet-18'
batch_size = 128
shuffle_buffer_size = 10000
replay_buffer_size = 2*classes_per_addition_step*per_class_samples_retained
# one of clf-clu, kld, ws
strategy = [Strategy.CLF]
#################################

#Creating output dir
base_output_dir = os.path.join('../output', dataset, 'StepClasses:{}'.format(classes_per_addition_step)
                               , '_'.join(strat.name for strat in strategy), str(feature_dims), model_arch)
os.makedirs(base_output_dir)


train_images, y_train, test_images, y_test, training_order, sequence = DataLoader.load_data(dataset, n_classes_per_step=classes_per_addition_step)

num_classes = len(np.unique(y_train))
# REQUIRED FOR MNIST
train_images = train_images.reshape(train_images.shape[0], 32, 32, 3).astype("float32") / 255.0
test_images = test_images.reshape(test_images.shape[0], 32, 32, 3).astype("float32") / 255.0

train_images, val_images, y_train, y_val = train_test_split(train_images, y_train, test_size=0.2)

resnet_model = resnet18(input_dims=(32, 32, 3), feature_dims=feature_dims, num_classes=num_classes, strategy=strategy)

model = DNet(model=resnet_model, strategy=strategy)
replay_manager = ReplayManager()
metric_manager = Metric(base_output_dir)

old_classes = []

for idx, training_batch in enumerate(training_order):

    #Learning step training data
    step_x, step_y = DataLoader.get_step_data(train_images, y_train, training_batch)
    categorical_y = modified_to_categorical(step_y, sequence)

    #Learning step validation data
    val_classes = old_classes + list(training_batch)
    val_x, val_y = DataLoader.get_step_data(val_images, y_val, val_classes)
    val_y_1h = modified_to_categorical(val_y, val_classes)

    x_replay = []
    y_replay = []

    jumbled_old_classes = copy.deepcopy(old_classes)    #Shuffle the classes around to move the instances all together

    if idx > 0:

        #Add output units to the old model

        # Fetch Replay data
        np.random.shuffle(jumbled_old_classes)
        for cls in jumbled_old_classes:
            replay_data = replay_manager.get_data(cls)
            x_replay.extend(replay_data.selected_samples)
            y_replay.extend([cls for i in range(replay_data.selected_samples.shape[0])])
        y_replay_1h = modified_to_categorical(y_replay, old_classes + list(training_batch))
        y_replay = np.asarray(y_replay)
        x_replay = np.asarray(x_replay)

        del replay_data

    # Learning + Replay
    best_model = None
    best_metric_to_track = 10000
    comparator = operator.lt

    for iter in range(learning_epochs+len(old_classes)//2):
        # To shuffle data in batches
        new_data = (
            tf.data.Dataset.from_tensor_slices((step_x, categorical_y, step_y)).shuffle(shuffle_buffer_size).batch(
                batch_size))
        total_batches = int(step_x.shape[0]/batch_size)

        # In case the replay size becomes significant
        replay_batching = False
        if len(x_replay)//total_batches >= batch_size//classes_per_addition_step:
            replay_batching = True
            replay_batch = len(x_replay)//total_batches
            old_data = (tf.data.Dataset.from_tensor_slices((x_replay, y_replay_1h, y_replay)).shuffle(replay_buffer_size).batch(replay_batch))

        with tqdm(total=total_batches) as pbar:
            for c_batch, (x, y_1h, y) in enumerate(new_data):
                if idx > 0:
                    if not replay_batching:
                        b_x, b_y_1h, b_y = x_replay, y_replay_1h, y_replay
                    else:
                        b_x, b_y_1h, b_y = old_data[c_batch]
                        b_x = b_x.numpy()
                        b_y_1h = b_y_1h.numpy()
                        b_y = b_y.numpy()

                    if Strategy.DIST in strategy:
                        inv_covs = [replay_manager.get_inv_cov(i) for i in b_y]
                        means = [replay_manager.get_mean(i) for i in b_y]
                    if Strategy.MIX_REPLAY in strategy:
                        x = np.concatenate([x.numpy(), b_x])
                        y_1h = np.concatenate([y_1h.numpy(), b_y_1h])
                        y = np.concatenate([y.numpy(), b_y])
                model.train(x, y_1h, y)
                if Strategy.DIST in strategy and idx > 0:
                    model.train_replay(b_x, b_y_1h, means, inv_covs)

                pbar.update(1)

        #Validation after epoch
        val_predictions, val_features = model.test(val_x)
        metric_manager.embedding_plot(val_features, val_y[:,0], 'val_embed_ls-{}_epoch-{}.png'.format(idx, iter))
        model.val_accuracy(val_predictions, val_y_1h)

        best_metric_to_track = model.update_best(model.clf_loss_metric.result(), best_metric_to_track, model.model.get_weights(), comparator)

        if idx > 0:
            mh_distances, mh_labels = replay_manager.get_mhdistance(val_features.numpy(), val_y[:,0])
            metric_manager.gen_box_plots(mh_distances, mh_labels, 'val_box_ls-{}_epoch-{}.png'.format(idx, iter))
            mean_distances, mean_labels = replay_manager.get_mean_distance(val_features.numpy(), val_y[:,0])
            metric_manager.gen_box_plots(mean_distances, mean_labels, 'val_box_euclidean_ls-{}_epoch-{}.png'.format(idx, iter))
            #metric_manager.gen_QQplot(val_features.numpy(), val_y, 'val_QQ_ls-{}_epoch-{}'.format(idx, iter))

        model.reset_recorded_metrics(iter)

    #Set the best model so far
    if model.best_weights_so_far is not None:
        model.model.set_weights(model.best_weights_so_far)
    else:
        raise AssertionError('Best weights cannot be None')
    model.best_weights_so_far = None

    if idx > 0:
        del jumbled_old_classes, y_replay_1h, y_replay, x_replay
    #del x, y, y_1h

    # Populate Replay Buffer
    for cls in training_batch:
        cls_x, _ = DataLoader.get_step_data(train_images, y_train, [cls])
        replay_manager.init_replay_library(cls)
        soft_labels, features = model.test(cls_x)
        replay_manager.fit_gaussian(features, cls)
        old_classes.append(cls)
        replay_manager.populate_replay_library(cls_x, cls)

    #Recompute validation using best model
    val_predictions, val_features = model.test(val_x)
    mh_distances, mh_labels = replay_manager.get_mhdistance(val_features.numpy(), val_y[:,0])
    metric_manager.gen_box_plots(mh_distances, mh_labels, 'val_box_ls-{}_epoch-END.png'.format(idx))
    mean_distances, mean_labels = replay_manager.get_mean_distance(val_features.numpy(), val_y[:,0])
    metric_manager.gen_box_plots(mean_distances, mean_labels, 'val_box_euclidean_ls-{}_epoch-END.png'.format(idx))
    #For measure of fit to validation


    # Evaluate on all classes
    test_x, test_y = DataLoader.get_step_data(test_images, y_test, old_classes)
    test_y_1h = modified_to_categorical(test_y, old_classes)
    predictions, prediction_features = model.test(test_x)
    prediction_label = np.argmax(predictions, axis=-1)
    model.test_accuracy(test_y_1h, predictions)

    print('Overall Accuracy: {}'.format(model.test_accuracy.result()))
    metric_manager.embedding_plot(prediction_features, test_y[:,0], 'test_embed_ls-{}_epoch-{}.png'.format(idx, iter))
    metric_manager.update_average_accuracy(model.test_accuracy.result())
    model.test_accuracy.reset_states()
    metric_manager.compute_confusion_matrix(test_y, prediction_label, old_classes, 'test_cm_{}.csv'.format(idx))
    mh_distances, mh_labels = replay_manager.get_mhdistance(prediction_features.numpy(), test_y[:,0])
    metric_manager.gen_box_plots(mh_distances, mh_labels, 'test_box_ls-{}_epoch-{}.png'.format(idx, iter))
    mean_distances, mean_labels = replay_manager.get_mean_distance(prediction_features.numpy(), test_y[:,0])
    metric_manager.gen_box_plots(mean_distances, mean_labels, 'test_box_euclidean_ls-{}_epoch-{}.png'.format(idx, iter))
    #metric_manager.gen_QQplot(prediction_features.numpy(), test_y, 'test_QQ_ls-{}_epoch-{}'.format(idx, iter))

    del test_x, test_y, test_y_1h, prediction_features, predictions

    #Save the model
    model.model.save(filepath=os.path.join(base_output_dir, 'iter_{}.h5'.format(idx)))
