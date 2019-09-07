__author__ = "Aditya Singh"
__version__ = "0.1"

"""
Addition step defines the stage when new classes are added for training
"""
import tensorflow as tf
from src.utility.dataloader import DataLoader, Strategy
from src.utility.Models import resnet18, add_output_units
from src.replay_library import ReplayManager
from src.utility.Models import DNet
import numpy as np
from src.utility.utils import custom_confusion_metrics, modified_to_categorical
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import copy

###### Training Parameters ######
dataset = 'mnist'
per_class_samples_retained = 20
feature_dims = 64
classes_per_addition_step = 2
learning_epochs = 0
model_arch = 'resnet-18'
batch_size = 128
shuffle_buffer_size = 10000
replay_buffer_size = 2*classes_per_addition_step*per_class_samples_retained
# one of clf-clu, kld, ws
second_stage_strategy = Strategy.CLF_CLU
#################################

#Creating output dir
base_output_dir = os.path.join('../output', dataset, second_stage_strategy.name, str(feature_dims), model_arch)
os.makedirs(base_output_dir)


train_images, y_train, test_images, y_test, training_order = DataLoader.load_data('mnist',
                                                                                  n_classes_per_step=classes_per_addition_step)
# REQUIRED FOR MNIST
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype("float32") / 255.0
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype("float32") / 255.0

train_images, val_images, y_train, y_val = train_test_split(train_images, y_train, test_size=0.1)

base_model, inp_layer = resnet18(input_dims=(28, 28, 1), feature_dims=feature_dims)
main_model = add_output_units(base_model=base_model, input_layer=inp_layer, number_of_extra_units=2,
                              iteration_number=0)
print(main_model.summary())

model = DNet()
model.replace_model(main_model)
replay_manager = ReplayManager()

old_classes = []

average_accuracy = []

for idx, training_batch in enumerate(training_order):

    #Learning step training data
    step_x, step_y = DataLoader.get_step_data(train_images, y_train, training_batch)
    categorical_y = modified_to_categorical(step_y, old_classes + list(training_batch))

    #Learning step validation data
    val_x, val_y = DataLoader.get_step_data(val_images, y_val, training_batch)
    val_y_1h = modified_to_categorical(val_y, old_classes + list(training_batch))

    x_replay = []
    y_replay = []
    jumbled_old_classes = copy.deepcopy(old_classes)    #Shuffle the classes around to move the instances all together

    best_model_weights = None
    if idx > 0:

        #Add output units to the old model
        new_model = add_output_units(base_model=model.model, input_layer=inp_layer
                                     , number_of_extra_units=len(training_batch), iteration_number=idx)
        model.replace_model(new_model)

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
    best_val_acc = 0
    for iter in range(learning_epochs+len(old_classes)):
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
                        x = np.concatenate([x.numpy(), x_replay])
                        y_1h = np.concatenate([y_1h.numpy(), y_replay_1h])
                        y = np.concatenate([y.numpy(), y_replay])
                    else:
                        b_x, b_y_1h, b_y = old_data[c_batch]
                        x = np.concatenate([x.numpy(), b_x.numpy()])
                        y_1h = np.concatenate([y_1h.numpy(), b_y_1h.numpy()])
                        y = np.concatenate([y.numpy(), b_y.nump()])

                pbar.update(1)
                model.train(x, y_1h, y)
                val_predictions, _ = model.test(val_x)
                model.val_accuracy(val_predictions, val_y_1h)
        if best_val_acc < model.val_accuracy.result():
            best_val_acc = model.val_accuracy.result()
            best_model_weights = copy.deepcopy(model.model.get_weights())

        model.reset_recorded_metrics(iter)

    #Set the best model so far
    if best_model_weights is not None:
        model.model.set_weights(best_model_weights)
    del best_model_weights

    if idx > 0:
        del jumbled_old_classes, y_replay_1h, y_replay, x_replay
    #del x, y, y_1h

    # Populate Replay Buffer
    for cls in training_batch:
        cls_x, _ = DataLoader.get_step_data(train_images, y_train, [cls])
        replay_manager.populate_replay_library(cls_x, cls)
        soft_labels, features = model.test(cls_x)
        replay_manager.fit_gaussian(features, cls)
        old_classes.append(cls)

    # Evaluate on all classes
    test_x, test_y = DataLoader.get_step_data(test_images, y_test, old_classes)
    test_y_1h = modified_to_categorical(test_y, old_classes)
    predictions, _ = model.test(test_x)
    prediction_label = np.argmax(predictions, axis=-1)
    model.test_accuracy(test_y_1h, predictions)
    print('Overall Accuracy: {}'.format(model.test_accuracy.result()))
    average_accuracy.append(model.test_accuracy.result())
    model.test_accuracy.reset_states()
    conf_matrix = custom_confusion_metrics(test_y, prediction_label, old_classes)
    np.savetxt(os.path.join(base_output_dir, 'val_cm_{}.csv'.format(idx)), conf_matrix, delimiter=',')
    del test_x, test_y

    #Save the model
    model.model.save(filepath=os.path.join(base_output_dir, 'iter_{}.h5'.format(idx)))

print((i, average_accuracy[i]) for i in range(len(average_accuracy)))
