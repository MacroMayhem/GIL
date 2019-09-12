import tensorflow as tf
from tensorflow.keras import layers, Sequential
import tensorflow.keras as keras
from src.losses import cluster_loss, mahalanobis_distance
from src.utility.dataloader import Strategy
import operator
import copy

class DNet(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super(DNet, self).__init__(*args)
        self.model = kwargs.get('model', None)

        self.strategy = kwargs.get('strategy', [Strategy.CLF, Strategy.CLU])
        self.clf_optimizer = tf.keras.optimizers.SGD(learning_rate=kwargs.get('lr_clf', 0.0001))
        self.clu_optimizer = tf.keras.optimizers.SGD(learning_rate=kwargs.get('lr_clu', 0.0001))
        self.dist_optimizer = tf.keras.optimizers.SGD(learning_rate=kwargs.get('lr_dist', 0.0001))

        self.clf_loss = tf.losses.categorical_crossentropy
        self.dist_loss = mahalanobis_distance
        self.clu_loss = cluster_loss

        self.combined_loss_metric = tf.keras.metrics.Mean(name='combined_loss')
        self.clf_loss_metric = tf.keras.metrics.Mean(name='clf_loss')
        self.clu_loss_metric = tf.keras.metrics.Mean(name='clu_loss')
        self.dist_loss_metric = tf.keras.metrics.Mean(name='dist_loss')
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        self.test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
        self.val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')

        self.best_weights_so_far = None

    def update_best(self, epoch_metric_value, best_metric_value, current_weights, comparator=operator.lt):
        if comparator(epoch_metric_value, best_metric_value):
            self.best_weights_so_far = copy.deepcopy(current_weights)
            return epoch_metric_value
        else:
            return best_metric_value

    def replace_model(self, new_model):
        self.model = new_model

    def clf_train_step(self, images, labels, setMetrics=True):
        predictions, _ = self.model(images)
        clf_loss = self.clf_loss(labels, predictions)
        if setMetrics:
            self.train_accuracy(labels, predictions)
        return clf_loss

    def clu_train_step(self, images, labels):
        _, norm_features = self.model(images)
        clu_loss = self.clu_loss(labels, norm_features, margin_multiplier=1.0)
        return clu_loss

    def dist_train_step(self, images, means, inv_cov, flags=None):
        _, norm_features = self.model(images)
        dist_loss = self.dist_loss(norm_features, means, inv_cov, flags)
        return dist_loss

    def compute_gradients(self, x, one_hot_y, y):
        with tf.GradientTape() as clf_tape, tf.GradientTape() as clu_tape:
            clf_loss = self.clf_train_step(x, one_hot_y)
            clu_loss = self.clu_train_step(x, y)
        if Strategy.CLF in self.strategy:
            clf_gradients = clf_tape.gradient(clf_loss, self.model.trainable_variables)
        else:
            clf_gradients = None
        if Strategy.CLU in self.strategy:
            clu_gradients = clu_tape.gradient(clu_loss, self.model.trainable_variables)
        else:
            clu_gradients = None

        self.update_metrics(clf_loss, clu_loss)
        return clf_gradients, clu_gradients

    def compute_replay_gradients(self, x, y_1h, means, inv_cov, flags=None):
        with tf.GradientTape() as clf_tape, tf.GradientTape() as dist_tape:
            clf_loss = self.clf_train_step(x, y_1h)
            dist_loss = self.dist_train_step(x, means, inv_cov, flags)
            n_dist_loss = tf.reshape(dist_loss/tf.keras.backend.max(tf.math.abs(dist_loss), axis=0), [dist_loss.shape[0]])
        clf_gradients = clf_tape.gradient(clf_loss, self.model.trainable_variables)
        dist_gradient = dist_tape.gradient(n_dist_loss, self.model.trainable_variables)

        self.update_metrics(clf_loss=clf_loss, dist_loss=n_dist_loss)
        return clf_gradients, dist_gradient

    def update_metrics(self, clf_loss=None, clu_loss=None, dist_loss=None):
        if clf_loss is not None and clu_loss is not None:
            self.combined_loss_metric(clf_loss + clu_loss)
        if clf_loss is not None:
            self.clf_loss_metric(clf_loss)
        if clu_loss is not None:
            self.clu_loss_metric(clu_loss)
        if dist_loss is not None:
            self.dist_loss_metric(dist_loss)

    def reset_recorded_metrics(self, iter):
        template = 'Epoch {}, Loss: {}, CLF Loss: {}, CLU Loss: {}, DIST Loss: {}, Train Acc: {}, Val Acc: {}'
        print(template.format(iter + 1, self.combined_loss_metric.result(), self.clf_loss_metric.result()
                              , self.clu_loss_metric.result(), self.dist_loss_metric.result()
                              , self.train_accuracy.result(), self.val_accuracy.result()))
        self.combined_loss_metric.reset_states()
        self.clf_loss_metric.reset_states()
        self.clu_loss_metric.reset_states()
        self.dist_loss_metric.reset_states()
        self.train_accuracy.reset_states()
        self.val_accuracy.reset_states()

    @tf.function
    def train_replay(self, train_x, train_1h_y, inv_cov, means, flags=None):
        clf_gradients, dist_gradients = self.compute_replay_gradients(train_x, train_1h_y, inv_cov, means, flags)
        self.clf_optimizer.apply_gradients(zip(clf_gradients, self.model.trainable_variables))
        self.dist_optimizer.apply_gradients(zip(dist_gradients, self.model.trainable_variables))

    @tf.function
    def train(self, train_x, train_1h_y, train_y):
        clf_gradients, clu_gradients = self.compute_gradients(train_x, train_1h_y, train_y)
        if Strategy.CLF in self.strategy:
            self.clf_optimizer.apply_gradients(zip(clf_gradients, self.model.trainable_variables))
        if Strategy.CLU in self.strategy:
            self.clu_optimizer.apply_gradients(zip(clu_gradients, self.model.trainable_variables))

    @tf.function
    def test(self, test_x):
        predictions, norm_features = self.model(test_x)
        return predictions, norm_features


# Copied from https://blog.csdn.net/abc13526222160/article/details/90057121
class BasicBlock:
    def __init__(self, filter_num, stride=1):

        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x:x

    def __call__(self, inputs):

        # [b, h, w, c]
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(inputs)

        output = layers.Add()([out, identity])
        output = layers.Activation('relu')(output)
        return output


def get_resnet(input_dims, feature_dims=128, resnet=18):
    if resnet == 18:
        layer_dims = [2, 2, 2, 2]
        inp_layer = layers.Input(shape=input_dims)
        x = layers.Conv2D(64, (3, 3), strides=(1, 1))(inp_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)

        x = build_resblock(x, 64, layer_dims[0])
        x = build_resblock(x, 128, layer_dims[1], stride=2)
        x = build_resblock(x, 256, layer_dims[2], stride=2)
        x = build_resblock(x, 512, layer_dims[3], stride=2)

        x = layers.GlobalAveragePooling2D()(x)
        output = layers.Dense(feature_dims)(x)
        return inp_layer, output
    elif resnet ==34:
        pass

def build_resblock(x, filter_num, blocks, stride=1):

    x = BasicBlock(filter_num, stride)(x)

    for _ in range(1, blocks):
        x = BasicBlock(filter_num, stride=1)(x)

    return x


def add_output_units(base_model, input_layer, number_of_extra_units, iteration_number=0, strategy=[Strategy.CLF]):
    fe_output = base_model.get_layer('dense').output
    if Strategy.CLU in strategy:
        norm_output = keras.layers.Lambda(lambda x: keras.backend.l2_normalize(x, axis=1))(fe_output)
    else:
        norm_output = fe_output

    if iteration_number == 0:
        ce_output = keras.layers.Dense(units=number_of_extra_units, name='output_{}'.format(iteration_number))(fe_output)
    else:
        old_layers = []
        for i in range(iteration_number):
            old_layers.append(base_model.get_layer('output_{}'.format(i)).output)
        extra_output_layer = keras.layers.Dense(units=number_of_extra_units, name='output_{}'.format(iteration_number))(fe_output)
        old_layers.append(extra_output_layer)
        ce_output = keras.layers.Concatenate()(old_layers)

    ce_output = keras.layers.Activation('softmax')(ce_output)
    return keras.Model(inputs=input_layer, outputs=[ce_output, norm_output])


def resnet18(input_dims, feature_dims, num_classes, strategy):
    inp_layer, fe_output = get_resnet(input_dims=input_dims, feature_dims=feature_dims)

    if Strategy.NORM in strategy:
        norm_output = keras.layers.Lambda(lambda x: keras.backend.l2_normalize(x, axis=1))(fe_output)
    else:
        norm_output = fe_output

    class_output_layer = keras.layers.Dense(units=num_classes, name='output')(fe_output)
    ce_output = keras.layers.Activation('softmax')(class_output_layer)

    return keras.Model(inputs=inp_layer, outputs=[ce_output, norm_output])