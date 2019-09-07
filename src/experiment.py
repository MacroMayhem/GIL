import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
np.random.seed(42)


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

dims = 4
n_samples = 32

samples = np.random.rand(n_samples, dims)
sample_mean = np.mean(samples, axis=0)
sample_cov = np.cov(samples.T)
det = np.linalg.det(sample_cov)
flags = np.random.choice([-1,1],n_samples)
print(det)
sample_inv_cov = np.linalg.inv(sample_cov)
print(is_pos_def(sample_inv_cov))
samples_mean = []
samples_cov = []
for i in range(n_samples):
    samples_mean.append(sample_mean)
    samples_cov.append(sample_inv_cov)

def mahalanobis_distance(samples, means, inv_covs, flags=None):
    """
    Computes mahalanobis distance for the batch
    :param samples: (batch, feature_dim)
    :param means: (batch, feature_dim)
    :param inv_covs: (batch, feature_dim, feature_dim)
    :param flags: (batch, 1) contains 1s and -1s ofr old and new classes
    :return:
    """
    batch_size, dims = samples.shape
    shifted_samples = tf.reshape(tf.subtract(samples, means), [batch_size, -1, dims])
    left_mult = tf.reshape(tf.linalg.matmul(shifted_samples, inv_covs), [batch_size, -1])
    output = tf.math.sqrt(tf.keras.backend.batch_dot(left_mult, tf.reshape(shifted_samples, [batch_size, -1]), axes=[1, 1]))
    if flags is not None:
        return tf.reduce_sum(tf.multiply( output, tf.reshape(flags, [batch_size, -1])), 1)
    else:
        return output

# def mahalanobis_distance(samples, samples_mean, samples_cov):
#     mean_shifted_samples = tf.subtract(samples, samples_mean)
#     reshaped_mean_shifted_samples = tf.reshape(mean_shifted_samples, [n_samples, -1, dims])
#     output = tf.linalg.matmul(reshaped_mean_shifted_samples, samples_cov)
#     output = tf.reshape(output, [n_samples, -1])
#     output = tf.math.sqrt(tf.keras.backend.batch_dot(output, mean_shifted_samples, axes=[1, 1]))
#     return output

output = mahalanobis_distance(samples, np.asarray(samples_mean), np.asarray(samples_cov), np.asarray(flags).astype(float))
print(output)