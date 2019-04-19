import tensorflow as tf
from keras import backend as K
import numpy as np


# custom activation functions
def sparsemax(z, epsilon=1e-7):
    dim = tf.shape(z)[1]
    batch_size = tf.shape(z)[0]
    dim_float = tf.cast(dim, tf.float32)

    d_range = tf.cast(tf.range(1, dim+1), tf.float32)
    b_range = tf.cast(tf.range(0, batch_size), tf.int32)

    z_sorted, _ = tf.nn.top_k(z, dim)
    z_cumsum = K.cumsum(z_sorted, axis=1)
    z_check = 1 + (d_range * z_sorted) > z_cumsum
    z_check = tf.cast(z_check, tf.int32)
    k_z = K.sum(z_check, axis=1)
    indices = tf.stack([b_range, k_z - 1], axis=1)
    tau_sum = tf.gather_nd(z_cumsum, indices)
    tau_z = (tau_sum - 1)/tf.cast(k_z, tf.float32)
    p = tf.maximum(0.0, z - tau_z[:, tf.newaxis])

    # Don't want any probability to go zero.
    epsilon = K.constant(epsilon)
    p = (p + epsilon)/(1. + dim_float*epsilon)

    return p

def sparsegen_lin(z, lamda=0.0, epsilon=1e-7):
    dim = tf.shape(z)[1]
    batch_size = tf.shape(z)[0]
    dim_float = tf.cast(dim, tf.float32)

    d_range = tf.cast(tf.range(1, dim+1), tf.float32)
    b_range = tf.cast(tf.range(0, batch_size), tf.int32)

    z = z * (1/(1-K.constant(lamda)))
    z_sorted, _ = tf.nn.top_k(z, dim)
    z_cumsum = K.cumsum(z_sorted, axis=1)
    z_check = 1 + (d_range * z_sorted) > z_cumsum
    z_check = tf.cast(z_check, tf.int32)
    k_z = K.sum(z_check, axis=1)
    indices = tf.stack([b_range, k_z - 1], axis=1)
    tau_sum = tf.gather_nd(z_cumsum, indices)
    tau_z = (tau_sum - 1)/tf.cast(k_z, tf.float32)
    p = tf.maximum(0.0, z - tau_z[:, tf.newaxis])

    # Don't want any probability to go zero.
    epsilon = K.constant(epsilon)
    p = (p + epsilon)/(1. + dim_float*epsilon)

    return p

def sparse_hourglass(z, qx=10.0, lamda= 0.0, epsilon=0):
    dim = tf.shape(z)[1]
    batch_size = tf.shape(z)[0]
    d_range = tf.cast(tf.range(1, dim+1), tf.float32)
    b_range = tf.cast(tf.range(0, batch_size), tf.int32)

    # computing alpha
    q = K.constant(qx)
    dim_float = tf.cast(dim, tf.float32)
    z_sum = K.reshape(K.abs(K.sum(z, axis=1)), (batch_size, 1))
    one_vec = tf.ones([1, dim])
    z_sum = K.dot(z_sum, one_vec)
    den = z_sum + q*dim_float
    alpha = (1 + q*dim_float)/den
    z = alpha * z

    # Euclidean projection onto simplex
    z_sorted, _ = tf.nn.top_k(z, dim)
    z_cumsum = K.cumsum(z_sorted, axis=1)
    z_check = 1 + (d_range * z_sorted) > z_cumsum
    z_check = tf.cast(z_check, tf.int32)
    k_z = K.sum(z_check, axis=1)
    indices = tf.stack([b_range, k_z - 1], axis=1)
    tau_sum = tf.gather_nd(z_cumsum, indices)
    tau_z = (tau_sum - 1)/tf.cast(k_z, tf.float32)
    p = tf.maximum(0.0, z - tau_z[:, tf.newaxis])

    # Don't want any probability to go zero.
    epsilon = K.constant(epsilon)
    p = (p + epsilon)/(1. + dim_float*epsilon)

    return p
