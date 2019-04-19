import tensorflow as tf
from keras import backend as K
import numpy as np
import os
import random as rn
from activations import sparsemax, sparsegen_lin

'''
labels is tensor having one-hot encoding of labels
'''
def sparsemax_loss():
  def sparsemax_loss(labels, z):
      sm=sparsemax(z, epsilon=0.0)
      support = tf.cast(sm>0, tf.float32)
  
      #sum over support 0.5 * (z_j^2 - \tau^2)
      sum_s = support * sm * (z - 0.5 * sm)
  
      # -z_k + 0.5 ||q||^2
      q_part = labels * (0.5 * labels - z)
  
      return tf.reduce_sum(sum_s+q_part, axis = -1)
  return sparsemax_loss


def sparsegen_lin_loss(lamda=0.0):
    def sparsegen_lin_loss(labels, z):
        sm = sparsegen_lin(z, lamda=lamda, epsilon=0.0)
        support = tf.cast(sm > 0, tf.float32)

        # sum over support 0.5 * (z_j^2 - \tau^2)
        sum_s = support * sm * (z - 0.5 * (1-lamda) * sm)

        # -z_k + 0.5 ||q||^2
        q_part = labels * ((1-lamda) * 0.5 * labels - z)

        return tf.reduce_sum(sum_s + q_part, axis=-1)

    return sparsegen_lin_loss


'''
labels is tensor having one-hot encoding of labels
'''
def sparse_hourglass_proxy_loss(qx=10.0):
  def sparse_hourglass_proxy_loss(labels, z):
      dim = tf.shape(z)[1]
      batch_size = tf.shape(z)[0]
  
      #compute alpha
      q = K.constant(qx)
      dim_float = tf.cast(dim, tf.float32)
      z_sum = K.reshape(K.abs(K.sum(z, axis=1)), (batch_size, 1))
      one_vec = tf.ones([1, dim])
      z_sum = K.dot(z_sum, one_vec)
      den = z_sum + q*dim_float
      alpha = (1 + q*dim_float)/den
      z = alpha * z
  
      return sparsemax_loss(z, labels)
  return sparse_hourglass_proxy_loss


'''
label_indices should be between 1 to K, it is a Tensor of indices
'''

def sparse_hourglass_multiclass_loss(qx=10.0):

  def sparse_hourglass_multiclass_loss(label_indices, z):
      batch_size = tf.shape(z)[0]
      dim = tf.shape(z)[1]
      labels = tf.one_hot(label_indices - 1, depth=dim)
      non_labels = 1 - labels
  
      # compute alpha
      q = K.constant(qx)
      dim_float = tf.cast(dim, tf.float32)
      z_sum = K.reshape(K.abs(K.sum(z, axis=1)), (batch_size, 1))
      one_vec = tf.ones([1, dim])
      z_sum = K.dot(z_sum, one_vec)
      den = z_sum + q * dim_float
      alpha = (1 + q * dim_float) / den
  
      true_val = K.dot(tf.reduce_sum(alpha * labels * z, axis=1, keep_dims=True), tf.ones((1, dim)))
      other_val = alpha * z
  
      vio = tf.maximum(non_labels - true_val + other_val, 0.0) / alpha
      v_sum = tf.reduce_sum(vio, axis = -1)
  
      return v_sum
  return sparse_hourglass_multiclass_loss
  
'''
batch_size 1 version.
def shg_multilabel_loss(z, rho, qx=10.0):
    batch_size = tf.shape(z)[0]
    dim = tf.shape(z)[1]

    # compute alpha
    q = K.constant(qx)
    dim_float = tf.cast(dim, tf.float32)
    z_sum = K.reshape(K.abs(K.sum(z, axis=1)), (batch_size, 1))
    one_vec = tf.ones([1, dim])
    z_sum = K.dot(z_sum, one_vec)
    den = z_sum + q * dim_float
    alpha = (1 + q * dim_float) / den

    # computing alpha(z_i - z_j) and rho_i - rho_j :
    # i on rows and j on columns
    z_diff = K.dot(K.transpose(z), tf.ones((1, dim))) - K.dot(tf.ones((dim, 1)), z)
    rho_diff = K.dot(K.transpose(rho), tf.ones((1, dim))) - K.dot(tf.ones((dim, 1)), rho)
    rho_diff = rho_diff / alpha

    # absolute loss
    abs_loss = K.abs(rho_diff - z_diff) / 2  # dividing by 2 to counter double counting.

    # crammer-singer hinge loss (kind of)
    hinge_loss = K.maximum(rho_diff - z_diff, 0.0)

    # selection matrices to choose which equation to use
    rho_ceil = tf.ceil(rho)  # 1 for non_zero probabilities
    both_pos = K.dot(K.transpose(rho_ceil), tf.ones((1, dim))) * K.dot(tf.ones((dim, 1)), rho_ceil)
    one_pos = K.dot(K.transpose(rho_ceil), tf.ones((1, dim))) * (1 - K.dot(tf.ones((dim, 1)), rho_ceil))

    loss = K.sum(both_pos * abs_loss + one_pos * hinge_loss)
    return loss
'''

'''
alpha to be fixed
'''
def sparse_hourglass_multilabel_loss(qx=10.0):
	# rho = y_true and z = y_pred
	def sparse_hourglass_multilabel_loss(rho, z):
		batch_size = tf.shape(z)[0]
		dim = tf.shape(z)[1]

		# compute alpha
		q = K.constant(qx)
		dim_float = tf.cast(dim, tf.float32)
		z_sum = K.reshape(K.abs(K.sum(z, axis=1)), (batch_size, 1))
		one_vec = tf.ones([1, dim])
		z_sum = K.dot(z_sum, one_vec)
		den = z_sum + q * dim_float
		alpha = (1 + q * dim_float) / den
		alpha = K.dot(tf.expand_dims(alpha, 2), tf.ones((1, dim)))

		# computing alpha(z_i - z_j) and rho_i - rho_j :
		# i on rows and j on columns
		# output dims below is batch_size x dim x dim
		z_diff = tf.matmul(tf.expand_dims(z, dim=2), tf.ones((batch_size, 1, dim))) - tf.matmul(tf.ones((batch_size, dim, 1)), tf.expand_dims(z, dim=1))
		rho_diff = tf.matmul(tf.expand_dims(rho, dim=2), tf.ones((batch_size, 1, dim))) - tf.matmul(tf.ones((batch_size, dim, 1)), tf.expand_dims(rho, dim=1))
		rho_diff = rho_diff / alpha

		# absolute loss
		abs_loss = K.abs(rho_diff - z_diff) / 2  # dividing by 2 to counter double counting.

		# crammer-singer hinge loss (kind of)
		hinge_loss = K.maximum(rho_diff - z_diff, 0.0)

		# selection matrices to choose which equation to use
		rho_ceil = tf.ceil(rho)  # 1 for non_zero probabilities
		both_pos = tf.matmul(tf.expand_dims(rho_ceil, dim=2), tf.ones((batch_size, 1, dim))) * tf.matmul(tf.ones((batch_size, dim, 1)), tf.expand_dims(rho_ceil, dim=1))
		one_pos = tf.matmul(tf.expand_dims(rho_ceil, dim=2), tf.ones((batch_size, 1, dim))) * (1 - tf.matmul(tf.ones((batch_size, dim, 1)), tf.expand_dims(rho_ceil, dim=1)))

		loss = K.sum(both_pos * abs_loss + one_pos * hinge_loss, axis = -1)
		return loss
	return sparse_hourglass_multilabel_loss

def sparsegen_lin_multilabel_loss(lamda=0.0):
	# rho = y_true and z = y_pred
	def sparsegen_lin_multilabel_loss(rho, z):
		batch_size = tf.shape(z)[0]
		dim = tf.shape(z)[1]
    		z = z * (1/(1-K.constant(lamda)))

		# computing alpha(z_i - z_j) and rho_i - rho_j :
		# i on rows and j on columns
		# output dims below is batch_size x dim x dim
		z_diff = tf.matmul(tf.expand_dims(z, dim=2), tf.ones((batch_size, 1, dim))) - tf.matmul(tf.ones((batch_size, dim, 1)), tf.expand_dims(z, dim=1))
		rho_diff = tf.matmul(tf.expand_dims(rho, dim=2), tf.ones((batch_size, 1, dim))) - tf.matmul(tf.ones((batch_size, dim, 1)), tf.expand_dims(rho, dim=1))

		# absolute loss
		abs_loss = K.abs(rho_diff - z_diff) / 2  # dividing by 2 to counter double counting.

		# crammer-singer hinge loss (kind of)
		hinge_loss = K.maximum(rho_diff - z_diff, 0.0)

		# selection matrices to choose which equation to use
		rho_ceil = tf.ceil(rho)  # 1 for non_zero probabilities
		both_pos = tf.matmul(tf.expand_dims(rho_ceil, dim=2), tf.ones((batch_size, 1, dim))) * tf.matmul(tf.ones((batch_size, dim, 1)), tf.expand_dims(rho_ceil, dim=1))
		one_pos = tf.matmul(tf.expand_dims(rho_ceil, dim=2), tf.ones((batch_size, 1, dim))) * (1 - tf.matmul(tf.ones((batch_size, dim, 1)), tf.expand_dims(rho_ceil, dim=1)))

		loss = K.sum(both_pos * abs_loss + one_pos * hinge_loss, axis = -1)
		return loss
	return sparsegen_lin_multilabel_loss
