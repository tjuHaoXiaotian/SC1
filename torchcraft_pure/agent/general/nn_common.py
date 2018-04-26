#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tensorflow as tf

from torchcraft_pure.agent.general.parameters import parameters


def soft_update_online_to_target(online_vars_by_name, target_vars_by_name):
    update_ops = []
    ### theta_prime = tau * theta + (1-tau) * theta_prime
    for (online_var_name, online_var) in online_vars_by_name.items():
        target_var = target_vars_by_name[online_var_name]
        theta_prime = parameters.tau * online_var + (1 - parameters.tau) * target_var
        assign_op = tf.assign(ref=target_var, value=theta_prime)
        update_ops.append(assign_op)

    return tf.group(*update_ops)


def copy_online_to_target(online_vars_by_name, target_vars_by_name):
    copy_ops = []
    ### theta_q -> theta_q_prime
    for (online_var_name, online_var) in online_vars_by_name.items():
        target_var = target_vars_by_name[online_var_name]
        assign_op = tf.assign(ref=target_var, value=online_var)
        copy_ops.append(assign_op)

    return tf.group(*copy_ops)


def conv2d(name, input, ks, stride, mean=0., stddev=0.1):
    with tf.name_scope(name):
        with tf.variable_scope(name):
            w = tf.get_variable('%s-w' % name, shape=ks,
                                initializer=tf.truncated_normal_initializer(mean=mean, stddev=stddev))
            b = tf.get_variable('%s-b' % name, shape=[ks[-1]],
                                initializer=tf.constant_initializer(value=0.01))
            out = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1], padding='SAME', name='%s-conv' % name)
            out = tf.nn.bias_add(out, b, name='%s-bias_add' % name)
    return out


def make_conv_bn_relu(name, input, ks, stride, is_training, mean=0., stddev=0.1):
    out = conv2d('%s-conv' % name, input, ks, stride, mean=mean, stddev=stddev)
    out = tf.layers.batch_normalization(out, name='%s-batch_norm' % name, training=is_training)
    out = tf.nn.relu(out, name='%s-relu' % name)
    return out


def make_fc(name, input, ks, keep_prob,
            activation_fn=None, activation_fn_name=None,
            batch_norm=False, is_training=True,
            mean=0., stddev=0.1):
    with tf.name_scope(name):
        with tf.variable_scope(name):
            w = tf.get_variable('%s-w' % name, shape=ks,
                                initializer=tf.truncated_normal_initializer(mean=mean, stddev=stddev))
            out = tf.matmul(input, w, name='%s-mat' % name)

            if batch_norm and activation_fn is not None:
                out = tf.layers.batch_normalization(out, name='%s-batch_norm' % name, training=is_training)
            else:
                b = tf.get_variable('%s-b' % name, shape=[ks[-1]],
                                    initializer=tf.constant_initializer(value=0.01))
                out = tf.nn.bias_add(out, b, name='%s-bias_add' % name)
            if activation_fn is not None:
                out = activation_fn(out, name='%s-%s' % (name, activation_fn_name))
            # out = tf.nn.dropout(out, keep_prob, name='%s-drop' % name)
    return out