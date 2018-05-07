#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tensorflow as tf
import tensorflow.contrib as tc


def soft_update_online_to_target(online_vars_by_name, target_vars_by_name, tau=0.01):
    update_ops = []
    ### theta_prime = tau * theta + (1-tau) * theta_prime
    for (online_var_name, online_var) in online_vars_by_name.items():
        target_var = target_vars_by_name[online_var_name]
        theta_prime = tau * online_var + (1 - tau) * target_var
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


# def make_fc(name, input, ks, keep_prob,
#             activation_fn=None, activation_fn_name=None,
#             batch_norm=False, is_training=True,
#             mean=0., stddev=0.1):
#     with tf.name_scope(name):
#         with tf.variable_scope(name):
#             # w = tf.get_variable('%s-w' % name, shape=ks, initializer=tf.truncated_normal_initializer(mean=mean, stddev=stddev))
#             w = tf.get_variable('%s-w' % name, shape=ks, initializer=tc.layers.xavier_initializer())
#             out = tf.matmul(input, w, name='%s-mat' % name)
#
#             if batch_norm:
#                 out = tc.layers.layer_norm(out, center=True, scale=True)
#                 # out = tf.layers.batch_normalization(out, name='%s-batch_norm' % name, training=is_training)
#             else:
#                 b = tf.get_variable('%s-b' % name, shape=[ks[-1]],
#                                     initializer=tf.constant_initializer(value=0.))
#                 out = tf.nn.bias_add(out, b, name='%s-bias_add' % name)
#             if activation_fn is not None:
#                 out = activation_fn(out, name='%s-%s' % (name, activation_fn_name))
#             # out = tf.nn.dropout(out, keep_prob, name='%s-drop' % name)
#     return out

def make_fc(name, input, units_num, activation_fn=None, batch_norm=False):
    with tf.name_scope(name):
        with tf.variable_scope(name):
            w = tf.get_variable('%s-w' % name, shape=[input.shape[-1], units_num],
                                initializer=tc.layers.xavier_initializer())
            out = tf.matmul(input, w, name='%s-mat' % name)
            if batch_norm:
                out = tc.layers.layer_norm(out, center=True, scale=True)
            else:
                b = tf.get_variable('%s-b' % name, shape=[units_num],
                                    initializer=tf.constant_initializer(value=0.))
                out = tf.nn.bias_add(out, b, name='%s-bias_add' % name)
            if activation_fn is not None:
                out = activation_fn(out)
    return out

def make_fc_with_s_a(name, input, units_num, activation_fn=None, batch_norm=False):
    with tf.name_scope(name):
        with tf.variable_scope(name):
            # w = tf.get_variable('%s-w' % name, shape=ks, initializer=tf.truncated_normal_initializer(mean=mean, stddev=stddev))
            w = tf.get_variable('%s-w' % name, shape=[input.shape[-1], units_num],
                                initializer=tc.layers.xavier_initializer())
            out = tf.matmul(input, w, name='%s-mat' % name)

            if batch_norm:
                out = tc.layers.layer_norm(out, center=True, scale=True)
            else:
                b = tf.get_variable('%s-b' % name, shape=[units_num],
                                    initializer=tf.constant_initializer(value=0.))
                out = tf.nn.bias_add(out, b, name='%s-bias_add' % name)
            if activation_fn is not None:
                out = activation_fn(out)
            # out = tf.nn.dropout(out, keep_prob, name='%s-drop' % name)
    return out


def make_dense(name, input, hidden_unit_size, batch_norm, activation_fn):
    with tf.name_scope(name):
        with tf.variable_scope(name):
            out = tf.layers.dense(input, hidden_unit_size, kernel_initializer=tc.layers.xavier_initializer())
            if batch_norm:
                out = tc.layers.layer_norm(out, center=True, scale=True)
            if activation_fn:
                out = activation_fn(out)
            return out


def make_dense_with_s_a(name, s, a, hidden_unit_size, batch_norm, activation_fn):
    with tf.name_scope(name):
        with tf.variable_scope(name):
            s = tf.layers.dense(s, hidden_unit_size, kernel_initializer=tc.layers.xavier_initializer(),
                                name='{}-s'.format(name))
            a = tf.layers.dense(a, hidden_unit_size, kernel_initializer=tc.layers.xavier_initializer(),
                                name='{}-a'.format(name))
            out = s + a
            if batch_norm:
                out = tc.layers.layer_norm(out, center=True, scale=True)
            if activation_fn:
                out = activation_fn(out)
            return out


def make_lstm_layer(name, s, lstm_size, layer_num, sequence_len, keep_prob):
    # RNN cell
    def make_cell(rnn_size, keep_prob):
        # Use a basic LSTM cell
        # lstm = tc.rnn.BasicLSTMCell(COMA_CFG.lstm_size, forget_bias=1.0, state_is_tuple=True)
        enc_cell = tc.rnn.LSTMCell(rnn_size,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        # Add dropout to the cell
        drop = tc.rnn.DropoutWrapper(enc_cell, output_keep_prob=keep_prob)
        return drop

    ### Build the LSTM Cell
    def build_cell(lstm_size, keep_prob):
        # Use a basic LSTM cell
        lstm = tc.rnn.BasicLSTMCell(lstm_size)

        # Add dropout to the cell
        drop = tc.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop

    with tf.name_scope(name):
        with tf.variable_scope(name):
            # stack up multiple LSTM layers
            cell = tc.rnn.MultiRNNCell([build_cell(lstm_size, keep_prob) for _ in range(layer_num)])
            # initial_state = cell.zero_state(batch_size, tf.float32)
            outputs, final_state = tf.nn.dynamic_rnn(cell=cell,
                                                     inputs=s,
                                                     # sequence_length=sequence_len,
                                                     dtype=tf.float32)
            # (LSTMStateTuple(c=<tf.Tensor 'Actor/online_actor/rnn/while/Exit_3:0' shape=(?, 32) dtype=float32>, h=<tf.Tensor 'Actor/online_actor/rnn/while/Exit_4:0' shape=(?, 32) dtype=float32>),)
            # print(final_state)
            # Tensor("Actor/online_actor/rnn/transpose_1:0", shape=(?, 9, 32), dtype=float32)
            # print(outputs)
            # 直接调用final_state 中的 h_state (final_state[1]) or outputs[:, -1, :] 来进行运算:
            state_feature = final_state[0][1]
            return state_feature