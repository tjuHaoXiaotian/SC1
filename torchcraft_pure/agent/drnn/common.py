#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tensorflow as tf
DDPG_CFG = tf.app.flags.FLAGS  # alias

def soft_update_online_to_target(online_vars_by_name, target_vars_by_name):
  update_ops = []
  ### theta_prime = tau * theta + (1-tau) * theta_prime
  for (online_var_name, online_var) in online_vars_by_name.items():
    target_var = target_vars_by_name[online_var_name]
    theta_prime = DDPG_CFG.tau * online_var + (1 - DDPG_CFG.tau) * target_var
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