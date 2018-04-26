#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np
from torchcraft_pure.agent.general.nn_common import *
from torchcraft_pure.agent.general.replay_buffer import ReplayBuffer, construct_transition
from torchcraft_pure.agent.general.parameters import parameters

parameters.map = 'm5v5_c_far'

# hyper parameters
parameters.input_dim = 11 * 10
parameters.single_input_dim = 11
parameters.action_dim = 10
parameters.self_agent_num = 5

# training parameters
parameters.replay_buffer_file_name = 'replay_buffer'
parameters.random_seed = 1234567
# set np seed
np.random.seed(parameters.random_seed)
parameters.replay_buff_size = 160000
parameters.replay_buff_save_segment_size = 10000
parameters.replay_buffer_file_path = "{}/replay_buffers".format(parameters.map)

parameters.tau = 0.01
parameters.save_model_num = 10
parameters.batch_size = 32
parameters.entropy_regularizer_lambda = 0.05
parameters.actor_learning_rate = 0.0005
parameters.critic_learning_rate = 0.0005
parameters.keep_prob = 1.
parameters.gamma = 0.95



class Agent(object):

    def __init__(self, training=True):
        self.is_training = training

        self.sess = tf.Session()
        self.actor = Actor(self.sess, parameters.input_dim, parameters.action_dim)
        self.critic = Critic(self.sess, parameters.input_dim, parameters.action_dim, parameters.self_agent_num)

        if self.is_training:
            # initialize replay buffer
            self.replay_buffer = ReplayBuffer(buffer_size=parameters.replay_buff_size,
                                              save_segment_size=parameters.replay_buff_save_segment_size,
                                              save_path=parameters.replay_buffer_file_path,
                                              seed=parameters.random_seed)

            # init model Saver()
            self.saver = tf.train.Saver(max_to_keep=parameters.save_model_num)

            # init nn weights and bias
            self.sess.run(tf.global_variables_initializer())

            # copy all variables from online nn to target nn
            self.actor.operation_update_TDnet_compeletely()
            self.critic.operation_update_TDnet_compeletely()

            # initialize visualization
            self.set_up_visualization()

    def set_up_visualization(self):
        '''
        set up visualization things of tensorboard.
        :return:
        '''
        with tf.name_scope("return"):
            self.return_of_each_episode_tensor = tf.placeholder(dtype=tf.float32, shape=[],
                                                                name="return_of_each_episode")
            tf.summary.scalar('return_of_each_episode', self.return_of_each_episode_tensor)  # tensorflow >= 0.12

        # add summary
        self.merged = tf.summary.merge_all()  # tensorflow >= 0.12
        # writer
        self.writer = tf.summary.FileWriter("{}/logs/".format(parameters.map),
                                            self.sess.graph)  # tensorflow >=0.12

    def select_action(self, observation):
        stochastic_action = self.actor.operation_choose_action(observation, is_training=False)
        print('stochastic_action: ' ,stochastic_action)
        return stochastic_action

    def predict_action(self, observation):
        greedy_action = self.actor.operation_greedy_action(observation, is_training=False)
        # print('greedy_action: ', greedy_action)
        return greedy_action

    def store_transition(self, pre_alive_friends_info, reward, cur_alive_friends_info, game_ended=False):
        '''
        :param pre_alive_friends_info:
        :param reward:
        :param cur_alive_friends_info:
        :return:
        '''
        for unit_id, unit_pre_transition in pre_alive_friends_info.items():
            s = unit_pre_transition['state']
            # set other actions: if unit is dead, the action is [0, 0, 0, ..., 0]
            a_other_units = np.zeros([parameters.self_agent_num, parameters.action_dim], dtype=np.float32)
            for other_id, other_unit_pre_tran in pre_alive_friends_info.items():
                if other_id != unit_id:  # except self
                    a_other_units[other_id] = other_unit_pre_tran['action']
            a_other_units = np.hstack(a_other_units)
            a_me = unit_pre_transition['action']
            r = reward
            terminated = cur_alive_friends_info is None or cur_alive_friends_info.get(unit_id,
                                                                                      None) is None or game_ended
            if not terminated:
                s_ = cur_alive_friends_info[unit_id]['state']
                if len(cur_alive_friends_info) > 1:  # there still exists other alive units.
                    other_s_ = {id: other_unit_tran['state'] for id, other_unit_tran in cur_alive_friends_info.items()
                                if id != unit_id}
                else:
                    other_s_ = None  # all others dead
            else:
                s_ = unit_pre_transition['state']
                other_s_ = None

            transition = construct_transition(s, a_other_units, a_me, r, s_, other_s_, terminated)
            self.replay_buffer.store(transition)

    def one_hot_action(self, action_id, action_dim):
        action = np.zeros([action_dim, ], dtype=np.int32)
        action[action_id] = 1
        return action

    def batch_training(self):
        assert self.replay_buffer is not None
        if self.replay_buffer.length > parameters.batch_size:

            (batch_s, batch_a_others, batch_a_me, batch_r, batch_s_, batch_s_others, batch_s_others_ids,
             batch_terminated) = self.replay_buffer.sample_batch(parameters.batch_size)
            # updating critic
            # 1: prepare the batch others_a_
            batch_a_others_s_ = []
            for ids, nxt_state_others in zip(batch_s_others_ids, batch_s_others):
                if nxt_state_others is None:  # terminated or all other dead
                    action_others_s_ = None
                else:
                    action_others_s_ = self.predict_action(nxt_state_others)  # corresponding to units ids, batch = len(ids)
                    # check predict_action out is []
                a_other_units_s_ = np.zeros([parameters.self_agent_num, parameters.action_dim], dtype=np.float32)
                if action_others_s_ is not None:
                    for id, a_o_s_ in zip(ids, action_others_s_):
                        a_other_units_s_[id] = self.one_hot_action(a_o_s_, parameters.action_dim)
                a_other_units_s_ = np.hstack(a_other_units_s_)
                batch_a_others_s_.append(a_other_units_s_)

            # 2: cal td target
            batch_td_target = self.critic.operation_get_td_target(
                batch_s_,
                batch_a_others_s_,
                batch_r,
                batch_terminated,
                is_training=True
            )

            # 3: training critic
            self.critic.operation_critic_learn(
                batch_s,
                batch_a_others,
                batch_a_me,
                batch_td_target,
                is_training=True
            )

            # updating actor
            # 4: calculate advantage
            actor_output_probability = self.actor.operation_cal_softmax_prob(
                batch_s,
                is_training=True
            )
            batch_advantages = self.critic.operation_cal_advantage(
                batch_s,
                batch_a_others,
                batch_a_me,
                actor_output_probability,
                is_training=True
            )
            # training actor
            cost = self.actor.operation_actor_learn(
                batch_s,
                batch_a_me,
                batch_advantages,
                is_training=True
            )

            # soft update the parameters of the two model
            self.actor.operation_soft_update_TDnet()
            self.critic.operation_soft_update_TDnet()


class Actor(object):
    def __init__(self, sess, input_dim, output_dim):
        self.sess = sess
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.is_training = tf.placeholder(tf.bool, shape=[], name="is_training")
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')

        self.state_inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim],
                                           name="state_inputs")
        # 实际执行的动作，也就是对应actor要更新的输出 Notice: 这里已经 one-hot了
        self.execute_action = tf.placeholder(dtype=tf.float32, shape=[None, self.output_dim], name="execute_action")
        # 执行上述动作 execute_action 的 advantage
        self.advantage = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="advantage")

        with tf.variable_scope("actor"):
            # online actor
            self.softmax_action_outputs = self._build_net("online_actor")
            self.online_policy_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                            scope='actor/online_actor')

            self.online_policy_net_vars_by_name = {var.name.strip('actor/online'): var
                                                   for var in self.online_policy_net_vars}

            # target actor : 输入的是 S' 输出 a'
            self.target_softmax_action_outputs = self._build_net("target_actor")
            self.target_policy_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                            scope='actor/target_actor')

            self.target_policy_net_vars_by_name = {var.name.strip('actor/target'): var
                                                   for var in self.target_policy_net_vars}

        self._build_update_graph()
        self._build_cost_graph()

    def _build_net(self, scope_name):
        with tf.name_scope(scope_name):
            with tf.variable_scope(scope_name):
                # fully connected
                out = make_fc('fc1', self.state_inputs, [self.input_dim, 512], self.keep_prob,
                              activation_fn=tf.nn.tanh, activation_fn_name="tanh", batch_norm=False,
                              is_training=self.is_training, mean=0., stddev=0.1)

                out = make_fc('fc2', out, [512, 128], self.keep_prob,
                              activation_fn=tf.nn.tanh, activation_fn_name="tanh", batch_norm=False,
                              is_training=self.is_training, mean=0., stddev=0.1)

                out = make_fc('out', out, [128, self.output_dim], self.keep_prob,
                              activation_fn=tf.nn.softmax, activation_fn_name="softmax", batch_norm=False,
                              is_training=self.is_training, mean=0., stddev=0.1)
                return out

    def _build_update_graph(self):
        # target net hard replacement
        self.hard_replace = copy_online_to_target(self.online_policy_net_vars_by_name,
                                                         self.target_policy_net_vars_by_name)

        # target net soft replacement
        self.soft_replace = soft_update_online_to_target(self.online_policy_net_vars_by_name,
                                                         self.target_policy_net_vars_by_name)

    def _build_cost_graph(self):
        # batch 维度上求平均
        self.cost = tf.reduce_mean(
            # action 维度上求和（只留下对应执行的动作维度）
            tf.log(tf.reduce_sum(self.softmax_action_outputs * self.execute_action, keep_dims=True,
                                 axis=1)) * self.advantage
        )

        eps = 1e-10
        y_clip = tf.clip_by_value(self.softmax_action_outputs, eps, 1.0 - eps)
        self.entropy_loss = -tf.reduce_mean(tf.reduce_sum(y_clip * tf.log(y_clip), axis=1))
        # self.entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=y_clip,dim=-1))
        self.entropy_rectifier = parameters.entropy_regularizer_lambda * self.entropy_loss

        with tf.name_scope("actor/loss"):
            self.total_cost = -(self.cost + self.entropy_rectifier)
            tf.summary.scalar('actor_total_loss', self.total_cost)  # tensorflow >= 0.12
            tf.summary.scalar('actor_loss', -self.cost)  # tensorflow >= 0.12

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='actor')):
            self.train = tf.train.AdamOptimizer(parameters.actor_learning_rate).minimize(self.total_cost)

    def operation_cal_softmax_prob(self, state_inputs, is_training):
        '''
        :param state_inputs:
        :param is_training:
        :return:
        '''
        prob_weights = self.sess.run(self.softmax_action_outputs, feed_dict={
            self.state_inputs: state_inputs,
            self.is_training: is_training,
            self.keep_prob: 1.,
        })
        return prob_weights

    def operation_choose_action(self, state_inputs, is_training):
        '''
        :param state_inputs:
        :param is_training:
        :return:
        '''
        prob_weights = self.sess.run(self.softmax_action_outputs, feed_dict={
            self.state_inputs: state_inputs,
            self.is_training: is_training,
            self.keep_prob: 1.,
        })
        # if self.has_nan(prob_weights):
        # print(prob_weights)

        # 按照给定的概率采样
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        print('random prob: ', prob_weights.ravel())
        return action

    def operation_greedy_action(self, state_inputs, is_training):
        '''
        :param state_inputs:
        :param is_training:
        :return:
        '''
        prob_weights = self.sess.run(self.target_softmax_action_outputs, feed_dict={
            self.state_inputs: state_inputs,
            self.is_training: is_training,
            self.keep_prob: 1.,
        })
        action = np.argmax(prob_weights, axis=1)
        return action

    def operation_actor_learn(self, state_inputs, execute_action, advantage, is_training):
        '''
        Traning the actor network
        :param state_inputs: state batch (sampled from the replay buffer)
        :param execute_action: action batch (sampled from the replay buffer, executed at that timestep)
        :param advantage: calculated advantage
        :param is_training:
        :return:
        '''
        _, cost = self.sess.run([self.train, self.total_cost], feed_dict={
            self.state_inputs: state_inputs,
            self.execute_action: execute_action,
            self.advantage: advantage,
            self.keep_prob: parameters.keep_prob,
            self.is_training: is_training
        })
        # print("cost: ", cost)
        return cost

    def operation_update_TDnet_compeletely(self):
        '''
        hard replacement
        :return:
        '''
        self.sess.run(self.hard_replace)

    def operation_soft_update_TDnet(self):
        '''
        soft replacement
        :return:
        '''
        self.sess.run(self.soft_replace)


class Critic(object):
    def __init__(self, sess, input_dim, action_dim, self_agent_num):
        self.sess = sess
        self.input_dim = input_dim
        self.output_dim = action_dim

        self.is_training = tf.placeholder(tf.bool, shape=[], name="is_training")
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')

        self.state_inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim],
                                           name="state_inputs")

        # 其他单位在 s 下选择的动作 (自己动作输入为 0)
        self.other_units_action_input = tf.placeholder(dtype=tf.float32,
                                                       shape=[None, action_dim * self_agent_num],
                                                       name='other_units_action_input')
        # 自己当时执行的动作
        self.self_action_input = tf.placeholder(dtype=tf.float32, shape=[None, action_dim],
                                                name='self_action_input')

        # actor 输出的执行各个动作概率
        self.actor_output_probability = tf.placeholder(dtype=tf.float32, shape=[None, action_dim],
                                                       name='actor_output_probability')

        self.q_value_label_input = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='q_value_label_input')
        self.reward = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='reward')
        self.terminal = tf.placeholder(dtype=tf.bool, shape=[None, 1], name='terminal')

        with tf.variable_scope("critic"):
            # online actor
            self.online_q_outputs = self._build_net("online_q")
            self.online_q_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                       scope='critic/online_q')
            self.online_policy_net_vars_by_name = {var.name.strip('critic/online'): var
                                                   for var in self.online_q_net_vars}
            # target actor
            self.target_q_outputs = self._build_net("target_q")
            self.target_q_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                       scope='critic/target_q')
            self.target_policy_net_vars_by_name = {var.name.strip('critic/target'): var
                                                   for var in self.target_q_net_vars}

        # define hard replacement and soft replacement
        self._build_update_graph()
        # define the target label of current Q value
        self._build_td_target_graph()
        # define the cost function
        self._build_cost_graph()
        # define the advantage function
        self._build_advantage()

    def _build_net(self, scope_name):
        with tf.name_scope(scope_name):
            with tf.variable_scope(scope_name):
                state_action = tf.concat([self.state_inputs, self.other_units_action_input], axis=1)
                # fully connected
                out = make_fc('fc1', state_action,
                              [self.input_dim + int(self.other_units_action_input.get_shape()[1]), 512], self.keep_prob,
                              activation_fn=tf.nn.tanh, activation_fn_name="tanh", batch_norm=False,
                              is_training=self.is_training, mean=0., stddev=0.1)

                out = make_fc('fc2', out, [512, 128], self.keep_prob,
                              activation_fn=tf.nn.tanh, activation_fn_name="tanh", batch_norm=False,
                              is_training=self.is_training, mean=0., stddev=0.1)

                out = make_fc('out', out, [128, self.output_dim], self.keep_prob,
                              activation_fn=None, activation_fn_name=None, batch_norm=False,
                              is_training=self.is_training, mean=0., stddev=0.1)
                with tf.name_scope("critic/q"):
                    tf.summary.histogram('critic/q', out)  # Tensorflow >= 0.12

                with tf.name_scope("critic/average_q"):
                    tf.summary.scalar('critic/average_q',
                                      tf.reduce_mean(tf.reduce_sum(out, axis=1, keep_dims=False)))  # tensorflow >= 0.12
                return out

    def _build_update_graph(self):
        # target net hard replacement
        self.hard_replace = copy_online_to_target(self.online_policy_net_vars_by_name,
                                                         self.target_policy_net_vars_by_name)

        # target net soft replacement
        self.soft_replace = soft_update_online_to_target(self.online_policy_net_vars_by_name,
                                                         self.target_policy_net_vars_by_name)

    def _build_td_target_graph(self):
        self.td_target = tf.where(self.terminal,
                                  self.reward,
                                  self.reward + parameters.gamma * tf.reduce_max(self.target_q_outputs, keep_dims=True,
                                                                            axis=1))

    def _build_cost_graph(self):
        # 批量计算执行 ai 的 Q(S, a-i, ai)
        online_output_q = tf.reduce_sum(tf.multiply(self.online_q_outputs, self.self_action_input), keep_dims=True,
                                        axis=1)

        with tf.name_scope("critic/loss"):
            self.cost = tf.reduce_mean(tf.square(self.q_value_label_input - online_output_q))
            tf.summary.scalar('critic_loss', self.cost)  # tensorflow >= 0.12

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Critic')):
            self.train = tf.train.AdamOptimizer(parameters.critic_learning_rate).minimize(self.cost)

    def _build_advantage(self):
        self.advantage = tf.reduce_sum(self.online_q_outputs * self.self_action_input, keep_dims=True,
                                       axis=1) - tf.reduce_sum(self.online_q_outputs * self.actor_output_probability,
                                                               keep_dims=True, axis=1)

    def operation_get_td_target(self, state_inputs_next, action_next_others, reward, terminal, is_training):
        '''
        Training to get td target
        :param action_next_others:  target actor output
        :param state_inputs_next:
        :param reward:
        :param terminal:
        :return:
        '''
        return self.sess.run(self.td_target, feed_dict={
            self.state_inputs: state_inputs_next,
            self.other_units_action_input: action_next_others,  # 应该是 target actor output
            self.reward: reward,
            self.terminal: terminal,
            self.is_training: is_training,
            self.keep_prob: 1.,
        })

    def operation_cal_advantage(self, state_inputs, action_others, self_action_input,
                                actor_output_probability,
                                is_training):
        batch_advantages = self.sess.run(self.advantage,
                                         feed_dict={
                                             self.state_inputs: state_inputs,
                                             self.other_units_action_input: action_others,
                                             self.self_action_input: self_action_input,
                                             self.actor_output_probability: actor_output_probability,
                                             self.is_training: is_training,
                                             self.keep_prob: 1.,
                                         })
        return batch_advantages

    def operation_critic_learn(self, state_inputs, other_unit_actions, self_action, td_target, is_training):
        _ = self.sess.run(self.train,
                          feed_dict={
                              self.state_inputs: state_inputs,
                              self.other_units_action_input: other_unit_actions,
                              self.self_action_input: self_action,
                              self.q_value_label_input: td_target,
                              self.is_training: is_training,
                              self.keep_prob: 1.
                          })

    def operation_update_TDnet_compeletely(self):
        '''
        hard replacement
        :return:
        '''
        self.sess.run(self.hard_replace)

    def operation_soft_update_TDnet(self):
        '''
        soft replacement
        :return:
        '''
        self.sess.run(self.soft_replace)
