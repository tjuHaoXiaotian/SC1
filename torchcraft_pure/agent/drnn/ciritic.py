# -*- coding: UTF-8 -*-
import tensorflow as tf
from torchcraft_pure.agent.drnn.common import soft_update_online_to_target

FLAGS = tf.app.flags.FLAGS  # alias


class Critic(object):

    def __init__(self, sess, action_dim, state_dim, agent_num, use_batch_norm):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_num = agent_num
        self.use_batch_norm = use_batch_norm
        self.is_training = tf.placeholder(tf.bool, shape=[], name="is_training")

        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        self.batch_size = tf.placeholder(tf.int32, shape=[], name="batch_size")
        self.sequence_length_friends = tf.placeholder(tf.int32, shape=[None], name="sequence_length_friends")
        self.sequence_length_enemies = tf.placeholder(tf.int32, shape=[None], name="sequence_length_enemies")

        self.state_input_friends = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim * FLAGS.agent_num],
                                                  name='state_inputs_friends')
        self.state_input_enemies = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim * FLAGS.enemy_num],
                                                  name='state_inputs_enemies')

        # TODO: action 输入的顺序怎么定？ 按距离排序了
        # 其他单位在 s 下选择的动作
        self.other_units_action_input = tf.placeholder(dtype=tf.float32,
                                                       shape=[None, self.action_dim * (self.agent_num - 1)],
                                                       name='other_units_action_input')
        # 自己当时执行的动作
        self.self_action_input = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim],
                                                name='self_action_input')
        # actor 输出的执行各个动作概率
        self.actor_output_probability = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim],
                                                       name='actor_output_probability')

        self.Q_value_label_input = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='Q_value_label_input')
        self.reward = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='reward')
        self.terminal = tf.placeholder(dtype=tf.bool, shape=[None, 1], name='terminal')

        with tf.variable_scope("Critic"):
            # online actor
            self.online_q_outputs = self._build_net(self.state_input_friends, self.state_input_enemies,
                                                    self.other_units_action_input, "online_q", trainable=True)
            self.online_q_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                       scope='Critic/online_q')
            self.online_policy_net_vars_by_name = {var.name.strip('Critic/online'): var
                                                   for var in self.online_q_net_vars}
            # target actor
            self.target_q_outputs = self._build_net(self.state_input_friends, self.state_input_enemies,
                                                    self.other_units_action_input, "target_q", trainable=False)
            self.target_q_net_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                       scope='Critic/target_q')
            self.target_policy_net_vars_by_name = {var.name.strip('Critic/target'): var
                                                   for var in self.target_q_net_vars}

        # define hard replacement and soft replacement
        self._build_update_graph()
        # define the target label of current Q value
        self._build_td_target_graph()
        # define the cost function
        self._build_cost_graph()
        # define the advantage function
        self._build_advantage()

    def _build_net(self, state_inputs_friends, state_inputs_enemies, other_units_action_input, scope, trainable):
        '''
        :param critic_state_inputs:
        :param other_units_action_input:
        :param scope:
        :param trainable:
        :return:
        '''
        with tf.variable_scope(scope):
            # ==================================== Basic LSTM ==================================
            # 这里暂时把所有单位由远及近 当成时序处理
            state_inputs_friends = tf.reshape(state_inputs_friends,
                                              [-1, FLAGS.agent_num, FLAGS.state_dim])
            state_inputs_enemies = tf.reshape(state_inputs_enemies,
                                              [-1, FLAGS.enemy_num, FLAGS.state_dim])
            with tf.variable_scope("friends"):
                state_feature_friends = self._basic_lstm_layer(state_inputs_friends, self.batch_size,
                                                           self.sequence_length_friends, self.keep_prob)
            with tf.variable_scope("enemies"):
                state_feature_enemies = self._basic_lstm_layer(state_inputs_enemies, self.batch_size,
                                                           self.sequence_length_enemies, self.keep_prob)
            state_action = tf.concat([state_feature_enemies, state_feature_friends, other_units_action_input], axis=1)
            layer2 = self._fully_connected(state_action,
                                           [FLAGS.lstm_size * 2 + self.action_dim * (self.agent_num - 1), 128], [128],
                                           activation_fn=tf.nn.relu,
                                           variable_scope_name="layer2", trainable=trainable)

            out = self._fully_connected(layer2, [128, self.action_dim],
                                        [self.action_dim],
                                        activation_fn=None, variable_scope_name="output",
                                        trainable=trainable)

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
                                  self.reward + FLAGS.gamma * tf.reduce_max(self.target_q_outputs, keep_dims=True,
                                                                            axis=1))

    def _build_cost_graph(self):
        # 批量计算执行 ai 的 Q(S, a-i, ai)
        online_output_q = tf.reduce_sum(tf.multiply(self.online_q_outputs, self.self_action_input), keep_dims=True,
                                        axis=1)

        with tf.name_scope("critic/loss"):
            self.cost = tf.reduce_mean(tf.square(self.Q_value_label_input - online_output_q))
            tf.summary.scalar('critic_loss', self.cost)  # tensorflow >= 0.12

        if self.use_batch_norm:
            # If we don't include the update ops as dependencies on the train step, the
            # tf.layers.batch_normalization layers won't update their population statistics,
            # which will cause the model to fail at inference time
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Critic')):
                self.train = tf.train.AdamOptimizer(FLAGS.critic_learning_rate).minimize(self.cost)
        else:
            self.train = tf.train.AdamOptimizer(FLAGS.critic_learning_rate).minimize(self.cost)

    def _build_advantage(self):
        self.advantage = tf.reduce_sum(self.online_q_outputs * self.self_action_input, keep_dims=True,
                                       axis=1) - tf.reduce_sum(self.online_q_outputs * self.actor_output_probability,
                                                               keep_dims=True, axis=1)

    def operation_get_TDtarget(self, batch_size, state_next_friends, state_next_enemies, sequence_length_friends,
                               sequence_length_enemies, action_next_others, reward, terminal, is_training):
        '''
        Training to get td target
        :param action_next:  target actor output
        :param state_next:
        :param reward:
        :param terminal:
        :return:
        '''
        return self.sess.run(self.td_target, feed_dict={
            self.batch_size: batch_size,
            self.state_input_friends: state_next_friends,
            self.state_input_enemies: state_next_enemies,
            self.sequence_length_friends: sequence_length_friends,
            self.sequence_length_enemies: sequence_length_enemies,
            self.other_units_action_input: action_next_others,  # 应该是 target actor output
            self.reward: reward,
            self.terminal: terminal,
            self.is_training: is_training,
            self.keep_prob: 1.,
        })

    def operation_cal_advantage(self, batch_size, state_friends, state_enemies, sequence_length_friends,
                                sequence_length_enemies, action_others, self_action_input, actor_output_probability,
                                is_training):
        batch_advantages = self.sess.run(self.advantage,
                                         feed_dict={
                                             self.batch_size: batch_size,
                                             self.state_input_friends: state_friends,
                                             self.state_input_enemies: state_enemies,
                                             self.sequence_length_friends: sequence_length_friends,
                                             self.sequence_length_enemies: sequence_length_enemies,
                                             self.other_units_action_input: action_others,
                                             self.self_action_input: self_action_input,
                                             self.actor_output_probability: actor_output_probability,
                                             self.is_training: is_training,
                                             self.keep_prob: 1.,
                                         })
        return batch_advantages

    def operation_critic_learn(self, batch_size, state_friends, state_enemies, sequence_length_friends,
                               sequence_length_enemies, other_unit_actions, self_action, TDtarget, is_training):
        '''
        Training the critic network
        :param TDtarget: the target label (calculated by self.operation_get_TDtarget())
        :param action: the batch action input which is sampled from the replay_buffer
        :param state: the batch sate input which is sampled from the replay_buffer
        :return:
        '''
        _ = self.sess.run(self.train,
                          feed_dict={
                              self.batch_size: batch_size,
                              self.state_input_friends: state_friends,
                              self.state_input_enemies: state_enemies,
                              self.sequence_length_friends: sequence_length_friends,
                              self.sequence_length_enemies: sequence_length_enemies,
                              self.other_units_action_input: other_unit_actions,
                              self.self_action_input: self_action,
                              self.Q_value_label_input: TDtarget,
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

    def _basic_lstm_layer(self, state_inputs, batch_size, sequence_len, keep_prob):
        # RNN cell
        def make_cell(rnn_size, keep_prob):
            # Use a basic LSTM cell
            # lstm = tf.contrib.rnn.BasicLSTMCell(FLAGS.lstm_size, forget_bias=1.0, state_is_tuple=True)
            enc_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                               initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            # Add dropout to the cell
            drop = tf.contrib.rnn.DropoutWrapper(enc_cell, output_keep_prob=keep_prob)
            return drop

        ### Build the LSTM Cell
        def build_cell(lstm_size, keep_prob):
            # Use a basic LSTM cell
            lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

            # Add dropout to the cell
            drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            return drop
        # Stack up multiple LSTM layers
        cell = tf.contrib.rnn.MultiRNNCell([build_cell(FLAGS.lstm_size, keep_prob) for _ in range(FLAGS.lstm_layer)])
        # initial_state = cell.zero_state(batch_size, tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(cell=cell,
                                                 inputs=state_inputs,
                                                 sequence_length=sequence_len,
                                                 dtype=tf.float32)
        # 直接调用final_state 中的 h_state (final_state[1]) or outputs[:, -1, :] 来进行运算:
        state_feature = final_state[0][1]
        return state_feature

    def _fully_connected(self, layer_in, weights_shape, biases_shape, activation_fn=None, variable_scope_name="layer",
                         trainable=True):
        if self.use_batch_norm and activation_fn:
            with tf.variable_scope(variable_scope_name):
                weights = self._weight_variable(weights_shape, trainable)
                # Batch normalization uses weights as usual, but does NOT add a bias term. This is because
                # its calculations include gamma and beta variables that make the bias term unnecessary.
                linear_output = tf.matmul(layer_in, weights)
                # Apply batch normalization to the linear combination of the inputs and weights
                batch_normalized_output = tf.layers.batch_normalization(linear_output, training=self.is_training)
                return activation_fn(batch_normalized_output)
        else:
            with tf.variable_scope(variable_scope_name):
                weights = self._weight_variable(weights_shape, trainable)
                biases = self._bias_variable(biases_shape, trainable)
                linear_output = tf.add(tf.matmul(layer_in, weights), biases)
            return linear_output if not activation_fn else activation_fn(linear_output)

    def _weight_variable(self, shape, trainable, name='weights'):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1), trainable=trainable, name=name)

    def _bias_variable(self, shape, trainable, name="bias"):
        return tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=shape), trainable=trainable, name=name)

    def _conv2d(self, x, w, stride=(1, 1, 1, 1)):
        return tf.nn.conv2d(x, w, strides=stride, padding='SAME')

    def _max_pooling(self, x, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1)):
        return tf.nn.max_pool(x, ksize, strides, padding='SAME')
