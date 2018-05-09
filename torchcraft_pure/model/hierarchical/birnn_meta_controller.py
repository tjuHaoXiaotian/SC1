from torchcraft_pure.model.hierarchical.meta_replay_buffer import ReplayBuffer
from torchcraft_pure.model.util.nn_util import *


class META_MAAC(object):
    def __init__(self, name, state_dim, action_dim, training_agent_num, hidden_layer_sizes, actor_lr=1e-4,
                 q_func_lr=1e-3, v_func_lr=1e-3, batch_norm=False,
                 replay_buffer_size=100000):
        # hyper parameters
        self.gamma = 0.95
        self.actor_lr = actor_lr
        self.q_func_lr = q_func_lr
        self.v_func_lr = v_func_lr
        self.tau = 0.01

        self.hidden_layer_sizes = hidden_layer_sizes

        # build placeholders
        self.global_state_input = tf.placeholder(shape=[None, state_dim], dtype=tf.float32,
                                                 name='{}/actor_state_input'.format(name))
        self.global_state_nxt_input = tf.placeholder(shape=[None, state_dim], dtype=tf.float32,
                                                     name='{}/actor_state_nxt_input'.format(name))

        self.q_action_input = [tf.placeholder(shape=[None, action_dim], dtype=tf.float32,
                                              name='{}/q_action_input/{}'.format(name, _)) for _ in
                               range(training_agent_num)]

        # to decide whether the unit is still alive
        self.still_alive_input = [tf.placeholder(shape=[None], dtype=tf.bool,
                                                 name='{}/still_alive_input/{}'.format(name, _)) for _ in
                                  range(training_agent_num)]

        self.reward = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='{}/reward'.format(name))
        self.keep_prob = tf.placeholder(shape=[], dtype=tf.float32, name='{}/keep_prob'.format(name))
        # self.is_training = tf.placeholder(shape=[], dtype=tf.bool, name='{}/is_training'.format(name))
        self.terminal = tf.placeholder(shape=[None, 1], dtype=tf.bool, name='{}/terminal'.format(name))

        # def global actor network architecture
        def actor_network(scope_name, s, reuse=False):
            with tf.name_scope(scope_name):
                with tf.variable_scope(scope_name, reuse=reuse):
                    x = s
                    x = make_fc('fc1', x, self.hidden_layer_sizes[0],
                                activation_fn=tf.nn.relu,
                                batch_norm=True
                                )

                    for idx, size in enumerate(self.hidden_layer_sizes[1:]):
                        x = make_fc('fc{}'.format(idx + 2), x, size,
                                    activation_fn=tf.nn.relu,
                                    batch_norm=batch_norm,
                                    )
                    softmax_n, logits_n = [], []
                    for agent_id in range(training_agent_num):
                        x = make_fc('fc{}-agent{}'.format(len(self.hidden_layer_sizes) + 1, agent_id), x,
                                    action_dim,
                                    activation_fn=None)
                        logits = x
                        x = tf.nn.softmax(logits=logits)
                        softmax_n.append(x)
                        logits_n.append(logits)
                    return softmax_n, logits_n

        def q_network(scope_name, s, a, reuse=False):
            with tf.name_scope(scope_name):
                with tf.variable_scope(scope_name, reuse=reuse):
                    x = make_fc('fc1-s', s, self.hidden_layer_sizes[0])
                    a = make_fc('fc1-a', a, self.hidden_layer_sizes[0])
                    x += a
                    x = tf.nn.relu(x)
                    # x = tc.layers.layer_norm(x, center=True, scale=True)
                    for idx, size in enumerate(self.hidden_layer_sizes[1:]):
                        x = make_fc('fc{}'.format(idx + 2), x, size,
                                    activation_fn=tf.nn.relu,
                                    batch_norm=batch_norm,
                                    )
                    x = make_fc('fc{}'.format(len(self.hidden_layer_sizes) + 1), x, 1)
                    return x

        def v_network(scope_name, s, reuse=False):
            with tf.name_scope(scope_name):
                with tf.variable_scope(scope_name, reuse=reuse):
                    x = s
                    x = make_fc('fc1', x, self.hidden_layer_sizes[0],
                                activation_fn=tf.nn.relu,
                                batch_norm=True
                                )

                    # x = tc.layers.layer_norm(x, center=True, scale=True)
                    for idx, size in enumerate(self.hidden_layer_sizes[1:]):
                        x = make_fc('fc{}'.format(idx + 2), x, size,
                                    activation_fn=tf.nn.relu,
                                    batch_norm=batch_norm,
                                    )
                    x = make_fc('fc{}'.format(len(self.hidden_layer_sizes) + 1), x, 1)
                    return x

        # build policy function
        self.action_prob_n, self.action_logits_n = actor_network('{}_actor'.format(name),
                                                                 s=self.global_state_input)

        # TODO: the goal agent may be already dead. (set goal to [0, 0, 0, 0])
        self.stochastic_action_n = []
        for agent_idx in range(training_agent_num):
            goal_action = tf.multinomial(logits=tf.nn.log_softmax(self.action_logits_n[agent_idx]), num_samples=1)
            # goal_action = tf.reshape(tf.multinomial(logits=tf.nn.log_softmax(self.action_logits_n[agent_idx]), num_samples=1), shape=[-1])
            one_hot_action = tf.one_hot(goal_action,
                                        depth=action_dim, dtype=tf.float32,
                                        on_value=1., off_value=0.
                                        )
            self.stochastic_action_n.append(
                tf.squeeze(tf.where(self.still_alive_input[agent_idx],
                                    one_hot_action,
                                    tf.zeros_like(one_hot_action)
                                    ))
            )

        self.greedy_action_n = []
        for agent_idx in range(training_agent_num):
            one_hot_action = tf.one_hot(
                tf.argmax(self.action_logits_n[agent_idx], axis=1), depth=action_dim, dtype=tf.float32,
                on_value=1., off_value=0.
            )
            self.greedy_action_n.append(
                tf.squeeze(tf.where(self.still_alive_input[agent_idx],
                                    one_hot_action,
                                    tf.zeros_like(one_hot_action)
                                    ))
            )

        # build q-value function
        self.q_output = q_network('{}_q_function'.format(name),
                                  s=self.global_state_input,
                                  a=tf.concat(self.q_action_input, axis=1))

        # build s-value function
        self.v_output = v_network('{}_v_function'.format(name), s=self.global_state_input)
        self.target_v_output = v_network('{}_target_v_function'.format(name), s=self.global_state_nxt_input)

        # networks parameters
        self.a_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='{}_actor'.format(name))

        self.q_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='{}_q_function'.format(name))

        self.v_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='{}_v_function'.format(name))
        self.v_params_vars_by_name = {var.name[len('{}_'.format(name)):]: var for var in self.v_params}
        self.vt_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='{}_target_v_function'.format(name))
        self.vt_params_vars_by_name = {var.name[len('{}_target_'.format(name)):]: var for var in self.vt_params}

        # build update function
        self.hard_replace = [copy_online_to_target(online_vars_by_name=self.v_params_vars_by_name,
                                                   target_vars_by_name=self.vt_params_vars_by_name)]

        self.soft_replace = [soft_update_online_to_target(online_vars_by_name=self.v_params_vars_by_name,
                                                          target_vars_by_name=self.vt_params_vars_by_name,
                                                          tau=self.tau)]

        # build actor loss and optimizer
        # cal advantage
        advantage = tf.stop_gradient(
            self.q_output - self.v_output
        )

        # compute the joint log(action probability)
        log_prob_n = []
        eps = 1e-10
        for agent_idx in range(training_agent_num):
            # TODO: action 可能是 [0, 0, 0, 0]
            # prob_a_alive = tf.boolean_mask(self.action_prob_n[agent_idx], mask=self.q_action_input[agent_idx])
            prob_a_alive = tf.reduce_sum(self.action_prob_n[agent_idx] * self.q_action_input[agent_idx], axis=1,
                                         keepdims=True)
            prob_a = tf.where(self.still_alive_input[agent_idx],
                              prob_a_alive,
                              tf.ones_like(prob_a_alive)
                              )
            prob_clip = tf.clip_by_value(prob_a, eps, 1.0 - eps)
            log_prob = tf.log(prob_clip)
            log_prob_n.append(tf.reshape(log_prob, shape=[-1, 1]))

        log_sum = tf.reduce_sum(tf.concat(log_prob_n, axis=1), axis=1, keepdims=True)
        actor_loss = -tf.reduce_mean(log_sum * advantage)
        self.actor_train = tf.train.AdamOptimizer(self.actor_lr).minimize(
            loss=actor_loss,
            var_list=self.a_params)

        # build v-function loss and optimizer
        online_q_value = q_network('{}_q_function'.format(name),
                                   s=self.global_state_input,
                                   a=tf.stop_gradient(tf.concat(self.stochastic_action_n, axis=1)),
                                   # TODO 单位也有可能已经死了，输出action应该为 0
                                   reuse=True
                                   )
        value_func_loss = 0.5 * tf.reduce_mean(
            (self.v_output - tf.stop_gradient(online_q_value)) ** 2)
        self.value_func_train = tf.train.AdamOptimizer(self.v_func_lr).minimize(
            loss=value_func_loss,
            var_list=self.v_params
        )

        # build q-function loss and optimizer
        q_target = tf.stop_gradient(
            tf.where(self.terminal, self.reward,
                     self.reward + self.gamma * self.target_v_output))
        q_func_loss = 0.5 * tf.reduce_mean(
            (q_target - self.q_output) ** 2)
        self.q_func_train = tf.train.AdamOptimizer(self.q_func_lr).minimize(
            loss=q_func_loss,
            var_list=self.q_params
        )

        # init replay buffer
        self.replay_buffer = ReplayBuffer(training_agent_num, replay_buffer_size)

    def train_actor(self, sess, state, action_n, alive_n):
        feed_q_action_input = {
            key: value for key, value in
            zip(self.q_action_input, action_n)
        }
        feed_still_alive_input = {
            key: value for key, value in
            zip(self.still_alive_input, alive_n)
        }
        feed_dict = {
            self.global_state_input: state,
            self.keep_prob: 1.,
        }
        feed_dict.update(feed_q_action_input)
        feed_dict.update(feed_still_alive_input)
        sess.run(self.actor_train, feed_dict=feed_dict)

    def train_value_function(self, sess, state, alive_n):
        feed_still_alive_input = {
            key: value for key, value in
            zip(self.still_alive_input, alive_n)
        }
        feed_dict = {
            self.global_state_input: state,
            self.keep_prob: 1.,
        }
        feed_dict.update(feed_still_alive_input)
        sess.run(self.value_func_train, feed_dict=feed_dict)

    def train_q_function(self, sess, state, action_n, reward, nxt_state, done):
        feed_q_action_input = {
            key: value for key, value in
            zip(self.q_action_input, action_n)
        }
        feed_dict = {
            self.global_state_input: state,
            self.reward: reward,
            self.global_state_nxt_input: nxt_state,
            self.terminal: done,
            self.keep_prob: 1.,
        }
        feed_dict.update(feed_q_action_input)
        sess.run(self.q_func_train, feed_dict=feed_dict)

    def train(self, sess, batch_size):
        obs, acts_n, rew, obs_nxt, done, alive_n = self.replay_buffer.sample(batch_size)
        self.train_actor(sess, obs, acts_n, alive_n)
        self.train_value_function(sess, obs, alive_n)
        self.train_q_function(sess, obs, acts_n, rew, obs_nxt, done)
        self.do_soft_update(sess)

    def get_greedy_action(self, sess, state, alive_n):
        feed_still_alive_input = {
            key: [value] for key, value in
            zip(self.still_alive_input, alive_n)
        }
        feed_dict = {
            self.global_state_input: state,
            self.keep_prob: 1.,
        }
        feed_dict.update(feed_still_alive_input)
        a_n = sess.run(self.greedy_action_n, feed_dict)
        return a_n

    def get_stochastic_action(self, sess, state, alive_n):
        feed_still_alive_input = {
            key: [value] for key, value in
            zip(self.still_alive_input, alive_n)
        }
        feed_dict = {
            self.global_state_input: state,
            self.keep_prob: 1.,
        }
        feed_dict.update(feed_still_alive_input)
        action_n = sess.run(self.stochastic_action_n, feed_dict)
        return action_n

    def experience(self, s, a_n, r, s_nxt, done, alive_n_t):
        self.replay_buffer.add(s, a_n, r, s_nxt, done, alive_n_t)

    def do_hard_replace(self, sess):
        sess.run(self.hard_replace)

    def do_soft_update(self, sess):
        sess.run(self.soft_replace)
