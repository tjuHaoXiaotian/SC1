from torchcraft_pure.model.util.nn_util import *
from torchcraft_pure.model.hierarchical.goal_replay_buffer import ReplayBuffer
import numpy as np


class AC(object):
    def __init__(self, name, state_dim, goal_dim, action_dim, hidden_layer_sizes, actor_lr=1e-4,
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
        self.state_input = tf.placeholder(shape=[None, state_dim], dtype=tf.float32,
                                                 name='{}/actor_state_input'.format(name))
        self.goal_state = tf.placeholder(shape=[None, goal_dim], dtype=tf.float32,
                                                 name='{}/goal_state_input'.format(name))

        self.state_nxt_input = tf.placeholder(shape=[None, state_dim], dtype=tf.float32,
                                                     name='{}/actor_state_nxt_input'.format(name))
        self.goal_state_nxt = tf.placeholder(shape=[None, goal_dim], dtype=tf.float32,
                                                 name='{}/goal_state_nxt_input'.format(name))

        self.q_action_input = tf.placeholder(shape=[None, action_dim], dtype=tf.float32,
                                              name='{}/q_action_input'.format(name))


        self.reward = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='{}/reward'.format(name))
        self.keep_prob = tf.placeholder(shape=[], dtype=tf.float32, name='{}/keep_prob'.format(name))
        # self.is_training = tf.placeholder(shape=[], dtype=tf.bool, name='{}/is_training'.format(name))
        self.terminal = tf.placeholder(shape=[None, 1], dtype=tf.bool, name='{}/terminal'.format(name))

        # def global actor network architecture
        def actor_network(scope_name, s, goal, reuse=False):
            with tf.name_scope(scope_name):
                with tf.variable_scope(scope_name, reuse=reuse):
                    x = tf.concat([s, goal], axis=1)
                    x = make_fc('fc1', x, self.hidden_layer_sizes[0],
                                activation_fn=tf.nn.relu,
                                batch_norm=True
                                )

                    for idx, size in enumerate(self.hidden_layer_sizes[1:]):
                        x = make_fc('fc{}'.format(idx + 2), x, size,
                                    activation_fn=tf.nn.relu,
                                    batch_norm=batch_norm,
                                    )
                    x = make_fc('fc{}'.format(len(self.hidden_layer_sizes) + 1), x,
                                action_dim,
                                activation_fn=None)
                    logits = x
                    x = tf.nn.softmax(logits=logits)
                    return x, logits

        def q_network(scope_name, s, goal, reuse=False):
            with tf.name_scope(scope_name):
                with tf.variable_scope(scope_name, reuse=reuse):
                    x = tf.concat([s, goal], axis=1)
                    x = make_fc('fc1', x, self.hidden_layer_sizes[0],
                                activation_fn=tf.nn.relu,
                                batch_norm=True
                                )

                    for idx, size in enumerate(self.hidden_layer_sizes[1:]):
                        x = make_fc('fc{}'.format(idx + 2), x, size,
                                    activation_fn=tf.nn.relu,
                                    batch_norm=batch_norm,
                                    )
                    x = make_fc('fc{}'.format(len(self.hidden_layer_sizes) + 1), x, action_dim)
                    return x

        def v_network(scope_name, s, goal, reuse=False):
            with tf.name_scope(scope_name):
                with tf.variable_scope(scope_name, reuse=reuse):
                    x = tf.concat([s, goal], axis=1)
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
        self.action_prob, self.action_logits = actor_network('{}_actor'.format(name),
                                                                 s=self.state_input,
                                                                 goal=self.goal_state
                                                             )

        self.stochastic_action = tf.squeeze(tf.one_hot(
            tf.multinomial(logits=tf.nn.log_softmax(self.action_logits), num_samples=1),
            depth=action_dim, dtype=tf.float32,
            on_value=1., off_value=0.
        ))

        self.greedy_action = tf.squeeze(tf.one_hot(
            tf.argmax(self.action_logits, axis=1),
            depth=action_dim, dtype=tf.float32,
            on_value=1., off_value=0.
        ))

        # build q-value function
        self.q_output = q_network('{}_q_function'.format(name), s=self.state_input, goal=self.goal_state)

        # build s-value function
        self.v_output = v_network('{}_v_function'.format(name), s=self.state_input, goal=self.goal_state)
        self.target_v_output = v_network('{}_target_v_function'.format(name), s=self.state_nxt_input, goal=self.goal_state_nxt)

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
            tf.reduce_sum(self.q_output * self.q_action_input, axis=1, keepdims=True) - self.v_output
        )

        # compute the joint log(action probability)
        eps = 1e-10
        prob_a = tf.boolean_mask(self.action_prob, mask=self.q_action_input)
        prob_clip = tf.clip_by_value(prob_a, eps, 1.0 - eps)
        log_prob = tf.log(prob_clip)
        log_prob = tf.reshape(log_prob, shape=[-1, 1])
        actor_loss = -tf.reduce_mean(log_prob * advantage)
        self.actor_train = tf.train.AdamOptimizer(self.actor_lr).minimize(
            loss=actor_loss,
            var_list=self.a_params)

        # build v-function loss and optimizer
        # online_q_value = tf.boolean_mask(self.q_output, mask=self.stochastic_action)
        # online_q_value = tf.reshape(online_q_value, shape=[online_q_value.shape[0], 1])
        online_q_value = tf.reduce_sum(self.q_output * self.stochastic_action, axis=1, keepdims=True)
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
            (q_target - tf.reduce_sum(self.q_output * self.q_action_input, axis=1, keepdims=True)) ** 2)
        self.q_func_train = tf.train.AdamOptimizer(self.q_func_lr).minimize(
            loss=q_func_loss,
            var_list=self.q_params
        )

        # init replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

    def train_actor(self, sess, state, goal, action):
        feed_dict = {
            self.state_input: state,
            self.goal_state: goal,
            self.q_action_input: action,
            self.keep_prob: 1.,
        }
        sess.run(self.actor_train, feed_dict=feed_dict)

    def train_value_function(self, sess, state, goal):
        sess.run(self.value_func_train, feed_dict={
            self.state_input: state,
            self.goal_state: goal,
            self.keep_prob: 1.,
        })

    def train_q_function(self, sess, state, goal, action, reward, nxt_state, goal_nxt, done):
        feed_dict = {
            self.state_input: state,
            self.goal_state: goal,
            self.q_action_input: action,
            self.reward: reward,
            self.state_nxt_input: nxt_state,
            self.goal_state_nxt: goal_nxt,
            self.terminal: done,
            self.keep_prob: 1.
        }
        sess.run(self.q_func_train, feed_dict=feed_dict)

    def train(self, sess, batch_size):
        obs, goals, acts, rew, obs_nxt, goals_nxt, done = self.replay_buffer.sample(batch_size)
        self.train_actor(sess, state=obs, goal=goals, action=acts)
        self.train_value_function(sess, state=obs, goal=goals)
        self.train_q_function(sess, state=obs, goal=goals, action=acts, reward=rew, nxt_state=obs_nxt, goal_nxt=goals_nxt, done=done)
        self.do_soft_update(sess)

    def get_greedy_action(self, sess, state, goal):
        a = sess.run(self.greedy_action, feed_dict={
            self.state_input: state,
            self.goal_state: goal,
            self.keep_prob: 1.,
        })
        return a

    def get_stochastic_action(self, sess, state, goal):
        action = sess.run(self.stochastic_action, feed_dict={
            self.state_input: state,
            self.goal_state: goal,
            self.keep_prob: 1.,
        })
        return action

    def experience(self, s, goal, a, r, s_nxt, goal_nxt, done):
        self.replay_buffer.add(s, goal, a, r, s_nxt, goal_nxt, done)

    def do_hard_replace(self, sess):
        sess.run(self.hard_replace)

    def do_soft_update(self, sess):
        sess.run(self.soft_replace)
