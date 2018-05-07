from torchcraft_pure.model.util.nn_util import *
from torchcraft_pure.model.util.replay_buffer import ReplayBuffer
import numpy as np


class MAAC(object):
    def __init__(self, agent_idx, name, state_dim, action_dim, training_agent_num, hidden_layer_sizes, actor_lr=1e-4,
                 critic_lr=1e-3, batch_norm=False,
                 replay_buffer_size=100000):
        # hyper parameters
        self.agent_idx = agent_idx
        self.gamma = 0.95
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = 0.01

        self.hidden_layer_sizes = hidden_layer_sizes

        # build placeholders
        # for actor
        self.actor_state_input = tf.placeholder(shape=[None, state_dim], dtype=tf.float32,
                                                name='{}/actor_state_input'.format(name))
        self.actor_state_nxt_input = tf.placeholder(shape=[None, state_dim], dtype=tf.float32,
                                                    name='{}/actor_state_nxt_input'.format(name))
        # for critic
        self.critic_state_input = [tf.placeholder(shape=[None, state_dim], dtype=tf.float32,
                                                  name='{}/critic_state_input/{}'.format(name, _)) for _ in
                                   range(training_agent_num)]
        self.critic_self_action_input = tf.placeholder(shape=[None, action_dim], dtype=tf.float32,
                                                       name='{}/critic_self_action_input/{}'.format(name,
                                                                                                    self.agent_idx))
        self.critic_action_input = [tf.placeholder(shape=[None, action_dim], dtype=tf.float32,
                                                   name='{}/critic_action_input/{}'.format(name, _)) for _ in
                                    range(training_agent_num)]
        self.critic_state_nxt_input = [tf.placeholder(shape=[None, state_dim], dtype=tf.float32,
                                                      name='{}/critic_state_nxt_input/{}'.format(name, _)) for _ in
                                       range(training_agent_num)]
        self.critic_self_nxt_action_input = tf.placeholder(shape=[None, action_dim], dtype=tf.float32,
                                                           name='{}/critic_self_nxt_action_input/{}'.format(name,
                                                                                                            self.agent_idx))
        self.critic_nxt_action_input = [tf.placeholder(shape=[None, action_dim], dtype=tf.float32,
                                                       name='{}/critic_nxt_action_input/{}'.format(name, _)) for _ in
                                        range(training_agent_num)]

        self.reward = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='{}/reward'.format(name))
        self.keep_prob = tf.placeholder(shape=[], dtype=tf.float32, name='{}/keep_prob'.format(name))
        # self.is_training = tf.placeholder(shape=[], dtype=tf.bool, name='{}/is_training'.format(name))
        self.terminal = tf.placeholder(shape=[None, 1], dtype=tf.bool, name='{}/terminal'.format(name))

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

                    x = make_fc('fc{}'.format(len(self.hidden_layer_sizes) + 1), x,
                                action_dim,
                                activation_fn=None)
                    logits = x
                    x = tf.nn.softmax(logits=logits)
                    return x, logits

        def critic_network(scope_name, s, a, reuse=False):
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
                    x = make_fc('fc{}'.format(len(self.hidden_layer_sizes) + 1), x, action_dim)
                    return x

        self.action_output_porb, self.action_logits = actor_network('{}_actor'.format(name), s=self.actor_state_input)
        self.stochastic_action = tf.multinomial(logits=tf.nn.log_softmax(self.action_logits), num_samples=1)
        # self.stochastic_action = tf.multinomial(logits=self.action_logits, num_samples=1)
        self.target_action_output_porb, self.target_action_logits = actor_network('{}_target_actor'.format(name),
                                                                                  s=self.actor_state_nxt_input)
        self.target_action_output = tf.one_hot(
            tf.argmax(self.target_action_logits, axis=1), depth=action_dim, dtype=tf.int32,
            on_value=1, off_value=0
        )

        self.q_output = critic_network('{}_critic'.format(name),
                                       s=tf.concat(self.critic_state_input, axis=1),
                                       a=tf.concat(self.critic_action_input, axis=1))
        self.target_q_output = critic_network('{}_target_critic'.format(name),
                                              s=tf.concat(self.critic_state_nxt_input, axis=1),
                                              a=tf.concat(self.critic_nxt_action_input, axis=1))

        # networks parameters
        self.a_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='{}_actor'.format(name))
        self.a_params_vars_by_name = {var.name[len('{}_'.format(name)):]: var for var in self.a_params}
        # print(self.a_params_vars_by_name)
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='{}_target_actor'.format(name))
        self.at_params_vars_by_name = {var.name[len('{}_target_'.format(name)):]: var for var in self.at_params}
        # print(self.at_params_vars_by_name)

        self.c_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='{}_critic'.format(name))
        self.c_params_vars_by_name = {var.name[len('{}_'.format(name)):]: var for var in self.c_params}
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='{}_target_critic'.format(name))
        self.ct_params_vars_by_name = {var.name[len('{}_target_'.format(name)):]: var for var in self.ct_params}

        # build update function
        self.hard_replace = [copy_online_to_target(online_vars_by_name=self.a_params_vars_by_name,
                                                   target_vars_by_name=self.at_params_vars_by_name),
                             copy_online_to_target(online_vars_by_name=self.c_params_vars_by_name,
                                                   target_vars_by_name=self.ct_params_vars_by_name)]

        self.soft_replace = [soft_update_online_to_target(online_vars_by_name=self.a_params_vars_by_name,
                                                          target_vars_by_name=self.at_params_vars_by_name,
                                                          tau=self.tau),
                             soft_update_online_to_target(online_vars_by_name=self.c_params_vars_by_name,
                                                          target_vars_by_name=self.ct_params_vars_by_name,
                                                          tau=self.tau)]

        # build actor loss and optimizer
        # cal advantage
        advantage = tf.stop_gradient(
            tf.reduce_sum(self.q_output * self.critic_self_action_input, axis=1, keepdims=True) - tf.reduce_sum(
                self.q_output * self.action_output_porb, axis=1, keepdims=True)
        )
        # prevent log(0) easy to cause NAN
        prob_a = tf.reduce_sum(self.action_output_porb * self.critic_self_action_input, axis=1, keepdims=True)
        eps = 1e-10
        prob_clip = tf.clip_by_value(prob_a, eps, 1.0 - eps)
        actor_loss = -tf.reduce_mean(
            tf.log(prob_clip) * advantage
        )
        self.actor_optimizer = tf.train.AdamOptimizer(self.actor_lr)
        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='{}_actor'.format(name))):
        self.actor_train = self.actor_optimizer.minimize(actor_loss, var_list=self.a_params)

        # build critic loss and optimizer
        q_target = tf.stop_gradient(
            tf.where(self.terminal, self.reward,
                     self.reward + self.gamma * tf.reduce_sum(self.target_q_output * self.critic_self_nxt_action_input,
                                                              axis=1, keepdims=True)))
        critic_loss = 0.5 * tf.reduce_mean(
            tf.square(q_target - tf.reduce_sum(self.q_output * self.critic_self_action_input, axis=1, keepdims=True)))
        self.critic_optimizer = tf.train.AdamOptimizer(self.critic_lr)

        # c1 = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='{}_critic'.format(name))
        # c2 = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='{}_critic_1'.format(name))
        # with tf.control_dependencies(c1.extend(c2)):
        self.critic_train = self.critic_optimizer.minimize(critic_loss, var_list=self.c_params)

        # init replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

    def train_actor(self, sess, state_n, action_n):
        feed_critic_state_input = {
            key: value for key, value in zip(self.critic_state_input, state_n)
        }
        feed_critic_action_input = {
            key: value if idx != self.agent_idx else np.zeros_like(action_n[self.agent_idx]) for idx, (key, value) in
            enumerate(zip(self.critic_action_input, action_n))
        }
        feed_dict = {
            self.actor_state_input: state_n[self.agent_idx],
            self.critic_self_action_input: action_n[self.agent_idx],
            self.keep_prob: 1.,
        }
        feed_dict.update(feed_critic_state_input)
        feed_dict.update(feed_critic_action_input)
        sess.run(self.actor_train, feed_dict=feed_dict)

    def train_critic(self, sess, state_n, action_n, reward, nxt_state_n, nxt_action_n, done):
        feed_critic_state_input = {
            key: value for key, value in zip(self.critic_state_input, state_n)
        }
        feed_critic_action_input = {
            key: value if idx != self.agent_idx else np.zeros_like(action_n[self.agent_idx]) for idx, (key, value) in
            enumerate(zip(self.critic_action_input, action_n))
        }
        feed_critic_state_nxt_input = {
            key: value for key, value in zip(self.critic_state_nxt_input, nxt_state_n)
        }
        feed_critic_nxt_action_input = {
            key: value if idx != self.agent_idx else np.zeros_like(nxt_action_n[self.agent_idx]) for idx, (key, value)
            in enumerate(zip(self.critic_nxt_action_input, nxt_action_n))
        }
        feed_dict = {
            self.critic_self_action_input: action_n[self.agent_idx],
            self.reward: reward,
            self.critic_self_nxt_action_input: nxt_action_n[self.agent_idx],
            self.terminal: done,
            self.keep_prob: 1.,
        }
        feed_dict.update(feed_critic_state_input)
        feed_dict.update(feed_critic_action_input)
        feed_dict.update(feed_critic_state_nxt_input)
        feed_dict.update(feed_critic_nxt_action_input)
        sess.run(self.critic_train, feed_dict=feed_dict)

    def train(self, sess, agent_list, batch_size):
        # 应该是我活着的时候的 sample
        index = self.replay_buffer.make_index(batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        act_next_n = []
        # act_next_n_state_ph = []
        for i in range(len(agent_list)):
            obs, act, rew, obs_next, done = agent_list[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
            # TODO: 这里有的单已经死了（act_next 最好是 0，这里不太对）
            act_next_n.append(sess.run(agent_list[i].target_action_output, feed_dict={
                agent_list[i].actor_state_nxt_input: obs_next
            }))
            # act_next_n.append(agent_list[i].target_action_output)
            # act_next_n_state_ph.append(agent_list[i].actor_state_nxt_input)
        # action_nxt_n_feed = {key: value for key, value in zip(act_next_n_state_ph, obs_next_n)}
        # action_nxt_n = sess.run(tf.concat(act_next_n, axis=1), feed_dict=action_nxt_n_feed)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        self.train_critic(sess, obs_n, act_n, rew, obs_next_n, act_next_n, done)
        self.train_actor(sess, obs_n, act_n)
        self.do_soft_update(sess)

    def greedy_action(self, sess, state):
        # TODO: 输出趋向于负无穷？？？
        a, a_p, logits = sess.run(
            [self.target_action_output, self.target_action_output_porb, self.target_action_logits], feed_dict={
                self.actor_state_nxt_input: state,
                self.keep_prob: 1.,
                # self.is_training: False
            })
        print(np.argmax(a), ": ", a_p, '             ', logits)
        if logits[0][0] != logits[0][0]:
            print('state: ', state)
        return a

    def action(self, sess, state):
        action, _ = sess.run([self.stochastic_action, self.action_output_porb], feed_dict={
            self.actor_state_input: state,
            self.keep_prob: 1.,
            # self.is_training: False
        })
        action = np.squeeze(action)
        if action != action:
            print("NAN occured.")
            print('action: ', action, ', prob: ', _)
        return action

    def experience(self, s, a, r, s_nxt, done, still_alive):
        self.replay_buffer.add(s, a, r, s_nxt, done, still_alive)

    def do_hard_replace(self, sess):
        sess.run(self.hard_replace)

    def do_soft_update(self, sess):
        sess.run(self.soft_replace)
