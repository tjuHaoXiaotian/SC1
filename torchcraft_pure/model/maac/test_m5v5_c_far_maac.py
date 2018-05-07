import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用 GPU 0
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # 使用 GPU 0，1
import numpy as np
import tensorflow as tf
from torchcraft_pure.env.BaseEnv import BaseEnv
import time

from torchcraft_pure.model.maac.maac import MAAC

gpu_fraction = 0.1
batch_size = 64
agent_num = 10
single_state_dim = 20
episodes = 6000
n_train_repeat = 1
buffer_size = 0
min_pool_size = buffer_size / 10
save_checkpoint_every_episode = 100

reload_dir_root = "m5v5"
save_dir_root = "m5v5_test"
model_name = 'maac'
save_dir = 1524809943

'''
TODO: 单位死亡后，critic 观察无法拼接，目前给死亡时，replay buffer 存入state/action全为0
'''

# test predator
agent1 = MAAC(name='marine0', agent_idx=0, state_dim=single_state_dim * agent_num, action_dim=10, training_agent_num=5,
              hidden_layer_sizes=[256, 64], actor_lr=1e-4, critic_lr=1e-4, batch_norm=True,
              replay_buffer_size=buffer_size)
agent2 = MAAC(name='marine1', agent_idx=1, state_dim=single_state_dim * agent_num, action_dim=10, training_agent_num=5,
              hidden_layer_sizes=[256, 64], actor_lr=1e-4, critic_lr=1e-4, batch_norm=True,
              replay_buffer_size=buffer_size)
agent3 = MAAC(name='marine2', agent_idx=2, state_dim=single_state_dim * agent_num, action_dim=10, training_agent_num=5,
              hidden_layer_sizes=[256, 64], actor_lr=1e-4, critic_lr=1e-4, batch_norm=True,
              replay_buffer_size=buffer_size)
agent4 = MAAC(name='marine3', agent_idx=3, state_dim=single_state_dim * agent_num, action_dim=10, training_agent_num=5,
              hidden_layer_sizes=[256, 64], actor_lr=1e-4, critic_lr=1e-4, batch_norm=True,
              replay_buffer_size=buffer_size)
agent5 = MAAC(name='marine4', agent_idx=4, state_dim=single_state_dim * agent_num, action_dim=10, training_agent_num=5,
              hidden_layer_sizes=[256, 64], actor_lr=1e-4, critic_lr=1e-4, batch_norm=True,
              replay_buffer_size=buffer_size)
agents = [agent1, agent2, agent3, agent4, agent5]

saver = tf.train.Saver(max_to_keep=100)


def get_agents_action(sess, o_n):
    actions = []
    for idx, o in enumerate(o_n):
        if o is not None:
            a = agents[idx].greedy_action(sess, [o])
            # because this has been one-hot
            a = np.argmax(a)
        else:
            a = None
        actions.append(a)
    return actions


if __name__ == '__main__':
    env = BaseEnv()

    # summary graph
    agent_reward_v = [tf.Variable(0, dtype=tf.float32) for i in range(agent_num)]
    agent_reward_op = [tf.summary.scalar('agent' + str(i) + '_reward', agent_reward_v[i]) for i in range(agent_num)]
    # agent_a1 = [tf.Variable(0, dtype=tf.float32) for i in range(agent_num)]
    # agent_a1_op = [tf.summary.scalar('agent' + str(i) + '_action_1', agent_a1[i]) for i in range(agent_num)]
    # agent_a2 = [tf.Variable(0, dtype=tf.float32) for i in range(agent_num)]
    # agent_a2_op = [tf.summary.scalar('agent' + str(i) + '_action_2', agent_a2[i]) for i in range(agent_num)]
    reward_100 = [tf.Variable(0, dtype=tf.float32) for i in range(agent_num)]
    reward_100_op = [tf.summary.scalar('agent' + str(i) + '_reward_100_mean', reward_100[i]) for i in range(agent_num)]
    reward_1000 = [tf.Variable(0, dtype=tf.float32) for i in range(agent_num)]
    reward_1000_op = [tf.summary.scalar('agent' + str(i) + '_reward_1000_mean', reward_1000[i]) for i in
                      range(agent_num)]
    reward_episode = [tf.Variable(0, dtype=tf.float32) for i in range(agent_num)]
    reward_episode_op = [tf.summary.scalar('agent' + str(i) + '_reward_episode', reward_episode[i]) for i in
                         range(agent_num)]
    # win times recent 100
    recent_100_win = tf.Variable(0, dtype=tf.float32)
    recent_100_win_op = tf.summary.scalar('recent_100_win', recent_100_win)
    # killed enemy number current episode
    killed_enemy_episode = tf.Variable(0, dtype=tf.float32)
    killed_enemy_episode_op = tf.summary.scalar('killed_enemy_episode', killed_enemy_episode)

    config = tf.ConfigProto()
    # 设置每个GPU应该拿出多少容量给进程使用，0.4代表 40%
    config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # reload model and parameters
    dir = "./{}/checkpoints/{}/{}".format(reload_dir_root, model_name, save_dir)
    saver.restore(sess, "{}/model_{}.ckpt".format(dir, 600))

    # init summary writer
    summary_writer = tf.summary.FileWriter('./{}/summaries/{}/{}'.format(save_dir_root, model_name, save_dir),
                                           graph=tf.get_default_graph())

    total_steps = 0
    reward_100_list = [[]] * agent_num
    recent_100_win_list = []
    for episode in range(episodes):
        step = 0
        [o_n, global_obs], done = env.reset(), False
        print('env reset!')
        ep_reward = [0] * agent_num
        killed_opp_mine = 0
        while not done:
            actions = get_agents_action(sess=sess, o_n=o_n[0:5])
            action_queue = [env.convert_discrete_action_2_sc1_action(agent_id, act_id) for agent_id, act_id in
                            enumerate(actions) if act_id is not None]

            [o_n_next, r_n, d_n, i_n], [global_obs, global_reward, done, info] = env.step(action_queue)
            killed_opp_mine += max(i_n['killed_enemy_num'][0:5])
            # record the reward
            for agent_index in range(agent_num):
                reward_100_list[agent_index].append(r_n[agent_index])
                reward_100_list[agent_index] = reward_100_list[agent_index][-1000:]

            o_n = o_n_next
            done = env.is_end()
            step += 1
            total_steps += 1

            # record episode reward
            for idx in range(agent_num):
                ep_reward[idx] += r_n[idx]

            # record something for each 100 step
            if total_steps % 100 == 0:
                for agent_index in range(agent_num):
                    summary_writer.add_summary(
                        sess.run(agent_reward_op[agent_index], {agent_reward_v[agent_index]: r_n[agent_index]}),
                        total_steps)
                    summary_writer.add_summary(sess.run(reward_100_op[agent_index], {
                        reward_100[agent_index]: np.mean(reward_100_list[agent_index][-100:])}), total_steps)

            if total_steps % 1000 == 0:
                # record the avg reward every 1000 step.
                for agent_index in range(agent_num):
                    summary_writer.add_summary(sess.run(reward_1000_op[agent_index],
                                                        {reward_1000[agent_index]: np.mean(
                                                            reward_100_list[agent_index])}),
                                               total_steps // 1000)

        # record episode reward and win times recent 100 episode
        for agent_index in range(agent_num):
            summary_writer.add_summary(sess.run(reward_episode_op[agent_index],
                                                {reward_episode[agent_index]: ep_reward[agent_index]}), episode)

        recent_100_win_list.append(1 if env.is_win() else 0)
        recent_100_win_list = recent_100_win_list[-100:]
        summary_writer.add_summary(sess.run(recent_100_win_op,
                                            {recent_100_win: np.sum(recent_100_win_list)}), episode)

        summary_writer.add_summary(sess.run(killed_enemy_episode_op,
                                            {killed_enemy_episode: killed_opp_mine}), episode)

    sess.close()