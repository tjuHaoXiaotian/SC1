import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用 GPU 0
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # 使用 GPU 0，1
import numpy as np
import tensorflow as tf
from torchcraft_pure.env.BaseEnv import BaseEnv
import time

from torchcraft_pure.model.hierarchical.meta_controller import META_MAAC
from torchcraft_pure.model.hierarchical.controller import AC

gpu_fraction = 0.5
batch_size = 64
agent_num = 10
enemy_num = 5
single_state_dim = 20
episodes = 40000
n_train_repeat = 1
meta_buffer_size = 100000
buffer_size = 10000
min_pool_size = buffer_size / 2
save_checkpoint_every_episode = 100

save_dir_root = "m5v5"
model_name = 'hierarch'
save_dir = int(time.time())

'''
TODO: 单位死亡后，critic 观察无法拼接，目前给死亡时，replay buffer 存入state/action全为0
'''

# goal: attack one enemy
agent0 = META_MAAC(name='meta_controller', state_dim=single_state_dim * agent_num, action_dim=enemy_num,
                   training_agent_num=5,
                   hidden_layer_sizes=[256, 64], actor_lr=1e-4, q_func_lr=1e-3, v_func_lr=1e-3, batch_norm=True,
                   replay_buffer_size=meta_buffer_size)

# original action: attack or move
agent1 = AC(name='marine0', state_dim=single_state_dim * agent_num, goal_dim=single_state_dim, action_dim=6,
            hidden_layer_sizes=[256, 64], actor_lr=1e-4, q_func_lr=1e-3, v_func_lr=1e-3, batch_norm=True,
            replay_buffer_size=buffer_size)
agent2 = AC(name='marine1', state_dim=single_state_dim * agent_num, goal_dim=single_state_dim, action_dim=6,
            hidden_layer_sizes=[256, 64], actor_lr=1e-4, q_func_lr=1e-3, v_func_lr=1e-3, batch_norm=True,
            replay_buffer_size=buffer_size)
agent3 = AC(name='marine2', state_dim=single_state_dim * agent_num, goal_dim=single_state_dim, action_dim=6,
            hidden_layer_sizes=[256, 64], actor_lr=1e-4, q_func_lr=1e-3, v_func_lr=1e-3, batch_norm=True,
            replay_buffer_size=buffer_size)
agent4 = AC(name='marine3', state_dim=single_state_dim * agent_num, goal_dim=single_state_dim, action_dim=6,
            hidden_layer_sizes=[256, 64], actor_lr=1e-4, q_func_lr=1e-3, v_func_lr=1e-3, batch_norm=True,
            replay_buffer_size=buffer_size)
agent5 = AC(name='marine4', state_dim=single_state_dim * agent_num, goal_dim=single_state_dim, action_dim=6,
            hidden_layer_sizes=[256, 64], actor_lr=1e-4, q_func_lr=1e-3, v_func_lr=1e-3, batch_norm=True,
            replay_buffer_size=buffer_size)

agents = [agent1, agent2, agent3, agent4, agent5]

saver = tf.train.Saver(max_to_keep=100)


def get_sub_actions_n(sess, o_n, goal_n):
    one_hot_actions, actions = [], []
    for idx, (o, goal), in enumerate(zip(o_n, goal_n)):
        if o is not None:  # still alive
            goal_unit_id = goal + (agent_num - enemy_num)
            goal_state = o[goal_unit_id * single_state_dim: (goal_unit_id + 1) * single_state_dim]
            a = agents[idx].get_stochastic_action(sess, [o], [goal_state])
        else:  # already dead
            a = None
        one_hot_actions.append(a)
        actions.append(np.argmax(a) if a is not None else None)
    return one_hot_actions, actions


def train_sub_agent(sess, agent_list, batch_size=batch_size):
    for agent in agent_list:
        agent.train(sess, batch_size)


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
    # global reward
    global_reward_episode = tf.Variable(0, dtype=tf.float32)
    global_reward_episode_op = tf.summary.scalar('global_reward_episode', global_reward_episode)
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

    # do hard replace
    agent0.do_hard_replace(sess)
    for agent in agents:
        agent.do_hard_replace(sess)

    # init summary writer
    summary_writer = tf.summary.FileWriter('./{}/summaries/{}/{}'.format(save_dir_root, model_name, save_dir),
                                           graph=tf.get_default_graph())

    total_steps = 0
    reward_100_list = [[]] * agent_num
    recent_100_win_list = []
    goal_idx = 0
    for episode in range(episodes):
        step = 0
        [o_n, global_obs], done = env.reset(), False
        print('env reset!')
        ep_reward = [0] * agent_num
        episode_global_reward = 0
        killed_opp_mine = 0

        while not done:
            # pick a goal
            s_t = global_obs
            s_tp1 = global_obs
            # alive units at time t
            alive_n_t = [True if o is not None else False for o in o_n[0:(agent_num - enemy_num)]]

            # select a goal for each alive agent
            one_hot_goal_n = agent0.get_stochastic_action(sess, state=[s_t], alive_n=alive_n_t)
            # convert to index
            goal_n = [np.argmax(one_hot_goal) for one_hot_goal in one_hot_goal_n]  # all zeros (dead agent) will => 0 (stop)
            # reset goal
            env.reset_goal_step()
            # goal reward
            goal_return = 0
            while not (done or env.is_goal_reached()):
                # print("goal: ", goal_idx)
                one_hot_actions_n, action_ids_n = get_sub_actions_n(sess=sess, o_n=o_n[0:(agent_num-enemy_num)], goal_n=goal_n)

                action_queue = [env.convert_discrete_action_2_sc1_action(agent_id, act_id, attack_target_id=goal_id) for
                                agent_id, (act_id, goal_id) in enumerate(zip(action_ids_n, goal_n)) if
                                act_id is not None]

                [o_n_next, r_n, d_n, i_n], [global_obs, global_reward, done, info] = env.step(action_queue)
                killed_opp_mine += info['killed_enemy_num']
                # record the reward
                for agent_index in range(agent_num):
                    reward_100_list[agent_index].append(r_n[agent_index])
                    reward_100_list[agent_index] = reward_100_list[agent_index][-1000:]

                # record replay buffer
                for idx, agent in enumerate(agents):
                    if o_n_next[idx] is not None:  # alive or just dead: the corresponding action is legal.
                        goal_unit_id = goal_n[idx] + (agent_num - enemy_num)
                        goal_state = o_n[idx][goal_unit_id * single_state_dim: (goal_unit_id + 1) * single_state_dim]
                        goal_state_nxt = o_n_next[idx][goal_unit_id * single_state_dim: (goal_unit_id + 1) * single_state_dim]
                        assert one_hot_actions_n[idx] is not None # alive agent
                        agent.experience(
                            s=o_n[idx],
                            goal=goal_state,
                            a=one_hot_actions_n[idx],
                            r=r_n[idx],
                            s_nxt=o_n_next[idx],
                            goal_nxt=goal_state_nxt,
                            done=d_n[idx])
                # if experience is enough, then begin training (sub controller).
                if total_steps >= min_pool_size and total_steps % 1 == 0:
                    if total_steps == min_pool_size:
                        print('begin training...')
                    for _ in range(n_train_repeat):
                        train_sub_agent(sess, agents, batch_size)

                # record episode reward
                for idx in range(agent_num):
                    ep_reward[idx] += r_n[idx]
                episode_global_reward += global_reward

                # record something for each 100 step
                if total_steps % 100 == 0:
                    for agent_index in range(agent_num):
                        summary_writer.add_summary(
                            sess.run(agent_reward_op[agent_index],
                                     {agent_reward_v[agent_index]: r_n[agent_index]}),
                            total_steps)
                        summary_writer.add_summary(sess.run(reward_100_op[agent_index], {
                            reward_100[agent_index]: np.mean(reward_100_list[agent_index][-100:])}),
                                                   total_steps)

                # record something for each 1000 step
                if total_steps % 1000 == 0:
                    # record the avg reward every 1000 step.
                    for agent_index in range(agent_num):
                        summary_writer.add_summary(sess.run(reward_1000_op[agent_index],
                                                            {reward_1000[agent_index]: np.mean(
                                                                reward_100_list[agent_index])}),
                                                   total_steps // 1000)

                o_n = o_n_next
                done = env.is_end()
                step += 1
                total_steps += 1
                goal_return += global_reward
                s_tp1 = global_obs
            goal_idx += 1
            agent0.experience(
                s=s_t,
                a_n=one_hot_goal_n,
                r=goal_return,
                s_nxt=s_tp1,
                done=done,
                alive_n_t=alive_n_t
            )

            # train meta controller
            if len(agent0.replay_buffer) >= min_pool_size:
                agent0.train(sess, batch_size)



        # record episode reward and win times recent 100 episode
        for agent_index in range(agent_num):
            summary_writer.add_summary(sess.run(reward_episode_op[agent_index],
                                                {reward_episode[agent_index]: ep_reward[agent_index]}), episode)
        summary_writer.add_summary(sess.run(global_reward_episode_op,
                                            {global_reward_episode: episode_global_reward}), episode)

        recent_100_win_list.append(1 if env.is_win() else 0)
        recent_100_win_list = recent_100_win_list[-100:]
        summary_writer.add_summary(sess.run(recent_100_win_op,
                                            {recent_100_win: np.sum(recent_100_win_list)}), episode)

        summary_writer.add_summary(sess.run(killed_enemy_episode_op,
                                            {killed_enemy_episode: killed_opp_mine}), episode)

        # save checkpoints
        if episode % save_checkpoint_every_episode == 0 and episode > 0:
            print('save checkpoints!')
            dir = "./{}/checkpoints/{}/{}".format(save_dir_root, model_name, save_dir)
            if not os.path.exists(dir):
                os.makedirs(dir)
            saver.save(sess, "{}/model_{}.ckpt".format(dir, episode))

    sess.close()
