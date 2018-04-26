#!/usr/bin/python
# -*- coding: UTF-8 -*-

from torchcraft_pure.env.Env5v5 import Env5v5, dprint
import time

from torchcraft_pure.agent.general.agent import Agent
from torchcraft_pure.agent.general.parameters import parameters

# hyper parameters
parameters.episodes = 1000
parameters.training = True




if __name__ == "__main__":
    env = Env5v5()
    agent = Agent(training=parameters.training)

    for ep in range(parameters.episodes):
        state, done = env.reset()

        # while battle not end
        while not done:
            alive_friends = state[0]
            alive_enemies = state[1]
            if env.is_first():
                # reset the system order of all units
                # reset the health value of all units
                env.reset_units_order(alive_friends, alive_enemies)

            action_queue = []
            cur_alive_friends_info = {}
            for friend in alive_friends:
                local_observation = env.make_local_observation(alive_friends, alive_enemies, friend)
                # print(local_observation)
                action_id = agent.select_action([local_observation])
                action_sc1 = env.convert_discrete_action_2_sc1_action(friend, action_id, alive_enemies)
                action_queue.append(action_sc1)

                # store each unit's s', terminated, new action
                unit_transition = {}
                unit_transition['state'] = local_observation
                unit_transition['terminated'] = False
                unit_transition['action'] = agent.one_hot_action(action_id, parameters.action_dim)
                cur_alive_friends_info[env.friends_tag_2_id[friend[0]]] = unit_transition
            if not env.is_first():
                # Notice: there are 2 ways ending a transition: (1) game is ended; (2) the unit is dead.
                agent.store_transition(pre_alive_friends_info, pre_reward, cur_alive_friends_info)

            # execute actions and receive reward.
            state, reward, done = env.step(action_queue)
            print('reward: ', reward)

            if done: # game ended
                agent.store_transition(cur_alive_friends_info, reward, None, True)
            # now, current state finished (become old), and update new records for pre states
            pre_alive_friends_info = cur_alive_friends_info
            pre_reward = reward

            # do batch training
            if parameters.training:
                agent.batch_training()

        print("win: ", env.is_win())
        # time.sleep(2)