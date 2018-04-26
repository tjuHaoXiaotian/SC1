#!/usr/bin/python
# -*- coding: UTF-8 -*-

import gym
import torchcraft as tc
import torchcraft.Constants as tcc
import math
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from gym import spaces

DEBUG = 1


def dprint(*msg, level=1):
    if DEBUG > level:
        print(*msg)


class State(object):
    def __init__(self):
        # properties may be changed with time
        self.x = None
        self.y = None
        self.velocity_x = None
        self.velocity_y = None
        self.pre_health = None  # pre step's health value
        self.health = None
        self.ground_cd = None


class Agent(object):
    def __init__(self, id):
        '''
           init agent properties
        '''
        # properties will not be changed
        self.id = id  # id we assigned to trace each agent
        self.tag = None  # game inner unique id
        self.type = None
        self.player_id = None  # 0: friend, 1: enemy
        self.max_cd = None
        self.max_health = None
        self.ground_range = None

        # properties may be changed with time
        self.is_dead = None
        self.just_dead = None
        self.state = State()

    def init_info(self, unit):
        self.tag = unit.id
        self.type = unit.type
        self.player_id = unit.playerId
        self.max_cd = unit.maxCD
        self.max_health = unit.max_health
        self.ground_range = unit.groundRange

        self.is_dead = False
        self.just_dead = False
        self.state.x = unit.x
        self.state.y = unit.y
        self.state.velocity_x = unit.velocityX
        self.state.velocity_y = unit.velocityY
        self.state.health = unit.health
        self.state.ground_cd = unit.groundCD

    def update_state_alive(self, unit):
        self.state.x = unit.x
        self.state.y = unit.y
        self.state.velocity_x = unit.velocityX
        self.state.velocity_y = unit.velocityY
        self.state.pre_health = self.state.health
        self.state.health = unit.health
        self.state.ground_cd = unit.groundCD

    def update_state_dead(self):
        if not self.is_dead:  # not dead pre step
            self.just_dead = True
        else:
            self.just_dead = False
        self.is_dead = True
        self.state.x = 0
        self.state.y = 0
        self.state.velocity_x = 0
        self.state.velocity_y = 0
        self.state.pre_health = self.state.health
        self.state.health = 0
        self.state.ground_cd = 0

    def is_enemy(self):
        return self.player_id == 1


class BaseEnv(gym.Env):

    def __init__(self, map='Maps/BroodWar/micro/m5v5_c_far.scm', battle_max_steps=200):
        # init default config
        self.default_config = {
            "hostname": "127.0.0.1",
            "port": 11111,
            "map": map,
            "skip_frames": 7,
            "set_speed": 0,
        }
        self.init_action_observation_space()
        # set battle max step
        self.battle_max_steps = battle_max_steps
        # init the server connection
        _ = self.__connect()

    def __connect(self):
        self.cl = tc.Client()
        dprint("connection start.....")
        connect_rt = self.cl.connect(self.default_config['hostname'], self.default_config['port'])
        dprint('conection rt: ', connect_rt)  # dprint True
        state = self.cl.init(micro_battles=True)
        for pid, player in state.player_info.items():
            dprint("player {} named {} is {}".format(player.id, player.name,
                                                     tc.Constants.races._dict[player.race]), level=2)

        # Initial setup the game
        _ = self.cl.send([
            [tcc.set_combine_frames, self.default_config['skip_frames']],
            [tcc.set_speed, self.default_config['set_speed']],
            [tcc.set_gui, 1],
            [tcc.set_cmd_optim, 1],
        ])
        # dprint('init set up: ', _)  #print True
        dprint("connection ended.....")
        return state

    def __dis_connect(self):
        if self.cl:
            _ = self.cl.close()
            dprint('close rt: ', _)

    def reset(self):
        state = self.cl.recv()
        if state.battle_frame_count > 0:  # part begin
            dprint("try restart......")
            restart_rt = self.cl.send([[tcc.restart]])
            dprint('restart rt: ', restart_rt)
            state = self.cl.recv()
            dprint('after sending restart command, game_ended is ', state.game_ended)
            dprint("try restart ended......")
            if state.game_ended:
                self.__dis_connect()
                state = self.__connect()
        else:
            dprint("fully new start, no need to restart......")

        while state.waiting_for_restart:
            state = self.cl.recv()

        # receiving the correct state information.
        self.inner_state = state
        # init agents information
        dprint('battle_frame_count: ', state.battle_frame_count)
        alive_friends = state.units[0]
        alive_enemies = state.units[1]
        alive_agents = alive_friends + alive_enemies
        # the total number of agents
        self.agent_num = len(alive_agents)
        # init n agents
        self.agents = [Agent(id) for id in range(self.agent_num)]
        # init agents' record
        self.agent_tag_2_id = {unit.id: id for id, unit in enumerate(alive_agents)}
        # init agents' info
        for id, unit in enumerate(alive_agents):
            self.agents[id].init_info(unit=unit)
        # reset battle step to 0
        self.battle_step = 0

        obs_n = []
        for agent in self.agents:
            obs_n.append(self.make_observation(agent, self.agents))
        return obs_n

    def step(self, actions):
        self.battle_step += 1
        # execute actions
        self.cl.send(actions)

        # update state to next
        state = self.cl.recv()
        self.inner_state = state

        # update all agents' state
        alive_friends = state.units[0]
        alive_enemies = state.units[1]
        alive_agents = alive_friends + alive_enemies
        dead_agents = set(range(self.agent_num))
        for alive_unit in alive_agents:
            self.agents[self.agent_tag_2_id[alive_unit.id]].update_state_alive(alive_unit)
            dead_agents.remove(self.agent_tag_2_id[alive_unit.id])

        for dead_id in dead_agents:
            self.agents[dead_id].update_state_dead()

        obs_n, reward_n, done_n = [], [], []
        info_n = {}
        for agent in self.agents:
            if not agent.is_dead:
                obs_n.append(self.make_observation(agent, self.agents))
                reward_n.append(self.make_reward(agent, self.agents))
                done_n.append(agent.is_dead)
                info_n[agent.id] = 'alive'
            else:
                if agent.just_dead:
                    obs_n.append(self.make_observation(agent, self.agents))
                    reward_n.append(self.make_reward(agent, self.agents))
                    done_n.append(agent.is_dead)
                    info_n[agent.id] = 'just dead'
                else:  # already dead
                    obs_n.append(None)
                    reward_n.append(None)
                    done_n.append(agent.is_dead)
                    info_n[agent.id] = 'already dead'
        return obs_n, reward_n, done_n, info_n

    def is_end(self):
        win_or_loss = self.inner_state.battle_just_ended
        over_step = self.battle_step > self.battle_max_steps
        if win_or_loss:
            dprint("win or loss.")
        if over_step:
            dprint("too much steps.")
        return win_or_loss or over_step

    def is_first(self):
        return self.battle_step == 0

    def is_win(self):
        return self.inner_state.battle_won

    def print_units(self, state):
        alive_friends = state.units[0]
        alive_enemies = state.units[1]
        print(alive_friends)
        print(alive_enemies)
        print()

    '''
      Unimplemented methods
    '''

    def make_observation(self, current_agent, all_agents, one_hot=True):
        '''
        :param current_agent:
        :param all_agents:
        :return: observation for current_agent
        '''
        obs_n = []
        for other in all_agents:
            if other is current_agent: continue  # skip the current agent
            if other.is_dead:
                obs_n.append(
                    [
                        other.player_id,  # player_id
                        other.type,  # type
                        0,  # distance
                        0,  # relative x
                        0,  # relative y
                        0,  # relative velocity x
                        0,  # relative velocity y
                        0,  # health
                        0  # ground_cd
                    ]
                )
            else:
                obs_n.append(
                    [
                        other.player_id,
                        other.type,
                        math.sqrt((other.state.x - current_agent.state.x) ** 2
                                  + (other.state.y - current_agent.state.y) ** 2),
                        other.state.x - current_agent.state.x,
                        other.state.y - current_agent.state.y,
                        other.state.velocity_x - current_agent.state.velocity_x,
                        other.state.velocity_y - current_agent.state.velocity_y,
                        other.state.health,
                        other.state.ground_cd
                    ]
                )
        if one_hot:
            obs_n = np.array(obs_n, dtype=np.float32)
            onehot_encoder = OneHotEncoder(sparse=False)
            category_pro = onehot_encoder.fit_transform(obs_n[:, [0, 1]])
            return np.concatenate(np.concatenate([category_pro, obs_n[:, 2:]], axis=1))
        else:
            return np.concatenate(obs_n)

    def make_reward(self, current_agent, all_agents):
        reward_n = []
        enemies_sum_delta_health = 0
        friends_sum_delta_health = 0
        friends_sum_health = 0
        enemies_sum_health = 0
        killed_enemy = 0
        killed_friend = 0
        inverse = False
        if current_agent.is_enemy():
            inverse = True
        for agent in self.agents:
            if agent.is_enemy():
                enemies_sum_delta_health += (agent.state.pre_health - agent.state.health)
                enemies_sum_health += agent.state.health
                if agent.just_dead:
                    killed_enemy += 1
            else:
                friends_sum_delta_health += (agent.state.pre_health - agent.state.health)
                friends_sum_health += agent.state.health
                if agent.just_dead:
                    killed_friend += 1

        if not inverse: # in my view
            reward_part1 = enemies_sum_delta_health - friends_sum_delta_health * 0.5
            reward_part2 = killed_enemy * 10
            dprint('kill: ', killed_enemy)
            reward_part3 = 200 + friends_sum_health if self.is_end() and self.is_win() else 0
        else: # in enemy's view
            reward_part1 = friends_sum_delta_health - enemies_sum_delta_health * 0.5
            reward_part2 = killed_friend * 10
            reward_part3 = 200 + enemies_sum_health if self.is_end() and not self.is_win() else 0
        return reward_part1 + reward_part2 + reward_part3

    def init_action_observation_space(self):
        '''
        set the action space
        :return:
        '''
        self.action_space = spaces.Discrete(5+5)
        self.observation_space = spaces.Box(low=0, high=1., shape=[10 * 9], dtype=np.float32)


import time

if __name__ == "__main__":
    env = BaseEnv()
    for ep in range(1000):
        obs_n = env.reset()
        # dprint(obs_n, level=1)
        while not env.is_end():
            obs_n, reward_n, done_n, info_n = env.step([])
            # dprint(obs_n, level=1)
            # dprint(obs_n[0].shape, level=1)
            dprint(reward_n, level=1)
            dprint(done_n, level=1)
            dprint(info_n, level=1)

            time.sleep(0.5)
        dprint("win: ", env.is_win())
        dprint("loss: ", not env.is_win())
