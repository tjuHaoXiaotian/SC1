#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
    reset():
        return [local_observation] * n, [global_observation]

    step():
        return [obs_n, reward_n, done_n, info_n], [global_observation, global reward, done, info]
'''

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
        self.view_rate = 1.2

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
        self.ground_range = unit.groundRange  # 16

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
            self.is_dead = True
            self.just_dead = True
            self.state.pre_health = self.state.health
            self.state.health = 0
        else:  # already dead
            self.just_dead = False
            self.state.x = 0
            self.state.y = 0
            self.state.velocity_x = 0
            self.state.velocity_y = 0
            self.state.ground_cd = 0
            self.state.pre_health = self.state.health
            self.state.health = 0

    def is_enemy(self):
        return self.player_id == 1


class BaseEnv(gym.Env):

    def __init__(self, map='Maps/BroodWar/micro/m5v5_c_far.scm', battle_max_steps=50, goal_max_steps=3):
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
        self.goal_max_steps = goal_max_steps
        # init the server connection
        _ = self.__connect()

    def __connect(self):
        '''
        connect with the server
        :return:
        '''
        self.cl = tc.Client()
        dprint("connection start.....")
        connect_rt = self.cl.connect(self.default_config['hostname'], self.default_config['port'])
        dprint('conection rt: ', connect_rt)  # dprint True
        state = self.cl.init(micro_battles=True)
        for pid, player in state.player_info.items():
            dprint("player {} named {} is {}".format(player.id, player.name,
                                                     tc.Constants.races._dict[player.race]), level=2)
        dprint(state.map_size, level=1)
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
        '''
        close connection
        :return:
        '''
        if self.cl:
            _ = self.cl.close()
            dprint('close rt: ', _)

    def reset(self):
        '''
        reset battle
        :return: [local_observation] * n, [global_observation]
        '''
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
        self.friend_num = len(alive_friends)
        # init n agents
        self.agents = [Agent(id) for id in range(self.agent_num)]
        # init agents' record
        self.agent_tag_2_id = {unit.id: id for id, unit in enumerate(alive_agents)}
        # init agents' info
        player_set = set()
        type_set = set()
        for id, unit in enumerate(alive_agents):
            self.agents[id].init_info(unit=unit)
            player_set.add(unit.playerId)
            type_set.add(unit.type)
        self.player_num = len(player_set)
        self.type_num = len(type_set)
        # reset battle step to 0
        self.battle_step = 0
        self.reset_goal_step()

        obs_n = []
        for agent in self.agents:
            obs_n.append(self.make_observation(agent, self.agents))
        return obs_n, self.make_observation(None, all_agents=self.agents, global_view=True, one_hot=True)

    def step(self, actions):
        '''
        push the game one step forward
        :param actions:
        :return: [obs_n, reward_n, done_n, info_n], [global_observation, global reward, done, info]
        '''
        self.battle_step += 1
        # update goal step
        self.update_goal_step()
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
        info_n = {
            'alive_info': {},
            'killed_enemy_num': None
        }
        killed_enemies = [0] * self.agent_num
        for agent in self.agents:
            if not agent.is_dead:
                obs_n.append(self.make_observation(agent, self.agents))
                r, kill_opp_num = self.make_reward(self.agents, agent.is_enemy(), current_agent=agent, local_reward=True)
                reward_n.append(r)
                killed_enemies[agent.id] = kill_opp_num
                done_n.append(agent.is_dead)
                info_n['alive_info'][agent.id] = 'alive'
            else:
                if agent.just_dead:
                    # obs_n is no use (record the end state)
                    obs_n.append(self.make_observation(agent, self.agents))
                    r, kill_opp_num = self.make_reward(self.agents, agent.is_enemy(), current_agent=agent, local_reward=True)
                    reward_n.append(r)
                    killed_enemies[agent.id] = kill_opp_num
                    done_n.append(agent.is_dead)
                    info_n['alive_info'][agent.id] = 'just dead'
                else:  # already dead
                    obs_n.append(None)
                    reward_n.append(0)
                    killed_enemies[agent.id] = 0
                    done_n.append(agent.is_dead)
                    info_n['alive_info'][agent.id] = 'already dead'
        info_n['killed_enemy_num'] = killed_enemies

        # prepare global observations
        global_obs = self.make_observation(None, all_agents=self.agents, global_view=True, one_hot=True)
        global_reward, _ = self.make_reward(self.agents, opponent_view=False)
        done = self.is_end()
        info = {
            'killed_enemy_num': _
        }
        return [obs_n, reward_n, done_n, info_n], [global_obs, global_reward, done, info]

    def is_end(self):
        '''
        to judge whether the current battle is end. (end or over the max battle step)
        :return:
        '''
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

    def reset_goal_step(self):
        self.goal_step = 0

    def update_goal_step(self):
        self.goal_step += 1

    def is_goal_reached(self):
        return self.goal_step == self.goal_max_steps

    def print_units(self, state):
        alive_friends = state.units[0]
        alive_enemies = state.units[1]
        print(alive_friends)
        print(alive_enemies)
        print()

    @staticmethod
    def in_view(current_agent, other):
        '''
        To judge wheter unit:other is in current_agent's view.
        :param current_agent:
        :param other:
        :return:
        '''
        dis = math.sqrt((other.state.x - current_agent.state.x) ** 2
                        + (other.state.y - current_agent.state.y) ** 2)
        if dis <= current_agent.view_rate * current_agent.ground_range:
            return True
        else:
            return False

    '''
      Unimplemented methods
    '''
    # TODO: global settings, may be different for different maps
    MAX_DIS = 48.
    MAX_R_X = 45.
    MAX_R_Y = 45.
    MAX_V = 4.

    MAP_CENTER_X = 88  # the center x of the map
    MAP_MAX_RELATIVE_X = 40
    MAP_CENTER_Y = 134 # the center y of the map
    MAP_MAX_RELATIVE_Y = 22

    def make_observation(self, current_agent, all_agents, global_view=False, one_hot=True):
        '''
        :param current_agent:
        :param all_agents:
        :return: observation for current_agent or for current player(global)
        '''
        obs_n = []
        for other in all_agents:
            # TODO: later to adjust whether to skip the current agent (current included)
            # TODO: (need current agent's information included in its view)
            # if other is current_agent: continue  # skip the current agent

            if other.is_dead or (
                    not global_view and not self.in_view(current_agent, other)):  # dead or (local view and not in view)
                obs_n.append(
                    [
                        other.id,  # id
                        other.player_id,  # player_id
                        other.type,  # type
                        0,  # relative distance to (current agent or center of the map)
                        0,  # relative x to (current agent or center of the map)
                        0,  # relative y to (current agent or center of the map)
                        0,  # relative velocity x (current agent or 0)
                        0,  # relative velocity y (current agent or 0)
                        0,  # health
                        0  # ground_cd
                    ]
                )
            else:
                if global_view:
                    obs_n.append(
                        [
                            other.id,
                            other.player_id,
                            other.type,
                            math.sqrt((other.state.x - self.MAP_CENTER_X) ** 2
                                      + (other.state.y - self.MAP_CENTER_Y) ** 2) / self.MAX_DIS,
                            (other.state.x - self.MAP_CENTER_X) / self.MAP_MAX_RELATIVE_X,
                            (other.state.y - self.MAP_CENTER_Y) / self.MAP_MAX_RELATIVE_Y,
                            other.state.velocity_x / self.MAX_V,
                            other.state.velocity_y / self.MAX_V,
                            other.state.health / other.max_health,
                            other.state.ground_cd / other.max_cd
                        ]
                    )
                else:  # local view
                    obs_n.append(
                        [
                            other.id,
                            other.player_id,
                            other.type,
                            math.sqrt((other.state.x - current_agent.state.x) ** 2
                                      + (other.state.y - current_agent.state.y) ** 2) / self.MAX_DIS,
                            (other.state.x - current_agent.state.x) / self.MAX_R_X,
                            (other.state.y - current_agent.state.y) / self.MAX_R_Y,
                            (other.state.velocity_x - current_agent.state.velocity_x) / self.MAX_V,
                            (other.state.velocity_y - current_agent.state.velocity_y) / self.MAX_V,
                            other.state.health / other.max_health,
                            other.state.ground_cd / other.max_cd
                        ]
                    )

            # see the max values: (only used for coding)
            # if not other.is_enemy() and not other.is_dead:
                # relative_d = math.sqrt((other.state.x - self.MAP_CENTER_X) ** 2
                #                       + (other.state.y - self.MAP_CENTER_Y) ** 2) / self.MAX_DIS
                # relative_x = (other.state.x - self.MAP_CENTER_X) / self.MAP_MAX_RELATIVE_X
                # relative_y = (other.state.y - self.MAP_CENTER_Y) / self.MAP_MAX_RELATIVE_Y
                # print()
                # print('step {}: '.format(self.battle_step), other.id)
                # print("relative X: ", relative_x)
                # print("relative Y: ", relative_y)
                # print("relative dis: ", relative_d)

                # relative_d = math.sqrt((other.state.x - current_agent.state.x) ** 2
                #                       + (other.state.y - current_agent.state.y) ** 2) / self.MAX_DIS
                # relative_x = (other.state.x - current_agent.state.x) / self.MAX_R_X
                # relative_y = (other.state.y - current_agent.state.y) / self.MAX_R_Y
                # print()
                # print('step {}: '.format(self.battle_step), other.id)
                # print("relative X: ", relative_x)
                # print("relative Y: ", relative_y)
                # print("relative dis: ", relative_d)

                # tmp = math.sqrt((other.state.x - current_agent.state.x) ** 2
                #                   + (other.state.y - current_agent.state.y) ** 2)
                # self.MAX_DIS = tmp if tmp > self.MAX_DIS else self.MAX_DIS
                # tmp1 = other.state.x - current_agent.state.x
                # self.MAX_R_X = tmp1 if tmp1 > self.MAX_R_X else self.MAX_R_X
                # tmp2 = other.state.y - current_agent.state.y
                # self.MAX_R_Y = tmp2 if tmp2 > self.MAX_R_Y else self.MAX_R_Y
                # print(self.MAX_R_X, self.MAX_R_Y, self.MAX_DIS)

        if one_hot:
            obs_n = np.array(obs_n, dtype=np.float32)
            onehot_encoder = OneHotEncoder(sparse=False, n_values=[self.agent_num, self.player_num, self.type_num])
            category_pro = onehot_encoder.fit_transform(obs_n[:, [0, 1, 2]])
            tmp_2d = np.concatenate([category_pro, obs_n[:, 3:]], axis=1)
            # print(tmp_2d.shape)
            return np.concatenate(tmp_2d)
        else:
            return np.concatenate(obs_n)

    def make_reward(self, all_agents, opponent_view=True, current_agent=None, local_reward = False):
        '''
        :param player_view: 0 or 1 (from the point view of me or the enemy)
        :param all_agents:
        :return: [reward for current agent, killed enemy number of previous step]
        '''
        enemies_sum_delta_health = 0
        friends_sum_delta_health = 0
        friends_sum_health = 0
        enemies_sum_health = 0
        killed_enemy = 0
        killed_friend = 0

        for agent in all_agents:
            # if use local reward, try to judge whether current unit is in view. Otherwise, in_view is always true.
            if current_agent is not None and local_reward:
                in_view = self.in_view(current_agent, agent)
            else:
                in_view = True

            if in_view and agent.is_enemy():
                enemies_sum_delta_health += (agent.state.pre_health - agent.state.health)
                enemies_sum_health += agent.state.health
                if agent.just_dead:
                    killed_enemy += 1
            elif in_view and not agent.is_enemy():
                friends_sum_delta_health += (agent.state.pre_health - agent.state.health)
                friends_sum_health += agent.state.health
                if agent.just_dead:
                    killed_friend += 1

        if not opponent_view:  # in my view
            reward_part1 = enemies_sum_delta_health - friends_sum_delta_health * 0.5
            reward_part2 = killed_enemy * 10
            dprint('kill: ', killed_enemy)
            reward_part3 = 200 + friends_sum_health if self.is_end() and self.is_win() else 0
            kill_opp_num = killed_enemy
        else:  # in enemy's view
            reward_part1 = friends_sum_delta_health - enemies_sum_delta_health * 0.5
            reward_part2 = killed_friend * 10
            reward_part3 = 200 + enemies_sum_health if self.is_end() and not self.is_win() else 0
            kill_opp_num = killed_friend
        return reward_part1 + reward_part2 + reward_part3, kill_opp_num

    def init_action_observation_space(self):
        '''
        set the action space
        :return:
        '''
        self.action_space = spaces.Discrete(5 + 5)
        self.observation_space = spaces.Box(low=0, high=1., shape=[10 * 9], dtype=np.float32)

    @staticmethod
    def one_hot_action(action_id, action_dim):
        action = np.zeros([action_dim, ], dtype=np.int32)
        action[action_id] = 1
        return action

    def convert_discrete_action_2_sc1_action(self, agent_id, action_id, attack_target_id=None):
        '''
        convert [0-9] to bwapi action
        0: stop
        1-4: move
        > 5: attack id
        :param agent_id:
        :param action_id:
        :param attack_target_id: a specific target to attack
        :return:
        '''
        _SCREEN_MIN = [48, 112]
        _SCREEN_MAX = [128, 156]
        # _SCREEN_MAX = [100, 156]

        # select position 选中当前单位
        # location = [self.agents[agent_id].state.x, self.agents[agent_id].state.y]
        if action_id == 0:  # stop
            return [
                tcc.command_unit_protected,
                self.agents[agent_id].tag,
                tcc.unitcommandtypes.Stop,
            ]
        elif action_id <= 4:  # move
            if action_id == 1:  # move_up
                target_location = [self.agents[agent_id].state.x, _SCREEN_MIN[1]]
            elif action_id == 2:  # move_right
                target_location = [_SCREEN_MAX[0], self.agents[agent_id].state.y]
            elif action_id == 3:  # move_down
                target_location = [self.agents[agent_id].state.x, _SCREEN_MAX[1]]
            else:  # move_left
                target_location = [_SCREEN_MIN[0], self.agents[agent_id].state.y]

            return [
                tcc.command_unit_protected,
                self.agents[agent_id].tag,
                tcc.unitcommandtypes.Move,
                -1,
                target_location[0],
                target_location[1]
            ]
        else:
            if attack_target_id is None:
                target_id = action_id - 5 + self.friend_num
            else:
                target_id = attack_target_id + self.friend_num
            if not self.agents[target_id].is_dead:
                return [
                    tcc.command_unit_protected,
                    self.agents[agent_id].tag,
                    tcc.unitcommandtypes.Attack_Unit,
                    self.agents[target_id].tag,
                ]
            else:  # 目标单位已死亡
                return [
                    tcc.command_unit_protected,
                    self.agents[agent_id].tag,
                    getattr(tcc.unitcommandtypes, 'Stop')
                ]


import time

if __name__ == "__main__":
    env = BaseEnv()
    for ep in range(1000):
        # obs_n = env.reset()
        [obs_n, global_obs], done = env.reset(), False

        dprint(obs_n, level=1)
        dprint(global_obs, level=1)
        while not env.is_end():
            # action_queue = [env.convert_discrete_action_2_sc1_action(agent_id, 2) for agent_id in range(5)]
            # [obs_n, reward_n, done_n, info_n], [global_obs, global_reward, done, info] = env.step(action_queue)
            [obs_n, reward_n, done_n, info_n], [global_obs, global_reward, done, info] = env.step([])
            # dprint(obs_n, level=1)
            # dprint(obs_n[0].shape, level=1)
            dprint('step: ', env.battle_step, level=-1)
            dprint(reward_n, level=-1)
            dprint(global_reward, level=-1)
            dprint(done_n, level=-1)
            dprint(done, level=-1)
            print()
            dprint(info_n, level=1)

            time.sleep(0.5)
        dprint("win: ", env.is_win())
        dprint("loss: ", not env.is_win())
