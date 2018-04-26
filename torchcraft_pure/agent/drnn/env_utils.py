#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import math
from collections import namedtuple
import torchcraft.Constants as tcc
from sklearn.preprocessing import OneHotEncoder

_SCREEN_SIZE = [255, 255]
_SCREEN_MIN = [48, 112]
# _SCREEN_MAX = [128, 156]
_SCREEN_MAX = [100, 156]
_MAX_DX = _SCREEN_MAX[0] - _SCREEN_MIN[0]
_MAX_DY = _SCREEN_MAX[1] - _SCREEN_MIN[1]
_MAX_VELOCITY = [4, 4]
_MAX_HEALTH = 40
_MAX_CD = 15
# _MAX_DISTANCE = math.sqrt(_SCREEN_SIZE[0] ** 2 + _SCREEN_SIZE[1] ** 2)
_MAX_DISTANCE = math.sqrt(_MAX_DX ** 2 + _MAX_DY ** 2)
_MAX_GROUND_RANGE = 16

def _unit_vec_raw(u):
    return np.array(
        (
            u.id,
            u.type,
            u.playerId,
            u.x,
            u.y,
            # u.pixel_x,
            # u.pixel_y,
            u.velocityX,
            u.velocityY,
            u.health,
            u.max_health,
            u.groundCD,
            u.maxCD,
            u.groundRange,

            # u.shield,
            # u.shieldArmor,
            # u.max_shield
        ), dtype=np.int32
    )

def get_units_info(my_units, enemy_units):
    friends = list(_unit_vec_raw(u) for u in my_units)
    if friends and len(friends) > 0:
        friends = np.stack(friends)
    enemies = list(_unit_vec_raw(u) for u in enemy_units)
    if enemies and len(enemies) > 0:
        enemies = np.stack(enemies)
    return friends, enemies

def cal_local_observation_for_unit(current_unit, current_alive_friends, current_alive_enemies, friends_tag_2_id, enemies_tag_2_id):
    '''
    为 current_unit 准备局部观察信息
    [id, alive, unit_type, unit_owner, relative_distance, dx, dy, velocity_x, velocity_y, health, cool_down, ground_range]
      5  2  1 2 8  共 18 维
    :param current_unit:  当前 unit 信息
    :param current_alive_friends:  当前所有存活的己方单位
    :param current_alive_enemies:  当前所有存活的敌方单位
    :return: [ 己方单位由近及远（存活—> 死亡）, 敌方单位由近及远（存活—> 死亡）] 每个单位 11 维信息 * 13 个单位

    '''
    alive_friends_order = []
    all_friends = []
    all_enemies = []
    # all_alives = []
    # 拼接存活 己方单位
    for friend in current_alive_friends:
        tag = friend[0]
        id = friends_tag_2_id[tag]
        alive = 1
        unit_type = friend[1]
        unit_owner = friend[2]
        dx = friend[3] - current_unit[3]
        dy = friend[4] - current_unit[4]
        relative_distance = math.sqrt(dx ** 2 + dy ** 2) / _MAX_DISTANCE
        dx /= _MAX_DX
        dy /= _MAX_DY
        velocity_x = friend[5] / _MAX_VELOCITY[0]
        velocity_y = friend[6] / _MAX_VELOCITY[1]

        health = friend[7] / _MAX_HEALTH
        cool_down = friend[9] / _MAX_CD
        ground_range = friend[11] / _MAX_GROUND_RANGE

        relative_friend = [id, alive, unit_type, unit_owner, relative_distance, dx, dy, velocity_x, velocity_y, health, cool_down, ground_range]
        alive_friends_order.append(id)
        all_friends.append(relative_friend)
        # print('me unit_type: ',unit_type)


    # 拼接存活 敌方单位
    for enemy in current_alive_enemies:
        tag = enemy[0]
        id = enemies_tag_2_id[tag]
        alive = 1
        unit_type = enemy[1]
        unit_owner = enemy[2]
        dx = enemy[3] - current_unit[3]
        dy = enemy[4] - current_unit[4]
        relative_distance = math.sqrt(dx ** 2 + dy ** 2) / _MAX_DISTANCE
        dx /= _MAX_DX
        dy /= _MAX_DY
        velocity_x = enemy[5] / _MAX_VELOCITY[0]
        velocity_y = enemy[6] / _MAX_VELOCITY[1]

        health = enemy[7] / _MAX_HEALTH
        cool_down = enemy[9] / _MAX_CD
        ground_range = enemy[11] / _MAX_GROUND_RANGE

        relative_enemy = [id, alive, unit_type, unit_owner, relative_distance, dx, dy, velocity_x, velocity_y, health,
                           cool_down, ground_range]
        all_enemies.append(relative_enemy)
        # print('enemy unit_type: ',unit_type)


    all_alive_friends_count = len(all_friends)
    all_alive_enemies_count = len(all_enemies)

    # 按照相对距离从远到近排序
    all_enemies = sorted(all_enemies, key=lambda x: (x[4]), reverse = True)

    # 按照相对距离从远到近排序
    alive_friends_and_order = sorted(zip(alive_friends_order,all_friends), key=lambda x: (x[1][4]), reverse = True)
    alive_friends_order = [item[0] for item in alive_friends_and_order if item[0] != friends_tag_2_id[current_unit[0]]]  # 我不一定是最后一个，真的有可能是摞在一起的，所以在这里提前把自己剔除掉
    all_friends = [item[1] for item in alive_friends_and_order]

    for friend_dead in range(len(friends_tag_2_id) - len(current_alive_friends)):
        id = 0 #TODO: 当前是无所谓的
        alive = 0
        unit_type = 0
        unit_owner = 0
        dx = 0
        dy = 0
        relative_distance = 0
        velocity_x = 0
        velocity_y = 0
        health = 0
        cool_down = 0
        ground_range = 0
        relative_friend_dead = [id, alive, unit_type, unit_owner, relative_distance, dx, dy, velocity_x, velocity_y, health, cool_down, ground_range]
        all_friends.append(relative_friend_dead)

    for enemy_dead in range(len(enemies_tag_2_id) - len(current_alive_enemies)):
        id = 0  # TODO: 当前是无所谓的
        alive = 0
        unit_type = 0
        unit_owner = 1
        dx = 0
        dy = 0
        relative_distance = 0
        velocity_x = 0
        velocity_y = 0
        health = 0
        cool_down = 0
        ground_range = 0
        relative_enemy_dead = [id, alive, unit_type, unit_owner, relative_distance, dx, dy, velocity_x, velocity_y, health, cool_down, ground_range]
        all_enemies.append(relative_enemy_dead)

    tmp_friends = np.array(all_friends, dtype=np.float32)
    tmp_enemies = np.array(all_enemies, dtype=np.float32)

    # one-hot encoding
    enc = OneHotEncoder()
    # id, alive, unit_type, unit_owner,
    fit_template = []
    for id in range(len(friends_tag_2_id)):
        # fit_template.append([id, 0, 0, 0])
        # fit_template.append([id, 1, 0, 0])
        fit_template.append([id, 0, 65, 0])
        fit_template.append([id, 0, 66, 0])
        fit_template.append([id, 1, 65, 0])
        fit_template.append([id, 1, 66, 0])

    for id in range(len(enemies_tag_2_id)):
        # fit_template.append([id, 0, 0, 0])
        # fit_template.append([id, 1, 0, 0])
        fit_template.append([id, 0, 65, 1])
        fit_template.append([id, 0, 66, 1])
        fit_template.append([id, 1, 65, 1])
        fit_template.append([id, 1, 66, 1])

    enc.fit(fit_template)

    part1_friends = enc.transform(tmp_friends[:, [0, 1, 2, 3]]).toarray()
    part2_friends = tmp_friends[:, 4:]
    features_friends = np.hstack([part1_friends,part2_friends])
    # print(features_friends.shape)
    features_friends = np.hstack(features_friends)

    part1_enemies = enc.transform(tmp_enemies[:, [0, 1, 2, 3]]).toarray()
    part2_enemies = tmp_enemies[:, 4:]

    features_enemies = np.hstack([part1_enemies, part2_enemies])
    features_enemies =  np.hstack(features_enemies)

    # alive_friends_order 决定了 action other 的顺序  None if len(alive_friends_order) == 0 else
    return [features_friends, features_enemies], [all_alive_friends_count, all_alive_enemies_count], alive_friends_order


Action = namedtuple('Action', ['stop', #'noop',
                                'move_up', 'move_right', 'move_down', 'move_left'
                               'attack_0', 'attack_1', 'attack_2', 'attack_3'])


def convert_discrete_action_2_sc1_action(unit, action_id, alive_enemies, all_enemies_id_2_tag):
    '''
    0: stop
    1-4: move
    > 5: attack id
    :param unit:
    :param action_id:
    :param alive_enemies:
    :param all_enemies_id_2_tag:
    :return:
    '''
    actions_queue = []
    # select position 选中当前单位
    location = [unit[3], unit[4]]
    if action_id == 0: # stop
        actions_queue.append([
            tcc.command_unit_protected,
            unit[0],
            tcc.unitcommandtypes.Stop,
        ])
    # elif action_id == 1:  # noop
    #     actions_queue.append([
    #         tcc.noop,
    #         # unit[0],
    #         # getattr(tcc.unitcommandtypes,'None')
    #     ])
    #     # print(getattr(tcc.unitcommandtypes,'None'))
    elif action_id <= 4:  # move
        if action_id == 1: # move_up
            target_location = [unit[3], _SCREEN_MIN[1]]
        elif action_id == 2: # move_right
            target_location = [_SCREEN_MAX[0], unit[4]]
        elif action_id == 3: # move_down
            target_location = [unit[3], _SCREEN_MAX[1]]
        else:  # move_left
            target_location = [_SCREEN_MIN[0], unit[4]]

        actions_queue.append([
            tcc.command_unit_protected,
            unit[0],
            tcc.unitcommandtypes.Move,
            -1,
            target_location[0],
            target_location[1]
        ])
    else:
        alive_enemies = {enemy[0]: enemy for enemy in alive_enemies}
        target = None
        for id in all_enemies_id_2_tag:
            if action_id - 5 == id:
                target = alive_enemies.get(all_enemies_id_2_tag[id], None)
                break
        # if action_id == 5: # attack_0
        #     target = alive_enemies.get(all_enemies_id_2_tag[0], None)
        # elif action_id == 6: # attack_1
        #     target = alive_enemies.get(all_enemies_id_2_tag[1], None)
        # elif action_id == 7: # attack_2
        #     target = alive_enemies.get(all_enemies_id_2_tag[2], None)
        # elif action_id == 8: # attack_3
        #     target = alive_enemies.get(all_enemies_id_2_tag[3], None)
        # else:  # attack_4
        #     target = alive_enemies.get(all_enemies_id_2_tag[4], None)
        if target is not None:
            actions_queue.append([
                tcc.command_unit_protected,
                unit[0],
                tcc.unitcommandtypes.Attack_Unit,
                target[0],
            ])
        else: # 目标单位已死亡
            actions_queue.append([
                tcc.command_unit_protected,
                unit[0],
                getattr(tcc.unitcommandtypes, 'Stop')
            ])
            pass
    return actions_queue

def one_hot_action(action_id, action_dim):
    action = np.zeros([action_dim,],dtype=np.int32)
    action[action_id] = 1
    return action