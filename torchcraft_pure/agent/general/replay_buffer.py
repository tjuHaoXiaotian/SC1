"""
Implementation of parameters - Deep Deterministic Policy Gradient - on gym-torcs.
with tensorflow.

ddpg paper:
    http://arxiv.org/pdf/1509.02971v2.pdf

Author: kenneth yu
"""
import numpy as np
from collections import deque, namedtuple
import copy
import os
import glob
import pickle as pkl
import time
from torchcraft_pure.agent.general.parameters import parameters

transition_fields = ['state', # s
                     'action_others', # a_others
                     'action', # a
                     'reward', # r
                     'next_state', # s'
                     'next_state_others', # a_others_s'
                     'terminated'
                     ]
Transition = namedtuple('Transition', transition_fields)


def construct_transition(state, action_others, action, reward, next_state,
                         next_state_others, terminated):
    transition = Transition(state=state, action_others=action_others, action=action,
                            reward=reward, next_state=next_state,
                            next_state_others=next_state_others,
                            terminated=terminated)
    return transition


class ReplayBuffer(object):
    def __init__(self, buffer_size, seed, save_segment_size=None, save_path=None):
        # The right side of the deque contains the most recent experiences
        self.buffer_size = buffer_size
        self.buffer = deque([], maxlen=buffer_size)
        if seed is not None:
            np.random.seed(seed)
        self.save = False
        if save_segment_size is not None:
            assert save_path is not None
            self.save = True
            self.save_segment_size = save_segment_size
            self.save_path = save_path
            self.save_data_cnt = 0
            self.save_segment_cnt = 0

    def store(self, transition):
        # deque can take care of max len.
        T = copy.deepcopy(transition)
        self.buffer.append(T)
        if self.save:
            self.save_data_cnt += 1
            if self.save_data_cnt >= self.save_segment_size:
                self.save_segment()
                self.save_data_cnt = 0
        del transition

    def get_item(self, idx):
        return self.buffer[idx]

    @property
    def length(self):
        '''
        element number
        :return:
        '''
        return len(self.buffer)

    @property
    def size(self):
        return self.buffer.__sizeof__()

    def sample_batch(self, batch_size):
        # minibatch = random.sample(self.buffer, batch_size)
        indices = np.random.permutation(self.length - 1)[:batch_size]
        s, a_others, a_me, r, s_, s_others,terminated = [], [], [], [], [], [], []
        s_others_ids = []
        for idx in indices:
            trans = self.buffer[idx]
            s.append(trans.state)
            a_others.append(trans.action_others)
            a_me.append(trans.action)
            r.append([trans.reward])
            s_.append(trans.next_state)
            if trans.next_state_others is None:
                s_others.append(None)
                s_others_ids.append(None)
            else:
                ids = []
                s_os = []
                for id, s_o in trans.next_state_others.items():
                    s_os.append(s_o)
                    ids.append(id)
                s_others_ids.append(ids)
                s_others.append(s_os)

            terminated.append([trans.terminated])

        return (s, a_others, a_me, r, s_, s_others, s_others_ids,terminated)
        # return (np.array(state_batch, dtype=np.float32),
        #         np.array(action_batch, dtype=np.float32),
        #         np.array(reward_batch, dtype=np.float32),
        #         np.array(next_state_batch, dtype=np.float32),
        #         np.array(terminated_batch,dtype=np.bool))

    def save_segment(self):
        self.save_segment_cnt += 1

        data = []
        start = self.length - self.save_segment_size  # always save latest data of segment_size
        end = self.length

        for idx in range(start, end):
            data.append(self.buffer[idx])

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        abs_file_name = os.path.abspath(os.path.join(self.save_path,
                                                     '_'.join(
                                                         [parameters.replay_buffer_file_name, str(self.save_segment_cnt),
                                                          time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
                                                          '.buffer'])))

        with open(abs_file_name, 'wb') as f:
            pkl.dump(data, f)

    def load(self, path):
        # load from file to buffer
        abs_file_pattern = os.path.abspath(os.path.join(path,
                                                        '_'.join([parameters.replay_buffer_file_name, '*'])))
        buffer_files = glob.glob(abs_file_pattern)
        for f_name in buffer_files:
            with open(f_name, 'rb') as f:
                data = pkl.load(f)
                self.buffer.extend(data)

    def clear(self):
        self.buffer.clear()
