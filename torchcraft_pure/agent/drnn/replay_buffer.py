"""
Implementation of FLAGS - Deep Deterministic Policy Gradient - on gym-torcs.
with tensorflow.

ddpg paper:
    http://arxiv.org/pdf/1509.02971v2.pdf

Author: kenneth yu
"""
import tensorflow as tf
import numpy as np
from collections import deque, namedtuple
import copy
import os
import glob
import pickle as pkl
import time

FLAGS = tf.app.flags.FLAGS  # alias

transition_fields = ['state',
                     'sequence_length',
                     'action_others',
                     'action',
                     'reward',
                     'next_state',
                     'next_sequence_length',
                     'next_state_others',
                     'next_others_sequence_lengths',
                     'terminated',
                     'only_self_alive']
Transition = namedtuple('Transition', transition_fields)


def construct_transition(state, sequence_length, action_others, action, reward, next_state, next_sequence_length,
                         next_state_others, next_others_sequence_lengths, terminated, only_self_alive):
    # TODO: 注意 model index 是不对齐的（维度不一致）
    transition = Transition(state=state, sequence_length=sequence_length, action_others=action_others, action=action,
                            reward=reward, next_state=next_state, next_sequence_length=next_sequence_length,
                            next_state_others=next_state_others, next_others_sequence_lengths=next_others_sequence_lengths,
                            terminated=terminated, only_self_alive=only_self_alive)
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
        return len(self.buffer)

    @property
    def size(self):
        return self.buffer.__sizeof__()

    def sample_batch(self, batch_size):
        # minibatch = random.sample(self.buffer, batch_size)
        indices = np.random.permutation(self.length - 1)[:batch_size]
        fr_states, em_state, fr_seq_len, em_seq_len, ac_others = [], [], [], [], []
        ac, reward = [], []
        nxt_fr_states, nxt_em_states, nxt_fr_sequence_len, nxt_em_sequence_len = [], [], [], []
        nxt_oth_fr_states, nxt_oth_em_states, nxt_oth_fr_seq_len, nxt_oth_em_seq_len = [], [], [], []
        terminated_batch = []

        for idx in indices:
            trans = self.buffer[idx]
            fr_states.append(trans.state[0])
            em_state.append(trans.state[1])
            fr_seq_len.append(trans.sequence_length[0])
            em_seq_len.append(trans.sequence_length[1])

            ac_others.append(trans.action_others)
            ac.append(trans.action)
            reward.append([trans.reward])

            nxt_fr_states.append(trans.next_state[0])
            nxt_em_states.append(trans.next_state[1])
            nxt_fr_sequence_len.append(trans.next_sequence_length[0])
            nxt_em_sequence_len.append(trans.next_sequence_length[1])

            if trans.next_state_others is None:
                nxt_oth_fr_states.append(None)
                nxt_oth_em_states.append(None)
                nxt_oth_fr_seq_len.append(0)
                nxt_oth_em_seq_len.append(0)
            else:
                nxt_fr_s = [item[0] for item in trans.next_state_others]
                nxt_em_s= [item[1] for item in trans.next_state_others]
                nxt_fr_seq = [item[0] for item in trans.next_others_sequence_lengths]
                nxt_em_seq = [item[1] for item in trans.next_others_sequence_lengths]
                nxt_oth_fr_states.append(nxt_fr_s)
                nxt_oth_em_states.append(nxt_em_s)
                nxt_oth_fr_seq_len.append(nxt_fr_seq)
                nxt_oth_em_seq_len.append(nxt_em_seq)

            terminated_batch.append([trans.terminated])

        return (fr_states, em_state, fr_seq_len, em_seq_len, ac_others,
                ac, reward,
                nxt_fr_states, nxt_em_states, nxt_fr_sequence_len, nxt_em_sequence_len,
                nxt_oth_fr_states, nxt_oth_em_states, nxt_oth_fr_seq_len, nxt_oth_em_seq_len,
                terminated_batch)
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
                                                         [FLAGS.replay_buffer_file_name, str(self.save_segment_cnt),
                                                          time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
                                                          '.buffer'])))

        with open(abs_file_name, 'wb') as f:
            pkl.dump(data, f)

    def load(self, path):
        # load from file to buffer
        abs_file_pattern = os.path.abspath(os.path.join(path,
                                                        '_'.join([FLAGS.replay_buffer_file_name, '*'])))
        buffer_files = glob.glob(abs_file_pattern)
        for f_name in buffer_files:
            with open(f_name, 'rb') as f:
                data = pkl.load(f)
                self.buffer.extend(data)

    def clear(self):
        self.buffer.clear()
