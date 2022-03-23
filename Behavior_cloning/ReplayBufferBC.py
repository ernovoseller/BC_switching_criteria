# -*- coding: utf-8 -*-
"""
Replay buffer to use with behavior cloning.

The behavior cloning implementation is based on code from the ThriftyDAgger repo, see
https://github.com/ryanhoque/thriftydagger/blob/master/thrifty/algos/lazydagger.py
"""

import numpy as np
import torch
import pickle

from Behavior_cloning.model import combined_shape


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer.
    """
    def __init__(self, obs_dim, act_dim, size, device):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32) 
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act 
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32, visible_idxs = []):
        """
        Limiting which data indices are visible--via the visible_idxs parameter--
        is useful when making sure only a bootstrapped sample of the data is
        visible during each training epoch.
        """
        if visible_idxs == []:
            idxs = np.random.choice(np.arange(self.size), size=batch_size, replace = False)
        else:
            idxs = np.random.choice(visible_idxs, size = batch_size, replace = False)
        
        batch = dict(obs=self.obs_buf[idxs],
                act=self.act_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k,v in batch.items()}

    def fill_buffer(self, obs, act):
        for i in range(len(obs)):
            self.store(obs[i], act[i])

    def save_buffer(self, name='replay'):
        pickle.dump({'obs_buf': self.obs_buf, 'act_buf': self.act_buf,
            'ptr': self.ptr, 'size': self.size}, open('{}_buffer.pkl'.format(name), 'wb'))
        print('buf size', self.size)

    def load_buffer(self, name='replay_buffer.pkl'):
        # device does not get transferred since it differs
        p = pickle.load(open('{}'.format(name), 'rb'))
        self.obs_buf = p['obs_buf']
        self.act_buf = p['act_buf']
        self.ptr = p['ptr']
        self.size = p['size']

    def clear(self):
        self.ptr, self.size = 0, 0

