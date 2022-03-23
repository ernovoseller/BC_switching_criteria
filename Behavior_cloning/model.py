
"""
Neural network models.

The implementation is based on code from the ThriftyDAgger repo, see
https://github.com/ryanhoque/thriftydagger/blob/master/thrifty/algos/core.py
"""

import numpy as np

import torch
import torch.nn as nn


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit, device):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit
        self.device = device

    def forward(self, obs):
        # Return output from network scaled to action space limits
        return self.act_limit * self.pi(obs).to(self.device)

class CNNActor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        c, h, w = obs_dim
        
        if h == 224:
        
            self.model = nn.Sequential(
                nn.Conv2d(c, 24, 5, 2),
                nn.MaxPool3d(2),
                nn.ELU(),
                nn.Conv2d(12, 10, 5, 2),
                nn.ELU(),
                nn.Conv2d(10, 8, 5, 2),
                nn.ELU(),
                nn.Conv2d(8, 6, 3, 1),
                nn.ELU(),
                nn.Conv2d(6, 6, 3, 1),
                nn.ELU(),
                nn.Conv2d(6, 4, 3, 1),
                nn.ELU(),
                #nn.Dropout(0.5),
                nn.Flatten(),
                #nn.Linear(100, 100),  #VAINAVI: previously (64,100) erroring here on matrix mul
                #nn.ELU(),
                nn.Linear(100, 50),
                nn.ELU(),
                nn.Linear(50, 10),
                nn.ELU(),
                nn.Linear(10, act_dim),
                nn.Tanh() # squash to [-1,1]
            )
        elif h == 64:
        
            self.model = nn.Sequential(
                nn.Conv2d(c, 24, 5, 2),
                #nn.MaxPool3d(2),
                nn.ELU(),
                nn.Conv2d(24, 10, 5, 2),
                nn.ELU(),
                nn.Conv2d(10, 4, 5, 2),
                nn.ELU(),
                # nn.Conv2d(8, 6, 3, 1),
                # nn.ELU(),
                #nn.Conv2d(6, 6, 3, 1),
                #nn.ELU(),
                #nn.Conv2d(6, 4, 3, 1),
                #nn.ELU(),
                #nn.Dropout(0.5),
                nn.Flatten(),
                #nn.Linear(100, 100),  #VAINAVI: previously (64,100) erroring here on matrix mul
                #nn.ELU(),
                nn.Linear(100, 50),
                nn.ELU(),
                nn.Linear(50, 10),
                nn.ELU(),
                nn.Linear(10, act_dim),
                nn.Tanh() # squash to [-1,1]
            )
            
        else:
            raise Exception('Image dimension not supported.')
            
            
    def forward(self, obs):
        #obs = obs.permute(0, 3, 1, 2)
        return self.model(obs)

class EnsembleCNN(nn.Module):
    def __init__(self, obs_dim, action_space, device, num_nets=1):
        super().__init__()
        self.obs_dim = obs_dim
        act_dim = action_space.shape[0]
        self.act_limit = action_space.high[0]
        self.num_nets = num_nets
        self.device = device
        # build policy and value functions
        self.pis = [CNNActor(obs_dim, act_dim).to(device) for _ in range(num_nets)]

    def act(self, obs, i=-1):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if len(obs.shape) == 3:
            obs = torch.unsqueeze(obs, 0)
        with torch.no_grad():
            if i >= 0: # optionally, only use one of the nets.
                return self.pis[i](obs).cpu().numpy()
            vals = list()
            for pi in self.pis:
                vals.append(pi(obs).cpu().numpy())
            return np.mean(np.array(vals), axis=0).squeeze()

    def variance(self, obs, idxs = None, norm = False):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if len(obs.shape) == 3:
            obs = torch.unsqueeze(obs, 0)
        with torch.no_grad():
            vals = list()
            for pi in self.pis:
                pred = pi(obs).cpu().numpy()
                if not idxs is None:
                    pred = pred[:, idxs]
                if norm:
                    pred = pred / np.linalg.norm(pred, axis = 1)[:, np.newaxis]
                vals.append(pred)
                
            return np.square(np.std(np.array(vals), axis=0)).mean()


