'''
The behavior cloning implementation is based on code from the ThriftyDAgger repo,
https://github.com/ryanhoque/thriftydagger/blob/master/thrifty/algos/lazydagger.py
'''
# -*- coding: utf-8 -*-
import argparse
import datetime
import pickle
import numpy as np
import torch
from torch.optim import Adam
import os
from gym import spaces

from Behavior_cloning.ReplayBufferBC import ReplayBuffer
from Behavior_cloning.model import EnsembleCNN



def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


class GymClothEnv():
    """
    If loading all demonstrations in advance, this is useful in order not to
    instantiate a full GymCloth environment.
    """
    def __init__(self, RGB):

        self.action_space = spaces.Box(-np.ones(4) * 1.1,
                                np.ones(4) * 1.1)
        if RGB:
            self.observation_space = spaces.Box(-1, 1, (224, 224, 3))
        else:
            self.observation_space = spaces.Box(-1, 1, (224, 224, 1))


def get_demo_data(demos_count, demo_file = None, demo_folder = None):
    """
    Loads/returns data from a particular demonstration.
    """
    
    if demo_file is not None:
        
        return demo_file[demos_count]['obs'][:-1], demo_file[demos_count]['act']
     
    elif demo_folder is not None:
    
        demo_filepath = os.path.join(demo_folder, 'Eps_' + str(demos_count) + '.pkl')    
    
        data = pickle.load(open(demo_filepath, 'rb'))
        
        return data['obs'], data['act']
 

def main():

    """Parse all input arguments."""
    parser = argparse.ArgumentParser(description='Arguments for running SAC in GymCloth.')

    parser.add_argument('--env_name', type = str, default='ClothFlatten')
    parser.add_argument('--logdir', default="bc_runs", help='exterior log directory')
    parser.add_argument('--logdir_suffix', default="", help='log directory suffix')

    parser.add_argument('--seed',type=int,default=123456,metavar='N',help='random seed (default: 123456)')
    parser.add_argument('--cuda', action="store_false", help='run on CUDA (default: True)')

    parser.add_argument('--RGB', action = "store_true", help = 'Default = False (use grayscale)')

    parser.add_argument('--augment', action = "store_true", help = 'Default = False')

    # BC args    
    parser.add_argument('--ensemble_size', type = int, default = 1, 
                        help = 'Number of networks in BC ensemble')

    parser.add_argument('--replay_size', type=int, default=100000, metavar='N', help='size of replay buffer (default: 1000000)')
    parser.add_argument('--lr', type=float, default=0.00025, metavar='G',
                        help='learning rate (default: 2.5e-4)')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='batch size (default: 64)')
    parser.add_argument('--grad_steps', type=int, default=400, metavar='N',
                        help='number of gradient updates per model update (default: 400)')

    parser.add_argument('--eps_per_iter', type = int, default = 10,
                        metavar = 'N', help = 'Number of episodes to add to buffer each iteration.')

    parser.add_argument('--weight_decay', type=float, default = 1e-5,
                        help = 'weight decay to add to BC loss.')

    args = parser.parse_args()
    
    
    """Set up log directories."""
    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d/%H-%M-%S")

    if args.logdir_suffix == '':
        logdir_base = os.path.join(args.logdir, date_string)
    else:
        logdir_base = os.path.join(args.logdir, date_string + '-' + args.logdir_suffix)

    if not os.path.exists(logdir_base):
        os.makedirs(logdir_base)

    pickle.dump(args, open(os.path.join(logdir_base, "args.pkl"), "wb"))

    if args.cuda:
        device = torch.device("cuda")
        print("using GPU")
    else:
        device = torch.device("cpu")
        print("using CPU")

    # For saving BC policy checkpoints:
    logdir_policy_info = os.path.join(logdir_base, 'Policy_info')

    if not os.path.exists(logdir_policy_info):
        os.makedirs(logdir_policy_info)

    set_seed(args.seed)

    RGB = args.RGB
    
    env  = GymClothEnv(RGB)

    obs_dim = list(env.observation_space.shape)
    obs_dim = np.roll(obs_dim, 1)    # Move number of channels to 0th position

    act_dim = list(env.action_space.shape)
    
    
    grad_steps = args.grad_steps       # Minibatches/gradient steps per model update phase
    eps_per_iter = args.eps_per_iter   # Number of demos to add to replay buffer per iteration
    batch_size = args.batch_size
    pi_lr = args.lr
    
    # Set up replay buffer:
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, 
                                 size=args.replay_size, device=device)
    
    # Instantiate behavior cloning model:
    bc_model = EnsembleCNN(obs_dim, env.action_space, device, 
                            num_nets = args.ensemble_size)

    # Set up optimizers
    pi_optimizers = [Adam(bc_model.pis[i].parameters(), lr=pi_lr,
                          weight_decay = args.weight_decay) for i in range(bc_model.num_nets)]

    # Set up function for computing policy loss
    def compute_loss_pi(data, i):
        
        o, a = data['obs'], data['act']

        a_pred = bc_model.pis[i](o)
        return torch.mean(torch.sum((a - a_pred)**2, dim=1))

    # Function for updating the policy during model training
    def update_pi(data, i):
        # run one gradient descent step for pi.
        pi_optimizers[i].zero_grad()
        loss_pi = compute_loss_pi(data, i)
        loss_pi.backward()
        pi_optimizers[i].step()
        return loss_pi.item()


    iter_count = 0   # For keeping track of when to save policies
    
    # Metrics (currently just storing training losses):
    loss_pi = []
    
    if not args.augment:
        demo_filepath = 'BC_data/data_25x25_match_real_norm.pkl'
        data = pickle.load(open(demo_filepath, 'rb'))
    else:
        # Currently, not adding this to the Github repo due to large filesize,
        # but can generate with augment_images.py.
        demo_folder = 'BC_data/data_25x25_match_real_norm_aug_flips_10'
    
    demos_count = 0    # Keeps track of how many demos added to replay buffer
    
    while demos_count < len(data):
        
        iter_count += 1    # For policy checkpoint filenames
        
        # Add new episodes to the replay buffer.
        for i in range(eps_per_iter):
            
            if i >= len(data):
                break
            
            if not args.augment:
                observations, actions = get_demo_data(demos_count, demo_file = data)
            else:
                observations, actions = get_demo_data(demos_count, demo_folder = demo_folder)
                
            
            for obs, action in zip(observations, actions):
                    
                # Append transition to replay buffer
                if not RGB:
                    obs = obs[:, :, 0]   
                    obs = np.expand_dims(obs, axis = -1)
                
                obs = np.transpose(obs, (2, 0, 1))
                obs = np.ascontiguousarray(obs)
                
                replay_buffer.store(obs, action)  
            
            print('Finished loading demo ' + str(i + 1))            
            
            demos_count += 1
            
        # Model update phase.
        if replay_buffer.size < batch_size:
            continue
        
        print('Updating BC model: iter %i, %i demos in buffer' % (iter_count, demos_count))

        # Sample indices of data that each ensemble member will be allowed to see during training.
        idxs = np.random.randint(replay_buffer.size, size = (bc_model.num_nets, replay_buffer.size))         
    
        loss_pi_ens = np.empty(bc_model.num_nets)
    
        for i in range(bc_model.num_nets):
            
            for _ in range(grad_steps):
                batch = replay_buffer.sample_batch(batch_size, visible_idxs = idxs[i])
                loss_pi_ = update_pi(batch, i)
            
            print(loss_pi_)
            
            loss_pi_ens[i] = loss_pi_
            #print('LossPi', loss_pi_)
            
        loss_pi.append(loss_pi_ens)

        # Save policy checkpoint:
        torch.save(bc_model, os.path.join(logdir_policy_info, 'Model_epoch_' \
                                          + str(iter_count) + '.pth'))    
   
        # Save metrics into an output file:  
        metrics_data = {'loss_pi': loss_pi}
        pickle.dump(metrics_data, open(os.path.join(logdir_base, 'Metrics.pkl'), 'wb'))




if __name__ == '__main__':
    main()
