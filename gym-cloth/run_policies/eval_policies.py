"""Use this for our analysis scripts (so it uses `demo_baselines.yaml`).

Use `demo_spaces.py` for other debugging.
"""
import subprocess
import pkg_resources
import numpy as np
import torch
import argparse
import os
from os.path import join
import sys
import time
import yaml
import logging
import pickle
import datetime
import cv2
import matplotlib as mpl
from gym_cloth.envs.cloth_env import ClothEnv
from collections import defaultdict

from analytic import Policy

analysis_path = os.path.dirname(os.path.abspath(__file__)) + "/../../"
sys.path.insert(0, analysis_path)
from analysis.color_change_data import tint_data, augment_obs
np.set_printoptions(edgeitems=10, linewidth=180, suppress=True)




class CornerPullingBCPolicy(Policy):

    def __init__(self, policy_path):
        super().__init__()
        device = None
        device_idx = 0
        if device_idx >= 0:
            device = torch.device("cuda")   #, device_idx)
        else:
            device = torch.device("cpu")
 
        behavior_cloning_path = os.path.dirname(os.path.abspath(__file__)) + "/../../"
        sys.path.insert(0, behavior_cloning_path)

        self.model = torch.load(policy_path, map_location=device)


    def get_action(self, obs, t, tint = True, augment = False, mode = 'eval'):
        
        if tint:
        
            obs = tint_data([obs], norm = False)[0]

        if augment:
            
            obs = augment_obs(obs, mode = mode)

        if tint:
            obs = obs[:, :, 0]
            obs = np.expand_dims(obs, axis = -1) 
            
        obs = np.transpose(obs, (2, 0, 1))
        obs = np.ascontiguousarray(obs)

        action = self.model.act(obs)
        return action



def run(args, policy, env, mode = 'eval'):
    """Run an analytic policy, using similar setups as baselines-fork.

    If we have a random seed in the args, we use that instead of the config
    file. That way we can run several instances of the policy in parallel for
    faster data collection.
    """

    # Book-keeping.
    num_episodes = 0
    stats_all = []
    coverage = []
    variance_inv = []
    nb_steps = []
    
    policy.set_env_cfg(env, cfg)

    obs = env.reset()
    # Go through one episode and put information in `stats_ep`.
    # Don't forget the first obs, since we need t _and_ t+1.
    stats_ep = defaultdict(list)
    stats_ep['obs'].append(obs)
    done = False
    num_steps = 0

    while not done:
        action = policy.get_action(obs, t=num_steps, mode = mode)
        # actions.append(action)
        obs, rew, done, info = env.step(action)
        
        stats_ep['obs'].append(obs) 
        stats_ep['rew'].append(rew)
        stats_ep['act'].append(action)
        stats_ep['done'].append(done)
        stats_ep['info'].append(info)
        num_steps += 1
    num_episodes += 1
    coverage.append(info['actual_coverage'])
    variance_inv.append(info['variance_inv'])
    nb_steps.append(num_steps)
    stats_all.append(stats_ep)
    print("\nInfo for most recent episode: {}".format(info))
    print("Finished {} episodes.".format(num_episodes))
    print('  {:.2f} +/- {:.1f} (coverage)'.format(
            np.mean(coverage), np.std(coverage)))
    print('  {:.2f} +/- {:.1f} ((inv)variance)'.format(
            np.mean(variance_inv), np.std(variance_inv)))
    print('  {:.2f} +/- {:.1f} (steps per episode)'.format(
            np.mean(nb_steps), np.std(nb_steps)))
    with open(args.result_path, 'wb') as fh:
        pickle.dump(stats_all, fh)

    assert len(stats_all) == args.max_episodes, len(stats_all)
    if env.render_proc is not None:
        env.render_proc.terminate()
        # env.cloth.stop_render()


if __name__ == "__main__":
    pp = argparse.ArgumentParser()
    pp.add_argument("--max_episodes", type=int, default=5)
    pp.add_argument("--seed", type=int)
    pp.add_argument("--tier", type=int, default = 3)
    pp.add_argument('--start_idx', type = int, default = 0)
    pp.add_argument('--end_idx', type = int, default = 199)
    pp.add_argument('--mode', type = str, default = 'eval')
    
    args = pp.parse_args()
    assert args.tier in [1,2,3]
    assert args.mode in ['eval', 'eval_new_dist']

    # Use this to store results. For example, these can be used to save the
    # demonstrations that we later load to augment DeepRL training. We can
    # augment the file name later in `run()`. Add policy name so we know the
    # source. Fortunately, different trials can be combined in a larger lists.
    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d/%H-%M-%S")

    save_folder = os.path.join('../logs', date_string)
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Each time we use the environment, we need to pass in some configuration.
    args.file_path = fp = os.path.dirname(os.path.realpath(__file__))
    args.cfg_file = join(fp, '../cfg/demo_baselines.yaml') # BASELINES!
    args.render_path = join(fp, '../render/build')    # Must be compiled!

    with open(args.cfg_file, 'r') as fh:
        cfg = yaml.safe_load(fh)
        if args.seed is not None:
            seed = args.seed
            cfg['seed'] = seed  # Actually I don't think it's needed but doesn't hurt?
        else:
            seed = cfg['seed']
            
        cfg['type'] = 'tier' + str(args.tier)
            
        if seed == 1500 or seed == 1600:
            print('Ideally, avoid using these two seeds.')
            sys.exit()
        assert cfg['env']['clip_act_space'] and cfg['env']['delta_actions']

    np.random.seed(seed)

    # Should seed env this way, following gym conventions.  NOTE: we pass in
    # args.cfg_file here, but then it's immediately loaded by ClothEnv. When
    # env.reset() is called, it uses the ALREADY loaded parameters, and does
    # NOT re-query the file again for parameters (that'd be bad!).
    env = ClothEnv(args.cfg_file)
    env.seed(seed)
    env.render(filepath=args.render_path)
    
    
    policy_idxs = np.arange(args.start_idx, args.end_idx)
    
    for rep in np.arange(args.max_episodes):
    
        for policy_idx in np.arange(args.start_idx, args.end_idx):
            """
            Example: split across 3 parallel runs with start_idx and end_idx
            set such that the 3 for loops traverse
            np.arange(0, 67), np.arange(67, 134), np.arange(134, 200)
            """
            
            policy_path = '../../bc_runs/2022-03-15/12-18-35-online_gs400_b64_seed1/Policy_info/Model_epoch_' + \
                str(policy_idx) + '.pth'
              
            policy = CornerPullingBCPolicy(policy_path)
            
            filename = 'Policy_' + str(policy_idx) + '_rep_' + str(rep) + '.pkl'
            args.result_path = join(fp, os.path.join(save_folder, filename))
    
            run(args, policy, env, mode = args.mode)


