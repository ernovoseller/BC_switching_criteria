# -*- coding: utf-8 -*-
"""
Load holdout set, and calculate validation loss and epistemic uncertainty metrics.
"""

import pickle
import os
import torch
import sys
import numpy as np
from PIL import Image

from color_change_data import tint_data

behavior_cloning_path = os.path.dirname(os.path.abspath(__file__)) + "/../"
sys.path.insert(0, behavior_cloning_path)

# To get 5 ensemble members, I would run 5 different random seeds.
BC_model_folders = ['../bc_runs/2022-03-15/12-18-35-online_gs400_b64_seed1/Policy_info/',
                    '../bc_runs/2022-03-17/01-09-31-online_gs400_b64_seed2/Policy_info/',
                    '../bc_runs/2022-03-17/01-10-05-online_gs400_b64_seed3/Policy_info/',
                    '../bc_runs/2022-03-17/01-10-42-online_gs400_b64_seed4/Policy_info/',
                    '../bc_runs/2022-03-17/01-11-09-online_gs400_b64_seed5/Policy_info/']
num_seeds = len(BC_model_folders)

# Load dataset of holdout oracle demonstrations:
data = pickle.load(open('../BC_data/holdout_set_oracle_tier3_200.pkl', 'rb'))

print('%i trajectories in the holdout dataset' % len(data))

# Save images corresponding to a few images from the holdout set. I added this in
# order to make sure there are no errors regarding color normalization, etc. in the 
# holdout set (relative to training).
img_save_folder = '../Plots/Sanity_check_holdout_trajectories'

if not os.path.exists(img_save_folder):
    os.makedirs(img_save_folder)

save_folder = '../logs'     # Metrics will get saved in here

if not os.path.exists(save_folder):
    os.makedirs(save_folder)


for i in range(20):
    
    obs = data[i]['obs'][0]

    im = Image.fromarray(obs)
    
    im.save(os.path.join(img_save_folder, 'Obs_' + str(i) + '.png'))


# Process holdout set:
holdout_set = {'obs': [], 'act': []}

for i in range(len(data)):
    
    print('Holdout %i of %i' % (i + 1, len(data)))
    
    observations = data[i]['obs']
    actions = data[i]['act']
    
    for obs, act in zip(observations[:-1], actions):
        
        # Process observation:
        obs = tint_data([obs], norm = False)[0]
        
        obs = np.expand_dims(obs[:, :, 0], axis = -1)
    
        obs = np.transpose(obs, (2, 0, 1))
        obs = np.ascontiguousarray(obs)
        
        # Add obs and action to holdout set:
        holdout_set['obs'].append(obs)
        holdout_set['act'].append(act)


device_idx = 0
if device_idx >= 0:
    device = torch.device("cuda")   #, device_idx)
else:
    device = torch.device("cpu")

num_policies = 199

validation_err = np.empty((num_policies, num_seeds))

# Calculating epistemic uncertainty in 3 different ways; we ended up finding that
# the 1st of these ("epist") gave the most stable performance. epist_delta and epist_delta_norm
# calculate epistemic uncertainty just over the direction of the action (not 
# the pick point), and epist_delta_norm normalizes these directions to unit magnitude.
epist = np.empty(num_policies)
epist_delta = np.empty(num_policies)
epist_delta_norm = np.empty(num_policies)

for training_idx in range(num_policies):
    
    print('Processing training iteration %i of %i' % (training_idx + 1, num_policies))
    
    pred_all = []
    
    for ens_idx, ens_folder in enumerate(BC_model_folders):
    
        policy_path = os.path.join(ens_folder, 'Model_epoch_' + str(training_idx) + '.pth')
        
        model = torch.load(policy_path, map_location=device)
        
        # Calculate prediction accuracy on validation set.

        a_pred = model.act(np.array(holdout_set['obs']))
        a_sup = np.array(holdout_set['act'])
        validation = np.sum(a_pred - a_sup, axis = 1)**2
        
        validation_err[training_idx, ens_idx] = np.mean(validation)
    
        # Store predictions for calculating epistemic uncertainty:
        pred_all.append(a_pred)
            
    # Calculate epistemic uncertainty across the ensemble:
    pred_all = np.array(pred_all)    # Dims: (num ens, num obs, action dim)
    
    num_obs = pred_all.shape[1]
    epist_tmp = np.empty(num_obs)
    epist_delta_tmp = np.empty(num_obs)
    epist_delta_norm_tmp = np.empty(num_obs)
    
    for i in range(num_obs):
        
        pred = pred_all[:, i, :]   # Dims: (num ens, action dim)
        
        epist_tmp[i] = np.square(np.std(np.array(pred), axis=0)).mean()
        
        pred_delta = np.array(pred[:, 2:])
        epist_delta_tmp[i] = np.square(np.std(pred_delta, axis=0)).mean()
        
        pred_delta_norm = pred_delta / np.linalg.norm(pred_delta, axis = 1)[:, np.newaxis]
        
        epist_delta_norm_tmp[i] = np.square(np.std(pred_delta_norm, axis=0)).mean()
    
    epist[training_idx] = np.mean(epist_tmp)
    epist_delta[training_idx] = np.mean(epist_delta_tmp)
    epist_delta_norm[training_idx] = np.mean(epist_delta_norm_tmp)
    
    
pickle.dump({'validation_err': validation_err, 'epist': epist,
             'epist_delta': epist_delta, 'epist_delta_norm': epist_delta_norm}, 
             open(os.path.join(img_save_folder, 'Validation_epist_unc.pkl'), 'wb'))
    
