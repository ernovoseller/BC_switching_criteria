# -*- coding: utf-8 -*-
"""
Given a dataset of real images from a physical robot system, this script
allows for evaluating a policy on these to see how well it makes predictions
on real images.
"""

import pickle
import os
import numpy as np
import cv2
import sys
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Folder for loading policies
folder_name = '../bc_runs/2022-03-15/12-18-35-online_gs400_b64_seed1/Policy_info/'

# Folder for saving the resulting plots:
save_folder = '../Plots/Eval_on_real_policy'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# For normalizing real robot system images:
minimum = 20
maximum = 155
newRange = 255
oldRange = maximum - minimum

def normalize(img_crop):
    img_crop = np.where(img_crop > minimum, img_crop, minimum)
    img_crop = np.where(img_crop < maximum, img_crop, maximum)
    var = newRange/oldRange
    img_crop = (img_crop - minimum) * var
    img_crop = img_crop.astype(np.uint8)
    img_crop = cv2.cvtColor(img_crop, cv2.COLOR_GRAY2BGR)
    img_crop = cv2.fastNlMeansDenoisingColored(img_crop, None, 7, 7, 7, 21)
    return img_crop


real_image_folder = '../YuMi_images/2022-03-15'

real_image_subfolders = ['18-31-17', '18-24-09', '17-20-35']

real_observations = []

for subfolder in real_image_subfolders:
    
    filepath = os.path.join(real_image_folder, subfolder, '25x25_real.pkl')

    data = pickle.load(open(filepath, 'rb'))
    obs_traj = data['obs_crop']

    for obs in obs_traj:
        
        obs = normalize(obs)
        
        obs = np.expand_dims(obs[:, :, 0], -1)
        
        obs = np.transpose(obs, (2, 0, 1))
        obs = np.ascontiguousarray(obs)
        
        real_observations.append(obs)


# For loading the policies:
behavior_cloning_path = os.path.dirname(os.path.abspath(__file__)) + "/../"
sys.path.insert(0, behavior_cloning_path)

policy_idxs = [10, 50, 100, 125, 150, 197]      # Policy idxs to evaluate
  
device_idx = 0
if device_idx >= 0:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
  
def act_to_kps(act):  # Converts actions to units of pixels
    act[0:2] = (act[0:2] + 1)/2
    x, y, dx, dy = act
    # Use 223 rather than 224 because pixels are indexed from 0 to 223
    x, y, dx, dy = int(x*223), 223 - int(y*223), int(dx*223), -int(dy*223)
    return [x, y], [x+dx, y+dy]  #[y, x], [y+dy, x+dx]
 
for obs_idx, obs in enumerate(real_observations): 
        
    print('Processing obs %i of %i' % (obs_idx + 1, len(real_observations)))
    
    for policy_idx in policy_idxs:
        
        model = torch.load(folder_name + 'Model_epoch_'
                                + str(policy_idx) + '.pth', map_location=device)
        
        action = model.act(obs)
    
        pick_px, place_px = act_to_kps(action) #(c,r)
    
        # Plot policy's predicted pick and place points on top of the image:
        plt.figure()
        plt.imshow(obs[0, :, :], vmin = 0, vmax = 255, cmap = 'gray')   
        
        plt.scatter(x=[pick_px[0]], y=[pick_px[1]], color='green')
        plt.scatter(x=[place_px[0]], y=[place_px[1]], color= 'blue')
    
        filepath = os.path.join(save_folder, 'Obs_' + str(obs_idx) + '.png')
        plt.savefig(filepath)
        plt.close()
    
