"""
This script provides functions for tinting and augmenting image observations.
"""

import os
import argparse
import pickle
import numpy as np

from imgaug.augmentables import Keypoint, KeypointsOnImage
from imgaug import augmenters as iaa


import sys
analysis_path = os.path.dirname(os.path.abspath(__file__)) + "/../analysis/"
sys.path.insert(0, analysis_path)
from tint_cloth import tint_cloth, normalize

# For tinting observations similarly to real images:
def tint_data(observation, norm = False):
    light_brown_multiplier = [1.0, 1.0, 1.0] #BGR
    dark_brown_multiplier = [0.5, 0.5, 0.6] #BGR
    background_multiplier = [0.9, 0.9, 0.9]
    new_obs = []
    for i in range(len(observation)):
        tinted_step = tint_cloth(observation[i], light_brown_multiplier, dark_brown_multiplier, background_multiplier)
       
        if norm:
            tinted_step = normalize(tinted_step)
            
        new_obs.append(tinted_step)
    return new_obs


# For applying image augmentations:
lc_limits = (0.9, 1.1)
add_limits = (-10, 10)
gaussian_noise_limits = (0, 0.0125*255)

lc_range = lc_limits[1] - lc_limits[0]
lc_limits_ext = (lc_limits[0] - 0.1 * lc_range, lc_limits[1] + 0.1 * lc_range)

add_range = add_limits[1] - add_limits[0]
add_limits_ext = (add_limits[0] - 0.1 * add_range, add_limits[1] + 0.1 * add_range)

gaussian_noise_limits_ext = (0, 1.2 * gaussian_noise_limits[1])

# During training, image flips can be applied
kpt_augs_train = [ 
    iaa.LinearContrast(lc_limits, per_channel=0.25),  # (0.8, 1.2)
    iaa.Add(add_limits, per_channel=False),    # (-20, 20)
    iaa.AdditiveGaussianNoise(scale=gaussian_noise_limits),
    iaa.flip.Flipud(0.5), # vertical flips
    iaa.flip.Fliplr(0.5) # horizontal flips
    ]
seq_kpts_train = iaa.Sequential(kpt_augs_train, random_order=True) 

# During evaluation, there are no image flips
kpt_augs_eval = [ 
    iaa.LinearContrast(lc_limits, per_channel=0.25),  # (0.8, 1.2)
    iaa.Add(add_limits, per_channel=False),    # (-20, 20)
    iaa.AdditiveGaussianNoise(scale=gaussian_noise_limits)
    ]
seq_kpts_eval = iaa.Sequential(kpt_augs_eval, random_order=True) 

# Expanding the range of the augmentations beyond what is seen during training:
kpt_augs_eval_new_dist = [ 
    iaa.LinearContrast(lc_limits_ext, per_channel=0.25),  # (0.8, 1.2)
    iaa.Add(add_limits_ext, per_channel=False),    # (-20, 20)
    iaa.AdditiveGaussianNoise(scale=gaussian_noise_limits_ext)
    ]
seq_kpts_eval_new_dist = iaa.Sequential(kpt_augs_eval, random_order=True) 


def act_to_kps(act):
    x, y, dx, dy = act
    x, y, dx, dy = int(x*224), int(y*224), int(dx*224), int(dy*224)
    return (x, y), (x+dx, y+dy)

def kps_to_act(kps):
    x = kps[0][0]/224
    y = kps[0][1]/224
    dx = (kps[1][0] - kps[0][0])/224
    dy = (kps[1][1] - kps[0][1])/224
    return np.clip(np.array([x, y, dx, dy]), -1, 1)

def augment_obs(obs, mode = 'train', action = None):
    """
    Augment the observation.
    There are currently 3 modes: train, eval, eval_new_dist.
    All 3 modes randomize brightness and contrast, and apply additive Gaussian noise.
    Training mode additionally applies flips to the images (which does not make sense
    to do during evaluation).
    eval_new_dist mode expands the range of the augmentations beyond what is seen
    during training, while eval mode keeps it the same.
    """
    
    if mode == 'train':
        seq = seq_kpts_train

        kps = [Keypoint(x, y) for x, y in act_to_kps(action)]
        kps = KeypointsOnImage(kps, shape=obs.shape) 
        
        img_aug, kps_aug = seq(image=obs, keypoints=kps)
        kps_aug = kps_aug.to_xy_array().astype(int)
 
        img_aug = img_aug[:, :, 0]
        img_aug = np.transpose(np.array([img_aug, img_aug, img_aug]), (1, 2, 0))        
     
        return img_aug, kps_to_act(kps_aug)

    elif mode == 'eval':
        seq = seq_kpts_eval
        
        img_aug = seq(image=obs)
        
    elif mode == 'eval_new_dist':
        seq = seq_kpts_eval_new_dist
        
        img_aug = seq(image=obs)      
        
    else:
        raise ValueError('Invalid mode passed to augment_obs.')

    img_aug = img_aug[:, :, 0]
    img_aug = np.transpose(np.array([img_aug, img_aug, img_aug]), (1, 2, 0))
   
    return img_aug


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, default='../BC_data/data_25x25_2000_iters.pkl')
    parser.add_argument('--normalize', type = bool, default = True)
    args = parser.parse_args()

    new_data_loc = '../BC_data/data_25x25_match_real_norm.pkl'

    combined = []
    with open(args.file, 'rb') as f:
        combined = pickle.load(f)

    for i in range(0, len(combined)):
        print("Processing iteration %d" % (i))
        observation = combined[i]['obs']
        tinted_obs = tint_data(observation, norm = args.normalize)
        
        combined[i]['obs'] = tinted_obs

    with open(new_data_loc, 'wb') as f:
        pickle.dump(combined, f)
