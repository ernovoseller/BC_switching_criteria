"""
This script applies augmentations to a dataset of GymCloth demonstrations.
"""


import numpy as np

import pickle
import os
import datetime

from PIL import Image

from analysis.color_change_data import augment_obs



def save_images(images_to_plot, filepath, imgs_per_row = 6):
    """
    Assumes all images have the same dimensions.
    """

    def float_to_int(im):
        if np.max(im) <= 1:
            im = im * 255
            im = im.astype(int)
        im = np.nan_to_num(im)
        return im
    images = []
    for frame in images_to_plot:
        if type(frame) == dict:
            frame = frame['obs']

        if type(frame) == np.ndarray:
            
            if len(frame.shape) == 3 and frame.shape[-1] == 1:
                im = frame[:, :, 0]
            else:
                im = frame
            im = Image.fromarray(im.astype('uint8'))
            images.append(im)
            
        else:
            raise ValueError
    
    width, height = images[0].size
    
    num_row = int(np.ceil(len(images) / imgs_per_row))
    num_col = imgs_per_row if len(images) >= imgs_per_row else len(images)
    
    new_im = Image.new('RGB', (width * num_col, height * num_row))
    
    x_offset = 0
    y_offset = 0
    
    for i, im in enumerate(images):
      new_im.paste(im, (x_offset, y_offset))
      x_offset += width
    
      if (i + 1) % imgs_per_row == 0:
          x_offset = 0
          y_offset += height
    
    new_im.save(filepath)    



def augment_images(seed=0, input_file='data.pkl', 
    save_folder = '', augment = True, save_aug_images = False, aug_multiple = 1,
    imgs_folder = 'augmented_images'):
    """
    obs_per_iter: environment steps per algorithm iteration
    num_nets: number of neural nets in the policy ensemble
    input_file: where initial BC data is stored (output of generate_offline_data())
    stop_updating: stop autotuning of switching thresholds at this iteration
    target_rate: desired rate of context switching
    q_learning: if True, train Q_risk safety critic
    gamma: discount factor for Q-learning
    execution_time: if non-zero, run this number of execution-time trials
    """

    np.random.seed(seed)

    input_data = pickle.load(open(input_file, 'rb'))

    num_bc = len(input_data)

    observations = []
    actions = []
    for i in range(num_bc):
        traj = input_data[i]['obs']
        for j in range(len(traj)-1):
            
            observations.append(traj[j])
            actions.append(input_data[i]['act'][j])

    # For saving some sample augmented images, just as a sanity check:
    if not os.path.exists(imgs_folder):
        os.makedirs(imgs_folder)

    mode = 'train'

    def augment(observations, actions):
        print("augmenting")

        count = 0
        
        observations_aug = []
        actions_aug = []

        for i in range(num_bc):
            
            print('Augmenting trajectory %i of %i' % (i + 1, num_bc))
            
            traj = input_data[i]['obs']

            observations_aug_ep = []
            actions_aug_ep = []

            for j in range(len(traj)-1):

                obs = observations[count]
                act = actions[count]
                
                count += 1
            
                for k in range(aug_multiple):
                
                    if mode == 'train':
                        img_aug, act_aug = augment_obs(obs, mode = mode, action = act)
                    else:
                        img_aug = augment_obs(obs, mode = mode)
                        act_aug = act

                    
                    observations_aug_ep.append(img_aug)
                    actions_aug_ep.append(act_aug)
                
                    if save_aug_images and i < 3 and k < 10:
                        
                        save_images([obs, img_aug], os.path.join(imgs_folder,
                                    'Img_' + str(i) + '_' + str(j) + '_' + str(k) + '.png'))
                 
            # Save a pkl file for this trajectory:
            save_data = {'obs': observations_aug_ep, 'act': actions_aug_ep}            
            save_filepath = os.path.join(save_folder, 'Eps_' + str(i) + '.pkl')
            pickle.dump(save_data, open(save_filepath, 'wb'))                
            
            observations_aug.extend(observations_aug_ep)
            actions_aug.extend(actions_aug_ep)
                
        return observations_aug, actions_aug

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    augment(observations, actions)
        


if __name__ == '__main__':

    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d/%H-%M-%S")
    
    # Number of augmented copies of each observation to save
    aug_multiple = 10
    
    # Path to demonstrations that we want to augment:
    input_file = 'BC_data/data_25x25_match_real_norm.pkl'
    
    # Folder in which to save results:
    save_folder = 'BC_data/data_25x25_match_real_norm_aug_' + str(aug_multiple)
    
    imgs_folder = "Plots/augmented_images"
    
    augment_images(input_file = input_file,
        save_folder = save_folder, save_aug_images = True,
        aug_multiple = aug_multiple, imgs_folder = imgs_folder)

