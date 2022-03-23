# BC_switching_criteria

This project involves determining when policies trained in simulation are ready to be deployed to the real world. We study behavior cloning for a fabric smoothing task, and utilize the [Gym-Cloth simulator](https://github.com/DanielTakeshi/gym-cloth), developed in Seita et al., "[Deep Imitation Learning of Sequential Fabric Smoothing From an Algorithmic Supervisor](https://arxiv.org/abs/1910.04854)", IROS 2020.

Installation instructions are as detailed for Gym-Cloth [here](https://github.com/DanielTakeshi/gym-cloth), except that our requirements.txt file includes several additional dependencies.

## Examples of how to run:

Before interacting with Gym-Cloth, it is important to run gym-cloth/setup.py, which "cythonizes" the Python .pyx files. For examples of how to do this, see the .sh files in 1. and 2.:

1) To generate the same oracle demonstration dataset that we use, run gym-cloth/run_oracle.sh.
2) To run a learned policy in the simulator, modify gym-cloth/run_policies.sh as needed.

To run online behavior cloning:

3) python Online_BC.py

To perform image augmentations on the dataset:

4) python augment_images.py 

5) analysis/ includes some data analysis and image processing scripts, which can be run similarly to 3) and 4).
