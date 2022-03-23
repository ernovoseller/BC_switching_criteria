from gym.envs.registration import register
from gym import envs

all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs]

if not 'cloth-v0' in env_ids:

    register(
        id='cloth-v0',
        entry_point='gym_cloth.envs:ClothEnv',
    )
