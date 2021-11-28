import numpy as np

from gym.spaces import *

from opentamp.envs import BaxterLeftBlockStackEnv

from policy_hooks.vae.vae_env import VAEEnvWrapper


class BlockSortEnv(VAEEnvWrapper):
    action_space = spaces.MultiDiscrete([3, 4, 4])
    observation_space = spaces.Box(0, 255, [80, 107, 3], dtype='uint8')

    def __init__(self):
        env = BaxterLeftBlockStackEnv()
        config = {}
        act_space = env.action_space
        prim_dims =  {'prim{}'.format(i): act_space.nvec[i] for i in range(0, len(act_space.nvec))}
        config['vae'] = {}
        config['vae']['task_dims'] = int(np.prod(list(prim_dims.values())))
        config['vae']['obs_dims'] = (env.im_height, env.im_wid, 3)
        config['vae']['weight_dir'] = '/home/michaelmcdonald/tampy/src/tf_saved/baxterleftblockstackenv_t20_vae_data_3_blocks'
        config['vae']['rollout_len'] = 20
        config['vae']['load_step'] = 200000
        config['vae']['train_mode'] = 'conditional'
        config['topic'] = 'LeftBlockStack'
        super(BlockSortEnv, self).__init__(config, env=env)
