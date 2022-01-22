from opentamp.src.policy_hooks.mcts_explore import MCTSExplore

from opentamp.src.policy_hooks.vae.trained_envs import *


env = BlockStackEnv()
mcts = MCSExplore(env)
mcts.run()
