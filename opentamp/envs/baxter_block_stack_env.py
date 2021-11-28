import itertools
import numpy as np
import random
import time

from gym import spaces

from opentamp.envs import BaxterMJCEnv


N_BLOCKS = 4
BLOCK_DIM = 0.03
# IMAGE_WIDTH, IMAGE_HEIGHT = (64, 48)
# IMAGE_WIDTH, IMAGE_HEIGHT = (96, 72)
IMAGE_WIDTH, IMAGE_HEIGHT = (107, 80)
# IMAGE_WIDTH, IMAGE_HEIGHT = (200, 150)
# POSSIBLE_BLOCK_LOCS = np.r_[list(itertools.product(range(45, 76, 10), range(20, 71, 10))), 
#                             list(itertools.product(range(45, 76, 10), range(-70, -21, 10)))].astype(np.float64) / 100.

# POSSIBLE_BLOCK_REGION_LOCS = np.array([
#     np.array(list(itertools.product(range(45, 60, 2), range(20, 70, 2)))).astype(np.float64) / 100.,
#     np.array(list(itertools.product(range(60, 72, 2), range(20, 70, 2)))).astype(np.float64) / 100.,
#     np.array(list(itertools.product(range(45, 60, 2), range(-70, -20, 2)))).astype(np.float64) / 100.,
#     np.array(list(itertools.product(range(60, 72, 2), range(-70, -20, 2)))).astype(np.float64) / 100.
# ])

POSSIBLE_LEFT_BLOCK_REGION_LOCS = np.array([
    np.array(list(itertools.product(list(range(47, 57, 2)), list(range(10, 30, 2))))).astype(np.float64) / 100.,
    np.array(list(itertools.product(list(range(62, 74, 2)), list(range(10, 30, 2))))).astype(np.float64) / 100.,
    np.array(list(itertools.product(list(range(47, 57, 2)), list(range(40, 65, 2))))).astype(np.float64) / 100.,
    np.array(list(itertools.product(list(range(62, 74, 2)), list(range(40, 65, 2))))).astype(np.float64) / 100.,
])

POSSIBLE_RIGHT_BLOCK_REGION_LOCS = np.array([
    np.array(list(itertools.product(list(range(47, 57, 2)), list(range(-30, -10, 2))))).astype(np.float64) / 100.,
    np.array(list(itertools.product(list(range(62, 74, 2)), list(range(-30, -10, 2))))).astype(np.float64) / 100.,
    np.array(list(itertools.product(list(range(47, 57, 2)), list(range(-65, -40, 2))))).astype(np.float64) / 100.,
    np.array(list(itertools.product(list(range(62, 74, 2)), list(range(-65, -40, 2))))).astype(np.float64) / 100.,
])

class BaxterBlockStackEnv(BaxterMJCEnv):
    def __init__(self):
        # self.action_space = spaces.Discrete(len(self.get_action_meanings()))
        self.action_space = spaces.MultiDiscrete((len(self.get_action_meanings()), N_BLOCKS, N_BLOCKS))

        include_items = []
        colors = [[1, 1, 1, 1], [0.75, 0.5, 1, 1], [0, 0, 1, 1], [1, 1, 0, 1], [1, 0, 1, 1], [0.5, 0.5, 0.5, 1], [0.5, 0.5, 0, 1], [0.5, 0, 0.5, 1], [0, 0.5, 0, 1], [0, 0, 0.5, 1]]
        for n in range(N_BLOCKS):
            next_block = {'name': 'block{0}'.format(n), 
                          'type': 'box', 
                          'is_fixed': False, 
                          'pos': (0., 0, 0),
                          'dimensions': (BLOCK_DIM, BLOCK_DIM, BLOCK_DIM), 
                          'mass': 1,
                          'rgba': colors.pop(0),
                          'sim_freq': 100}
            include_items.append(next_block)

        obs_include = ['forward_image']
        super(BaxterMJCEnv, self).__init__(mode='discrete', 
                                             include_items=include_items, 
                                             items=[],
                                             obs_include=obs_include, 
                                             im_dims=(IMAGE_WIDTH, IMAGE_HEIGHT),
                                             view=False)

        seed = int(1000*time.time()) % 1000
        np.random.seed(seed)
        random.seed(seed)
        self.randomize_init_state()
        self.reset()
        time.sleep(0.5)
        self.render(camera_id=1)
        self.render(camera_id=1)
        self.physics.data.qpos[16] = np.pi/2
        self.physics.forward()
        self.n_blocks = N_BLOCKS
        self.main_camera_id = 1
        self.observation_space = spaces.Box(low=0, high=256, shape=(80, 107, 3), dtype=np.uint8)


    def reset(self):
        self.physics.data.qpos[:] = self.init_state
        return super(BaxterMJCEnv, self).reset()


    def reset_goal(self):
        self.goal = {}
        order = list(range(N_BLOCKS))
        random.shuffle(order)
        height = np.random.choice(list(range(2, N_BLOCKS+1)))
        for i in range(0, height):
            if i > 0:
                pos = self.get_item_pos('block{0}'.format(order[i-1]))
                pos += np.array([0, 0, 2*BLOCK_DIM])
            else:
                pos = self.get_item_pos('block{0}'.format(order[0]))
            self.goal['block{0}'.format(order[i])] = pos

        for i in range(height-1, N_BLOCKS):
            self.goal['block{0}'.format(order[i])] = self.get_item_pos('block{0}'.format(order[i]))

        return self.obs_goal()


    def obs_goal(self):
        cur_state = self.physics.data.qpos.copy()
        for i in range(N_BLOCKS):
            self.set_item_pos('block{0}'.format(i), self.goal['block{0}'.format(i)], forward=False)
        self.physics.forward()
        obs = self.render(camera_id=self.main_camera_id)
        self.physics.data.qpos[:] = cur_state
        self.physics.forward()
        return obs


    def check_goal(self):
        correct = 0
        for i in range(N_BLOCKS):
            if np.linalg.norm(self.get_item_pos('block{0}'.format(i)) - self.goal['block{0}'.format(i)]) < 0.08:
                correct += 1
        return correct


    def randomize_init_state(self):
        locs1 = POSSIBLE_LEFT_BLOCK_REGION_LOCS.copy()
        locs2 = POSSIBLE_RIGHT_BLOCK_REGION_LOCS.copy()
        locs = np.r_[locs1, locs2]
        np.random.shuffle(locs)
        # for i in range(N_BLOCKS):
        #     self.set_item_pos('block{0}'.format(i), np.r_[locs[i], -0.02], forward=False)
        for i in range(N_BLOCKS):
            l = locs[i]
            ind = np.random.choice(list(range(len(l))))
            loc = list(l[ind])
            self.set_item_pos('block{0}'.format(i), np.r_[loc, -0.02], forward=False)
        self.physics.forward()
        self.init_state = self.physics.data.qpos.copy()


    def _randomize_goal(self):
        order = list(range(N_BLOCKS))
        random.shuffle(order)
        pos1 = self.get_item_pos('block{0}'.format(order[0]))
        self.goal = {}


    def get_action_meanings(self):
        # meanings = ['NOOP']
        # meanings = []
        # for i in range(N_BLOCKS):
        #     # meanings.append('LEFTGRASP_BLOCK{0}'.format(i))
        #     # meanings.append('RIGHTGRASP_BLOCK{0}'.format(i))
        #     meanings.append('LEFTCENTER_BLOCK{0}'.format(i))
        #     meanings.append('RIGHTCENTER_BLOCK{0}'.format(i))
        #     for j in range(N_BLOCKS):
        #         if i==j: continue
        #         meanings.append('LEFTSTACK_BLOCK{0}_BLOCK{1}'.format(i, j))
        #         meanings.append('LEFTMOVE_BLOCK{0}_BLOCK{1}'.format(i, j))
        #         meanings.append('RIGHTSTACK_BLOCK{0}_BLOCK{1}'.format(i, j))
        #         meanings.append('RIGHTMOVE_BLOCK{0}_BLOCK{1}'.format(i, j))
        # meanings = ['LEFTSTACK', 'LEFTMOVE', 'LEFTCENTER', 'RIGHTSTACK', 'RIGHTMOVE', 'RIGHTCENTER']
        meanings = ['LEFTSTACK', 'LEFTMOVE', 
                    'RIGHTSTACK', 'RIGHTMOVE',
                    'LEFTMOVE_L', 'RIGHTMOVE_R']
        return meanings


    def step(self, action, mode=None, obs_include=None, view=True, debug=False):
        if mode is None:
            mode = self.ctrl_mode

        cmd = np.zeros((18))
        abs_cmd = np.zeros((18))

        r_grip = 0
        l_grip = 0

        if mode == 'discrete':
            old_obs_include = self.obs_include
            if obs_include is not None:
                self.obs_include = obs_include

            action_type = self.get_action_meanings()[action[0]]
            item1_pos = self.get_item_pos('block{0}'.format(action[1])) - [0, 0, 0.02]
            item2_pos = self.get_item_pos('block{0}'.format(action[2])) - [0, 0, 0.02]
            # action_meaning = self.get_action_meanings()[action].split('_')
            # action_type = action_meaning[0]
            # item1_pos = self.get_item_pos(action_meaning[1].lower())
            # item2_pos = self.get_item_pos(action_meaning[2].lower()) if len(action_meaning) > 2 else None

            grasp = np.array([0, 0, 0.01])
            if 'STACK' not in action_type and item1_pos[2] > 0.02: # and N_BLOCKS > 3:
                obs = [self.get_obs(view=False)]
            elif action_type == 'LEFTCENTER':
                obs = self.move_left_to(item1_pos+grasp, [0.55, 0, 0])
            elif action_type == 'RIGHTCENTER':
                obs = self.move_right_to(item1_pos+grasp, [0.55, 0, 0])
            elif action_type == 'LEFTSTACK':
                obs = self.move_left_to(item1_pos+grasp, item2_pos + [0, 0, 2*BLOCK_DIM+0.02])
            elif action_type == 'RIGHTSTACK':
                obs = self.move_right_to(item1_pos+grasp, item2_pos + [0, 0, 2*BLOCK_DIM+0.02])
            elif action_type == 'RIGHTCENTER':
                obs = self.move_right_to(item1_pos+grasp, item2_pos + [0, 0, 2*BLOCK_DIM+0.02])
            elif action_type == 'LEFTMOVE':
                # pos2 = [item2_pos[0], item2_pos[1] - 0.15, 0]
                pos2 = [item1_pos[0], np.maximum(item1_pos[1] - 0.2, -0.1), 0]
                obs = self.move_left_to(item1_pos+grasp, pos2)
            elif action_type == 'RIGHTMOVE':
                # pos2 = [item2_pos[0], item2_pos[1] + 0.15, 0]
                pos2 = [item1_pos[0], np.minimum(item1_pos[1] + 0.2, 0.1), 0]
                obs = self.move_right_to(item1_pos+grasp, pos2)
            elif action_type == 'LEFTMOVE_L':
                # pos2 = [item2_pos[0], item2_pos[1] - 0.15, 0]
                pos2 = [item1_pos[0], np.maximum(item1_pos[1] + 0.2, -0.1), 0]
                obs = self.move_left_to(item1_pos+grasp, pos2)
            elif action_type == 'RIGHTMOVE_R':
                # pos2 = [item2_pos[0], item2_pos[1] + 0.15, 0]
                pos2 = [item1_pos[0], np.minimum(item1_pos[1] - 0.2, 0.1), 0]
                obs = self.move_right_to(item1_pos+grasp, pos2)
            else:
                raise NotImplementedError('Action {0} not found.'.format(action_type))
            # elif action_type == 'LEFTGRASP':
            #     self.move_left_to_grasp(item1_pos)
            # elif action_type == 'RIGHTGRASP':
            #     self.move_right_to_grasp(item1_pos)


            self.obs_include = old_obs_include

            obs = [self.render(camera_id=1)]
            return obs[-1], self.check_goal(), self.check_goal(), {} # Reward == is done
            
        return super(BaxterBlockStackEnv, self).step(action, mode, obs_include, view, debug)

    def check_goal(self):
        return 0


class BaxterLeftBlockStackEnv(BaxterBlockStackEnv):
    def randomize_init_state(self):
        locs = POSSIBLE_LEFT_BLOCK_REGION_LOCS.copy()
        np.random.shuffle(locs)
        for i in range(N_BLOCKS):
            l = locs[i]
            ind = np.random.choice(list(range(len(l))))
            loc = list(l[ind])
            self.set_item_pos('block{0}'.format(i), np.r_[loc, -0.02], forward=False)
        self.physics.forward()
        self.init_state = self.physics.data.qpos.copy()


    def _randomize_goal(self):
        order = list(range(N_BLOCKS))
        random.shuffle(order)
        pos1 = self.get_item_pos('block{0}'.format(order[0]))
        self.goal = {}


    def get_action_meanings(self):
        meanings = ['LEFTSTACK', 'LEFTMOVE', 'LEFTMOVE_L']
        return meanings


    def step(self, action, mode=None, obs_include=None, view=True, debug=False):
        if mode is None:
            mode = self.ctrl_mode

        cmd = np.zeros((18))
        abs_cmd = np.zeros((18))

        r_grip = 0
        l_grip = 0

        if mode == 'discrete':
            old_obs_include = self.obs_include
            if obs_include is not None:
                self.obs_include = obs_include

            action_type = self.get_action_meanings()[action[0]]
            item1_pos = self.get_item_pos('block{0}'.format(action[1])) - [0, 0, 0.02]
            item2_pos = self.get_item_pos('block{0}'.format(action[2])) - [0, 0, 0.02]
            # action_meaning = self.get_action_meanings()[action].split('_')
            # action_type = action_meaning[0]
            # item1_pos = self.get_item_pos(action_meaning[1].lower())
            # item2_pos = self.get_item_pos(action_meaning[2].lower()) if len(action_meaning) > 2 else None

            grasp = np.array([0, 0, 0.01])
            if action_type == 'LEFTCENTER':
                obs = self.move_left_to(item1_pos+grasp, [0.55, 0, 0])
            elif action_type == 'LEFTSTACK':
                obs = self.move_left_to(item1_pos+grasp, item2_pos + [0, 0, 2*BLOCK_DIM+0.01])
            elif action_type == 'LEFTMOVE':
                # pos2 = [item2_pos[0], item2_pos[1] - 0.15, 0]
                pos2 = [item1_pos[0], np.maximum(item1_pos[1] - 0.2, -0.1), 0]
                obs = self.move_left_to(item1_pos+grasp, pos2)
            elif action_type == 'LEFTMOVE_L':
                # pos2 = [item2_pos[0], item2_pos[1] - 0.15, 0]
                pos2 = [item1_pos[0], np.maximum(item1_pos[1] + 0.2, -0.1), 0]
                obs = self.move_left_to(item1_pos+grasp, pos2)
            else:
                raise NotImplementedError('Action {0} not found.'.format(action_type))

            self.obs_include = old_obs_include

            obs = [self.render(camera_id=1)]
            return obs[-1], self.check_goal(), self.check_goal(), {} # Reward == is done
            
        return super(BaxterBlockStackEnv, self).step(action, mode, obs_include, view, debug)


