from opentamp.envs.baxter_mjc_env import *
from opentamp.util_classes.mjc_xml_utils import MUJOCO_MODEL_X_OFFSET, MUJOCO_MODEL_Z_OFFSET
from gym import spaces

NO_CLOTH = 0
NO_FOLD = 1
ONE_FOLD = 2
TWO_FOLD = 3
WIDTH_GRASP = 4
LENGTH_GRASP = 5
TWO_GRASP = 6
HALF_WIDTH_GRASP = 7
HALF_LENGTH_GRASP = 8
TWIST_FOLD = 9
RIGHT_REACHABLE = 10
LEFT_REACHABLE = 11
IN_RIGHT_GRIPPER = 12
IN_LEFT_GRIPPER = 13
LEFT_FOLD_ON_TOP = 14
RIGHT_FOLD_ON_TOP = 15
DIAGONAL_GRASP = 16

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64


class BaxterClothEnv(BaxterMJCEnv):
    def __init__(self, cloth_info, cloth_pos=(0.5, -0.2, 0.0), obs_include=[], im_dims=(IMAGE_WIDTH, IMAGE_HEIGHT), mode='end_effector_pos', mult=1e2, view=True):
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(8,))
        cloth = get_deformable_cloth(cloth_info['width'], 
                                     cloth_info['length'], 
                                     cloth_info['spacing'], 
                                     cloth_info['radius'],
                                     cloth_pos)
        self._cloth_present = True
        self.cloth_width = cloth_info['width']
        self.cloth_length = cloth_info['length']
        self.cloth_sphere_radius = cloth_info['radius']
        self.cloth_spacing = cloth_info['spacing']
        self.cloth_info = cloth_info
        super(BaxterClothEnv, self).__init__(mode=mode, 
                                             items=[cloth], 
                                             obs_include=obs_include,
                                             im_dims=im_dims,
                                             max_iter=500,
                                             view=view)

    def reset(self):
        self.randomize_cloth()
        self.physics.data.qpos[1:8] = self._calc_ik(START_EE[:3], START_EE[3:7], True, False)
        self.physics.data.qpos[10:17] = self._calc_ik(START_EE[7:10], START_EE[10:14], False, False)
        self.physics.forward()
        self._cur_iter = 0

        return super(BaxterClothEnv, self).reset()


    def get_state(self):
        return self.physics.data.qpos.copy()


    def set_state(self, state):
        self.physics.data.qpos[:] = state
        self.physics.forward()


    def __getstate__(self):
        return self.physics.data.qpos.tolist()


    def __setstate__(self, state):
        self.physics.data.qpos[:] = state
        self.physics.forward()


    def _set_obs_info(self, obs_include):
        ind = super(BaxterClothEnv, self)._set_obs_info(obs_include)

        if 'cloth_joints' in obs_include or not len(obs_include):
            n_jnts = 6 + 2 * self.cloth_width * self.cloth_length - 1
            self._obs_inds['cloth_joints'] = (ind, ind+n_jnts)
            self._obs_shape['cloth_joints'] = (n_jnts,)
            ind += n_jnts

        if 'cloth_points' in obs_include or not len(obs_include):
            n_pnts = self.cloth_width * self.cloth_length
            self._obs_inds['cloth_points'] = (ind, ind+3*n_pnts)
            self._obs_shape['cloth_points'] = (n_pnts, 3)
            ind += 3*n_pnts

        self.dO = ind
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(ind,), dtype='float32')
        return ind


    def get_obs(self, obs_include=None, view=False):
        if obs_include is None:
            obs_include = self.obs_include

        obs = super(BaxterClothEnv, self).get_obs(obs_include)
        if (not len(obs_include) or 'cloth_joints' in obs_include) and 'cloth_joints' in self._obs_inds:
            jnts = self.get_cloth_joints().flatten()
            inds = self._obs_inds['cloth_joints']
            obs[inds[0]:inds[1]] = jnts

        if (not len(obs_include) or 'cloth_points' in obs_include) and 'cloth_points' in self._obs_inds:
            pnts = self.get_cloth_points().flatten()
            inds = self._obs_inds['cloth_points']
            obs[inds[0]:inds[1]] = pnts

        return np.array(obs)


    def get_cloth_point(self, x, y):
        if not self._cloth_present:
            raise AttributeError('No cloth in model (remember to supply cloth_info).')

        model = self.physics.model
        name = 'B{0}_{1}'.format(x, y)
        if name in self._ind_cache:
            point_ind = self._ind_cache[name]
        else:
            point_ind = model.name2id(name, 'body')
            self._ind_cache[name] = point_ind
        return self.physics.data.xpos[point_ind]


    def get_leftmost_cloth_point(self, points=None):
        # "Leftmost" along the y-axis
        if points is None:
            points = self.get_cloth_points()
        return max(points, key=lambda p: p[1])


    def get_rightmost_cloth_point(self, points=None):
        # "Rightmost" along the y-axis
        if points is None:
            points = self.get_cloth_points()
        return max(points, key=lambda p: -p[1])


    def get_uppermost_cloth_point(self, points=None):
        # "Uppermost" along the x-axis
        if points is None:
            points = self.get_cloth_points()
        return max(points, key=lambda p: p[0])


    def get_lowermost_cloth_point(self, points=None):
        # "Lowermost" along the x-axis
        if points is None:
            points = self.get_cloth_points()
        return max(points, key=lambda p: -p[0])


    def get_leftmost_cloth_corner(self):
        # "Leftmost" along the y-axis
        corners = self.get_corners()
        return self.get_leftmost_cloth_point(corners)


    def get_rightmost_cloth_corner(self):
        # "Rightmost" along the y-axis
        corners = self.get_corners()
        return self.get_rightmost_cloth_point(corners)


    def get_uppermost_cloth_corner(self):
        # "Uppermost" along the x-axis
        corners = self.get_corners()
        return self.get_uppermost_cloth_point(corners)


    def get_lowermost_cloth_corner(self):
        # "Lowermost" along the x-axis
        corners = self.get_corners()
        return self.get_lowermost_cloth_point(corners)


    def get_highest_left_cloth_corner(self):
        # "Highest" along the z-axis
        corners = self.get_corners()
        corners = sorted(corners, lambda c1, c2: 1 if c1[2] < c2[2] else -1)
        return corners[0] if corners[0][1] > corners[1][1] else corners[1]


    def get_highest_right_cloth_corner(self):
        # "Highest" along the z-axis
        corners = self.get_corners()
        corners = sorted(corners, lambda c1, c2: 1 if c1[2] < c2[2] else -1)
        return corners[0] if corners[0][1] <= corners[1][1] else corners[1]


    def get_leftmost_reachable_corner(self, arm='left'):
        corners = self.get_corners()
        corners = sorted(corners, lambda c1, c2: 1 if c1[1] < c2[1] else -1)
        for i in range(len(corners)):
            c = corners[i]
            if self._check_ik([c[0], c[1], 0.65], [0, 0, 1, 0], arm=='right'):
                if i < 2:
                    label2 = 'left'
                    label1 = 'top' if c[0] > corners[(i + 1) % 2][0] else 'bottom'
                else:
                    label2 = 'right'
                    label1 = 'top' if c[0] > corners[2+((i + 1) % 2)][0] else 'bottom'
                return c, '{0}_{1}'.format(label1, label2)

        return None, 'No Corner'


    def get_rightmost_reachable_corner(self, arm='right'):
        corners = self.get_corners()
        corners = sorted(corners, lambda c1, c2: 1 if c1[1] > c2[1] else -1)
        for i in range(len(corners)):
            c = corners[i]
            if self._check_ik([c[0], c[1], 0.65], [0, 0, 1, 0], arm=='right'):
                if i < 2:
                    label2 = 'right'
                    label1 = 'top' if c[0] > corners[(i + 1) % 2][0] else 'bottom'
                else:
                    label2 = 'left'
                    label1 = 'top' if c[0] > corners[2+((i + 1) % 2)][0] else 'bottom'
                return c, '{0}_{1}'.format(label1, label2)

        return None, 'No Corner'


    def get_pos_from_label(self, label, mujoco_frame=True):
        corners = self.get_corners()
        corners = sorted(corners, lambda c1, c2: 1 if c1[1] < c2[1] else -1)
        pos = None
        if label == 'top_left':
            pos = corners[0] if corners[0][0] > corners[1][0] else corners[1]
        if label == 'bottom_left':
            pos = corners[0] if corners[0][0] < corners[1][0] else corners[1]
        if label == 'top_right':
            pos = corners[2] if corners[2][0] > corners[3][0] else corners[3]
        if label == 'bottom_right':
            pos = corners[2] if corners[2][0] < corners[3][0] else corners[3]
        if label == 'leftmost':
            pos = self.get_leftmost_cloth_corner()
        if label == 'rightmost':
            pos = self.get_rightmost_cloth_corner()
        if label == 'highest_left':
            pos = self.get_highest_left_cloth_corner()
        if label == 'highest_right':
            pos = self.get_highest_right_cloth_corner()
        if pos is not None:
            if not mujoco_frame:
                pos[2] -= MUJOCO_MODEL_Z_OFFSET
                pos[0] -= MUJOCO_MODEL_X_OFFSET
            return pos
        return super(BaxterClothEnv, self).get_pos_from_label(label, mujoco_frame)


    def get_corners(self, mujoco_frame=True):
        corners = [self.get_cloth_point(0, 0),
                   self.get_cloth_point(0, self.cloth_width-1),
                   self.get_cloth_point(self.cloth_length-1, 0),
                   self.get_cloth_point(self.cloth_length-1, self.cloth_width-1)]
        if not mujoco_frame:
            for c in corners:
                c[2] -= MUJOCO_MODEL_Z_OFFSET
                c[0] -= MUJOCO_MODEL_X_OFFSET
        return corners


    def get_cloth_points(self):
        points_inds = []
        model = self.physics.model
        for x in range(self.cloth_length):
            for y in range(self.cloth_width):
                name = 'B{0}_{1}'.format(x, y)
                points_inds.append(model.name2id(name, 'body'))
        return self.physics.data.xpos[points_inds]


    def get_cloth_joints(self):
        return self.physics.data.qpos[19:25+2*self.cloth_width*self.cloth_length-1]


    def set_cloth_joints(self, qpos):
        self.physics.data.qpos[19:25+2*self.cloth_width*self.cloth_length-1] = qpos
        self.physics.forward()


    def activate_cloth_eq(self):
        for i in range(self.cloth_length):
            for j in range(self.cloth_width):
                pnt = self.get_cloth_point(i, j)
                right_ee = self.get_right_ee_pos()
                left_ee = self.get_left_ee_pos()
                right_grip, left_grip = self.get_gripper_joint_angles()

                r_eq_name = 'right{0}_{1}'.format(i, j)
                l_eq_name = 'left{0}_{1}'.format(i, j)
                if r_eq_name in self._ind_cache:
                    r_eq_ind = self._ind_cache[r_eq_name]
                else:
                    r_eq_ind = self.physics.model.name2id(r_eq_name, 'equality')
                    self._ind_cache[r_eq_name] = r_eq_ind

                if l_eq_name in self._ind_cache:
                    l_eq_ind = self._ind_cache[l_eq_name]
                else:
                    l_eq_ind = self.physics.model.name2id(l_eq_name, 'equality')
                    self._ind_cache[l_eq_name] = l_eq_ind

                if np.all(np.abs(pnt - right_ee) < GRASP_THRESHOLD) and right_grip < 0.015:
                    # if not self.physics.model.eq_active[r_eq_ind]:
                    #     self._shift_cloth_to_grip(right_ee, (i, j))
                    self.physics.model.eq_active[r_eq_ind] = True
                # else:
                #     self.physics.model.eq_active[r_eq_ind] = False

                if np.all(np.abs(pnt - left_ee) < GRASP_THRESHOLD) and left_grip < 0.015:
                    # if not self.physics.model.eq_active[l_eq_ind]:
                    #     self._shift_cloth_to_grip(left_ee, (i, j))
                    self.physics.model.eq_active[l_eq_ind] = True
                # else:
                #     self.physics.model.eq_active[l_eq_ind] = False


    def _shift_cloth(self, x, y, z):
        self.physics.data.qpos[19:22] += (x, y, z)
        self.physics.forward()


    def randomize_cloth(self, x_bounds=(-0.15, 0.25), y_bounds=(-0.7, 1.1)):
        if not self._cloth_present:
            raise AttributeError('This environment does not contain a cloth.')

        n_folds = np.random.randint(3, 15)
        inds = np.random.choice(list(range(25, 25+2*self.cloth_width*self.cloth_length-1)), n_folds)
        for i in inds:
            self.physics.data.qpos[i] = np.random.uniform(-1, 1)
        self.physics.forward()

        x = np.random.uniform(x_bounds[0], x_bounds[1])
        y = np.random.uniform(y_bounds[0],y_bounds[1])
        z = np.random.uniform(0, 0.0)
        self._shift_cloth(x, y, z)

        jnt_angles = self.get_joint_angles()
        self.physics.step()

        self.physics.data.qpos[1:19] = jnt_angles
        self.physics.forward()


    def is_done(self):
        state = self.check_cloth_state()
        if ONE_FOLD in state or TWO_FOLD in state: return True

        return super(BaxterClothEnv, self).is_done()


    def compute_reward(self):
        start_t = time.time()
        state = self.check_cloth_state()
        state.append(IN_LEFT_GRIPPER)

        if NO_CLOTH in state: return 0

        if TWO_FOLD in state: return 1e4

        reward = 0

        ee_right_pos = self.get_right_ee_pos()
        ee_left_pos = self.get_left_ee_pos()

        corner1 = self.get_item_pos('B0_0')
        corner2 = self.get_item_pos('B0_{0}'.format(self.cloth_width-1))
        corner3 = self.get_item_pos('B{0}_0'.format(self.cloth_length-1))
        corner4 = self.get_item_pos('B{0}_{1}'.format(self.cloth_length-1, self.cloth_width-1))
        corners = [corner1, corner2, corner3, corner4]


        min_right_dist = min([np.linalg.norm(ee_right_pos-corners[i]) for i in range(4)])
        min_left_dist = min([np.linalg.norm(ee_left_pos-corners[i]) for i in range(4)])

        if ONE_FOLD in state:
            reward += 5e3
            if self.cloth_length % 2:
                mid1 = self.get_item_pos('B{0}_0'.format(self.cloth_length // 2))
                mid2 = self.get_item_pos('B{0}_{1}'.format(self.cloth_length // 2, self.cloth_width-1))
            else:
                mid1 = (self.get_item_pos('B{0}_0'.format(self.cloth_length // 2)-1) \
                        + self.get_item_pos('B{0}_0'.format(self.cloth_length // 2))) / 2.0
                mid2 = (self.get_item_pos('B{0}_{1}'.format(self.cloth_length // 2 - 1, self.cloth_width-1)) \
                        + self.get_item_pos('B{0}_{1}'.format(self.cloth_length // 2, self.cloth_width-1))) / 2.0

            if self.cloth_width % 2:
                mid3 = self.get_item_pos('B0_{0}'.format(self.cloth_width // 2))
                mid4 = self.get_item_pos('B{0}_{1}'.format(self.cloth_length-1, self.cloth_width // 2))
            else:
                mid3 = (self.get_item_pos('B0_{0}'.format(self.cloth_width // 2)-1) \
                        + self.get_item_pos('B0_{0}'.format(self.cloth_width // 2))) / 2.0
                mid4 = (self.get_item_pos('B{0}_{1}'.format(self.cloth_length-1, self.cloth_width // 2 - 1)) \
                        + self.get_item_pos('B{0}_{1}'.format(self.cloth_length-1, self.cloth_width // 2))) / 2.0

            min_dist = min([
                            np.linalg.norm(corner1-ee_left_pos) + np.linalg.norm(mid3-ee_right_pos),
                            np.linalg.norm(corner1-ee_right_pos) + np.linalg.norm(mid3-ee_left_pos),
                            np.linalg.norm(corner3-ee_left_pos) + np.linalg.norm(mid4-ee_right_pos),
                            np.linalg.norm(corner3-ee_right_pos) + np.linalg.norm(mid4-ee_left_pos),
                           ])
            reward -= min_dist
            reward -= 1e1 * np.linalg.norm(corner1 - corner3)
            reward -= 1e1 * np.linalg.norm(mid3 - mid4)
            reward += 1e1 * (0.75 * self.cloth_spacing - np.linalg.norm(corner1 - mid3))
            reward += 1e1 * (0.75 * self.cloth_spacing - np.linalg.norm(corner3 - mid4))

        elif LENGTH_GRASP in state:
            reward += 5e2
            right_corner = min(corners, key=lambda c: np.linalg.norm(ee_right_pos-c))
            left_corner = min(corners, key=lambda c: np.linalg.norm(ee_left_pos-c))

            mid5 = self.get_item_pos('B0_{0}'.format(int((self.cloth_width - 1.5) // 2)))
            mid6 = self.get_item_pos('B0_{0}'.format(int((self.cloth_width + 1.5) // 2)))
            mid7 = self.get_item_pos('B{0}_{1}'.format(self.cloth_length-1, int((self.cloth_width - 1.5) // 2)))
            mid8 = self.get_item_pos('B{0}_{1}'.format(self.cloth_length-1, int((self.cloth_width + 1.5) // 2)))
            min_dist = min([
                            np.linalg.norm(corner1-ee_left_pos) + np.linalg.norm(corner3-ee_right_pos),
                            np.linalg.norm(corner1-ee_right_pos) + np.linalg.norm(corner3-ee_left_pos),
                            np.linalg.norm(corner2-ee_left_pos) + np.linalg.norm(corner4-ee_right_pos),
                            np.linalg.norm(corner2-ee_right_pos) + np.linalg.norm(corner4-ee_left_pos),
                           ])
            reward -= 5e1 * min_dist
            reward -= 5e1 * np.linalg.norm(corner1[:2] - corner2[:2])
            reward -= 5e1 * np.linalg.norm(corner3[:2] - corner4[:2])
            reward += 5e1 * np.linalg.norm(corner1[:2] - corner3[:2])
            reward += 5e1 * np.linalg.norm(corner2[:2] - corner4[:2])
            reward -= 1e1 * np.linalg.norm(mid5[:2] - mid6[:2])
            reward -= 1e1 * np.linalg.norm(mid7[:2] - mid8[:2])

        elif TWIST_FOLD in state:
            reward += 7.5e1
            left_most_corner = max(corners, key=lambda c: c[0])
            bottom_left_corner = self.get_pos_from_label('bottom_left')
            reward -= 5e0*np.linalg.norm(ee_right_pos-bottom_left_corner)
            reward -= 5e0*np.linalg.norm(ee_left_pos-left_most_corner)

        elif DIAGONAL_GRASP in state:
            reward += 2.5e1
            right_corner = min(corners, key=lambda c: np.linalg.norm(ee_right_pos-c))
            left_corner = min(corners, key=lambda c: np.linalg.norm(ee_left_pos-c))
            reward += np.linalg.norm(left_corner-right_corner)
            reward += 1e1*min(corners, key=lambda c:c[0])[0]
            if left_corner[0] > 0.55:
                reward -= max(corners, key=lambda c:c[2])[2]

        elif LEFT_REACHABLE in state and RIGHT_REACHABLE in state:
            reward += 1e1
            right_corner = min(corners, key=lambda c: np.linalg.norm(ee_right_pos-c))
            left_corner = max(corners, key=lambda c: np.linalg.norm(ee_left_pos-c))
            reward -= np.linalg.norm(ee_right_pos - right_corner)
            reward -= np.linalg.norm(ee_left_pos - left_corner)
            reward += min(np.linalg.norm(ee_left_pos - ee_right_pos), 0.1)

        elif IN_RIGHT_GRIPPER:
            left_most_corner = max(corners, key=lambda c: c[0])
            reward += 5e0
            reward += left_most_corner[1]

        elif IN_LEFT_GRIPPER:
            right_most_corner = min(corners, key=lambda c: c[0])
            reward += 5e0
            reward -= right_most_corner[1]

        elif RIGHT_REACHABLE in state:
            left_most_corner = max(corners, key=lambda c: c[0])
            reward -= 1e1 * np.linalg.norm(ee_right_pos-left_most_corner)

        elif LEFT_REACHABLE in state:
            right_most_corner = min(corners, key=lambda c: c[0])
            reward -= 1e1 * np.linalg.norm(ee_left_pos-right_most_corner)

        return reward


    def check_cloth_state(self):
        if not self._cloth_present: return [NO_CLOTH]

        state = []

        # Check full fold
        corner1 = self.get_item_pos('B0_0')
        corner2 = self.get_item_pos('B0_{0}'.format(self.cloth_width-1))
        corner3 = self.get_item_pos('B{0}_0'.format(self.cloth_length-1))
        corner4 = self.get_item_pos('B{0}_{1}'.format(self.cloth_length-1, self.cloth_width-1))

        corners = [corner1, corner2, corner3, corner4]
        check1 = all([all([np.max(np.abs(corners[i][:2] - corners[j][:2])) < 0.04 for j in range(i+1, 4)]) for i in range(4)])

        mid1 = self.get_item_pos('B{0}_0'.format(int((self.cloth_length - 1.5) // 2)))
        mid2 = self.get_item_pos('B{0}_0'.format(int((self.cloth_length + 1.5) // 2)))
        mid3 = self.get_item_pos('B{0}_{1}'.format(int((self.cloth_length - 1.5) // 2), self.cloth_width-1))
        mid4 = self.get_item_pos('B{0}_{1}'.format(int((self.cloth_length + 1.5) // 2), self.cloth_width-1))
        mids = [mid1, mid2, mid3, mid4]
        check2 = all([all([np.max(np.abs(mids[i][:2] - mids[j][:2])) < 0.04 for j in range(i+1, 4)]) for i in range(4)])

        mid5 = self.get_item_pos('B0_{0}'.format(int((self.cloth_width - 1.5) // 2)))
        mid6 = self.get_item_pos('B0_{0}'.format(int((self.cloth_width + 1.5) // 2)))
        mid7 = self.get_item_pos('B{0}_{1}'.format(self.cloth_length-1, int((self.cloth_width - 1.5) // 2)))
        mid8 = self.get_item_pos('B{0}_{1}'.format(self.cloth_length-1, int((self.cloth_width + 1.5) // 2)))
        mids = [mid5, mid6, mid7, mid8]
        check2 = all([all([np.max(np.abs(mids[i][:2] - mids[j][:2])) < 0.04 for j in range(i+1, 4)]) for i in range(4)])

        check3 = np.linalg.norm(corner1[:2] - mid1[:2]) > 0.75 * self.cloth_spacing and \
                 np.linalg.norm(corner1[:2] - mid5[:2]) > 0.75 * self.cloth_spacing and \
                 np.linalg.norm(mid1[:2] - mid5[:2]) > 0.75 * self.cloth_spacing

        if check1 and check2 and check3: state.append(TWO_FOLD)

        # Check length-wise fold
        corner1 = self.get_item_pos('B0_0')
        corner2 = self.get_item_pos('B0_{0}'.format(self.cloth_width-1))
        check1 = np.max(np.abs(corner1[:2] - corner2[:2])) < 0.04

        mid1 = self.get_item_pos('B0_{0}'.format(int((self.cloth_width - 1.5) // 2)))
        mid2 = self.get_item_pos('B0_{0}'.format(int((self.cloth_width + 1.5) // 2)))
        check2 = np.max(np.abs(mid1[:2] - mid2[:2])) < 0.02

        corner3 = self.get_item_pos('B{0}_0'.format(self.cloth_length-1))
        corner4 = self.get_item_pos('B{0}_{1}'.format(self.cloth_length-1, self.cloth_width-1))
        check3 = np.max(np.abs(corner3[:2] - corner4[:2])) < 0.04

        mid3 = self.get_item_pos('B{0}_{1}'.format(self.cloth_length-1, int((self.cloth_width - 1.5) // 2)))
        mid4 = self.get_item_pos('B{0}_{1}'.format(self.cloth_length-1, int((self.cloth_width + 1.5) // 2)))
        check4 = np.max(np.abs(mid3[:2] - mid4[:2])) < 0.04

        check5 = np.linalg.norm(corner1[:2] - mid1[:2]) > 0.75 * self.cloth_spacing and \
                 np.linalg.norm(corner3[:2] - mid3[:2]) > 0.75 * self.cloth_spacing

        if check1 and check2 and check3 and check4 and check5: state.append(ONE_FOLD)

        # Check twist-fold
        corner1 = self.get_item_pos('B0_0')
        corner2 = self.get_item_pos('B0_{0}'.format(self.cloth_width-1))
        corner3 = self.get_item_pos('B{0}_0'.format(self.cloth_length-1))
        corner4 = self.get_item_pos('B{0}_{1}'.format(self.cloth_length-1, self.cloth_width-1))

        dist1 = np.linalg.norm(corner1 - corner4)
        dist2 = np.linalg.norm(corner2 - corner3)
        if dist1 > dist2:
            check1 = dist1 > 0.9 * (self.cloth_spacing*np.sqrt(self.cloth_width**2+self.cloth_length**2))
            check2 = np.abs(corner1[0] - corner4[0]) < 0.08

            far_x_pos = 0.8 * (self.cloth_length * self.cloth_width / dist1)
            check3 = corner3[0] - corner1[0] > far_x_pos and corner2[0] - corner4[0] > far_x_pos
        else:
            check1 = dist2 > 0.9 * (self.cloth_spacing*np.sqrt(self.cloth_width**2+self.cloth_length**2))
            check2 = np.abs(corner2[0] - corner3[0]) < 0.08

            far_x_pos = 0.8 * (self.cloth_length * self.cloth_width / dist1)
            check3 = corner1[0] - corner3[0] > far_x_pos and corner4[0] - corner2[0] > far_x_pos
        if check1 and check2 and check3: state.append('TWIST_FOLD')


        # Check two corner grasp
        corner1 = self.get_item_pos('B0_0')
        corner2 = self.get_item_pos('B0_{0}'.format(self.cloth_width-1))
        corner3 = self.get_item_pos('B{0}_0'.format(self.cloth_length-1))
        corner4 = self.get_item_pos('B{0}_{1}'.format(self.cloth_length-1, self.cloth_width-1))

        ee_left_pos = self.get_left_ee_pos()
        ee_right_pos = self.get_right_ee_pos()
        grips = self.get_gripper_joint_angles()

        check1 = np.linalg.norm(ee_right_pos - corner1) < 0.02 and grips[0] < 0.04 and np.linalg.norm(ee_left_pos - corner3) < 0.02 and grips[1] < 0.04
        check2 = np.linalg.norm(ee_left_pos - corner1) < 0.02 and grips[1] < 0.04 and np.linalg.norm(ee_right_pos - corner3) < 0.02 and grips[0] < 0.04

        check3 = np.linalg.norm(ee_right_pos - corner2) < 0.02 and grips[0] < 0.04 and np.linalg.norm(ee_left_pos - corner4) < 0.02 and grips[1] < 0.04
        check4 = np.linalg.norm(ee_left_pos - corner2) < 0.02 and grips[1] < 0.04 and np.linalg.norm(ee_right_pos - corner4) < 0.02 and grips[0] < 0.04

        if check1 or check2 or check3 or check4: state.append(LENGTH_GRASP)

        for c in corners:
            if self._check_ik([c[0], c[1], 0.675], [0, 0, 1, 0], True):
                state.append(RIGHT_REACHABLE)
                break

        for c in corners:
            if self._check_ik([c[0], c[1], 0.675], [0, 0, 1, 0], False):
                state.append(LEFT_REACHABLE)
                break

        if any([np.all(np.abs(ee_right_pos - c) < 0.02) and grips[0] < 0.05 for c in corners]): state.append(IN_RIGHT_GRIPPER)
        if any([np.all(np.abs(ee_left_pos - c) < 0.02) and grips[1] < 0.05 for c in corners]): state.append(IN_LEFT_GRIPPER)

        if TWIST_FOLD in state:
            edge1, edge2 = [], []
            for i in range(self.cloth_length):
                edge1.append(self.cloth_point(i, 0))
                edge2.append(self.cloth_point(i, self.cloth_width-1))
            edges = np.array([edge1, edge2])
            left_edge = np.argmax(np.sum(edges[:,:,1]), axis=1)
            top_edge = np.argmax(np.sum(edges[:,:,2]), axis=1)
            if left_edge == top_edge:
                state.append(LEFT_FOLD_ON_TOP)
            else:
                state.append(RIGHT_FOLD_ON_TOP)

        return state


class BaxterContinuousClothEnv(BaxterClothEnv):
    def __init__(self):
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(8,))
        cloth_info = {'width': 5, 'length': 3, 'spacing': 0.1, 'radius': 0.01}
        cloth = get_deformable_cloth(cloth_info['width'], 
                                     cloth_info['length'], 
                                     cloth_info['spacing'], 
                                     cloth_info['radius'],
                                     (0.5, -0.2, 0.0))
        self._cloth_present = True
        self.cloth_width = cloth_info['width']
        self.cloth_length = cloth_info['length']
        self.cloth_sphere_radius = cloth_info['radius']
        self.cloth_spacing = cloth_info['spacing']
        self.cloth_info = cloth_info
        obs_include = set(['end_effector', 'cloth_points', 'cloth_joints'])
        super(BaxterClothEnv, self).__init__(mode='end_effector_pos', 
                                             items=[cloth], 
                                             obs_include=obs_include,
                                             im_dims=(IMAGE_WIDTH, IMAGE_HEIGHT),
                                             view=False)



class BaxterDiscreteClothEnv(BaxterClothEnv):
    def __init__(self):
        self.action_space = spaces.Discrete(16)
        cloth_info = {'width': 5, 'length': 3, 'spacing': 0.1, 'radius': 0.01}
        cloth = get_deformable_cloth(cloth_info['width'], 
                                     cloth_info['length'], 
                                     cloth_info['spacing'], 
                                     cloth_info['radius'],
                                     (0.5, -0.2, 0.0))
        self._cloth_present = True
        self.cloth_width = cloth_info['width']
        self.cloth_length = cloth_info['length']
        self.cloth_sphere_radius = cloth_info['radius']
        self.cloth_spacing = cloth_info['spacing']
        self.cloth_info = cloth_info
        obs_include = set(['end_effector', 'cloth_points', 'cloth_joints'])
        super(BaxterClothEnv, self).__init__(mode='discrete_pos', 
                                             items=[cloth], 
                                             obs_include=obs_include, 
                                             im_dims=(IMAGE_WIDTH, IMAGE_HEIGHT),
                                             view=False)
