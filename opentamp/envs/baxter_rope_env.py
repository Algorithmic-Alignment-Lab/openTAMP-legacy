from opentamp.envs.baxter_mjc_env import *
from opentamp.util_classes.mjc_xml_utils import MUJOCO_MODEL_X_OFFSET, MUJOCO_MODEL_Z_OFFSET
from gym import spaces


IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64


class BaxterRopeEnv(BaxterMJCEnv):
    def __init__(self, rope_info, rope_pos=(0.5, -0.2, 0.0), obs_include=[], im_dims=(IMAGE_WIDTH, IMAGE_HEIGHT), mult=1e2, view=True):
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(8,))
        rope = get_rope(rope_info['length'], 
                         rope_info['spacing'], 
                         rope_info['radius'],
                         rope_pos)
        self._rope_present = True
        self.rope_length = rope_info['length']
        self.rope_sphere_radius = rope_info['radius']
        self.rope_spacing = rope_info['spacing']
        self.rope_info = rope_info
        super(BaxterRopeEnv, self).__init__(mode='end_effector_pos', 
                                             items=[rope], 
                                             obs_include=obs_include,
                                             im_dims=im_dims,
                                             max_iter=500,
                                             view=view)

    def reset(self):
        self.randomize_rope()
        self.physics.data.qpos[1:8] = self._calc_ik(START_EE[:3], START_EE[3:7], True, False)
        self.physics.data.qpos[10:17] = self._calc_ik(START_EE[7:10], START_EE[10:14], False, False)
        self.physics.forward()
        self._cur_iter = 0

        return super(BaxterRopeEnv, self).reset()


    def _set_obs_info(self, obs_include):
        ind = super(BaxterRopeEnv, self)._set_obs_info(obs_include)

        if 'rope_joints' in obs_include or not len(obs_include):
            n_jnts = 6 + 2 * self.rope_length - 1
            self._obs_inds['rope_joints'] = (ind, ind+n_jnts)
            self._obs_shape['rope_joints'] = (n_jnts,)
            ind += n_jnts

        if 'rope_points' in obs_include or not len(obs_include):
            n_pnts = self.rope_length
            self._obs_inds['rope_points'] = (ind, ind+3*n_pnts)
            self._obs_shape['rope_points'] = (n_pnts, 3)
            ind += 3*n_pnts

        self.dO = ind
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(ind,), dtype='float32')
        return ind


    def get_obs(self, obs_include=None):
        if obs_include is None:
            obs_include = self.obs_include

        obs = super(BaxterRopeEnv, self).get_obs(obs_include)
        if (not len(obs_include) or 'rope_joints' in obs_include) and 'rope_joints' in self._obs_inds:
            jnts = self.get_rope_joints().flatten()
            inds = self._obs_inds['rope_joints']
            obs[inds[0]:inds[1]] = jnts

        if (not len(obs_include) or 'rope_points' in obs_include) and 'rope_points' in self._obs_inds:
            pnts = self.get_rope_points().flatten()
            inds = self._obs_inds['rope_points']
            obs[inds[0]:inds[1]] = pnts

        return np.array(obs)


    def get_rope_point(self, x):
        if not self._rope_present:
            raise AttributeError('No rope in model (remember to supply rope_info).')

        model = self.physics.model
        name = 'B{0}'.format(x)
        if name in self._ind_cache:
            point_ind = self._ind_cache[name]
        else:
            point_ind = model.name2id(name, 'body')
            self._ind_cache[name] = point_ind
        return self.physics.data.xpos[point_ind]


    def get_leftmost_rope_point(self, points=None):
        # "Leftmost" along the y-axis
        if points is None:
            points = self.get_rope_points()
        return max(points, key=lambda p: p[1])


    def get_rightmost_rope_point(self, points=None):
        # "Rightmost" along the y-axis
        if points is None:
            points = self.get_rope_points()
        return max(points, key=lambda p: -p[1])


    def get_uppermost_rope_point(self, points=None):
        # "Uppermost" along the x-axis
        if points is None:
            points = self.get_rope_points()
        return max(points, key=lambda p: p[0])


    def get_lowermost_rope_point(self, points=None):
        # "Lowermost" along the x-axis
        if points is None:
            points = self.get_rope_points()
        return max(points, key=lambda p: -p[0])


    def get_left_rope_endpoint(self):
        # "Leftmost" along the y-axis
        endpoints = self.get_endpoints()
        return self.get_leftmost_rope_point(endpoints)


    def get_right_rope_endpoint(self):
        # "Rightmost" along the y-axis
        corners = self.get_corners()
        return self.get_rightmost_rope_point(corners)


    def get_upper_rope_endpoint(self):
        # "Uppermost" along the x-axis
        endpoints = self.get_endpoints()
        return self.get_uppermost_rope_point(corners)


    def get_lower_rope_endpoint(self):
        # "Lowermost" along the x-axis
        endpoints = self.get_endpoints()
        return self.get_lowermost_rope_point(corners)


    def get_pos_from_label(self, label, mujoco_frame=True):
        endpoints = self.get_endpoints()
        endpoints = sorted(endpoints, lambda c1, c2: 1 if c1[1] < c2[1] else -1)
        pos = None
        if label == 'left_endpoint':
            pos = endpoints[0] if endpoints[0][0] > endpoints[1][0] else endpoints[1]
        if label == 'right_endpoint':
            pos = endpoints[0] if endpoints[1][0] > endpoints[0][0] else endpoints[1]
        if label == 'upper_endpoint':
            pos = endpoints[0] if endpoints[0][1] > endpoints[1][1] else endpoints[1]
        if label == 'lower_endpoint':
            pos = endpoints[0] if endpoints[1][1] > endpoints[0][1] else endpoints[1]
        if label == 'leftmost':
            pos = self.get_leftmost_rope_point()
        if label == 'rightmost':
            pos = self.get_rightmost_rope_point()
        if label == 'uppermost':
            pos = self.get_uppermost_rope_point()
        if label == 'lowermost':
            pos = self.get_lowermost_rope_point()
        if pos is not None:
            if not mujoco_frame:
                pos[2] -= MUJOCO_MODEL_Z_OFFSET
                pos[0] -= MUJOCO_MODEL_X_OFFSET
            return pos
        return super(BaxterRopeEnv, self).get_pos_from_label(label, mujoco_frame)


    def get_endpoints(self, mujoco_frame=True):
        endpoints = [self.get_rope_point(0),
                   self.get_rope_point(self.rope_length-1)]
        if not mujoco_frame:
            for c in endpoints:
                c[2] -= MUJOCO_MODEL_Z_OFFSET
                c[0] -= MUJOCO_MODEL_X_OFFSET
        return endpoints


    def get_rope_points(self):
        points_inds = []
        model = self.physics.model
        for x in range(self.rope_length):
            name = 'B{0}'.format(x)
            points_inds.append(model.name2id(name, 'body'))
        return self.physics.data.xpos[points_inds]


    def get_rope_joints(self):
        return self.physics.data.qpos[19:25+2*self.rope_length-1]


    def set_rope_joints(self, qpos):
        self.physics.data.qpos[19:25+2*self.rope_length-1] = qpos
        self.physics.forward()


    def _shift_rope(self, x, y, z):
        self.physics.data.qpos[19:22] += (x, y, z)
        self.physics.forward()


    def randomize_rope(self, x_bounds=(-0.15, 0.25), y_bounds=(-0.7, 1.1)):
        if not self._rope_present:
            raise AttributeError('This environment does not contain a rope.')

        n_folds = np.random.randint(3, 15)
        inds = np.random.choice(list(range(25, 25+2*self.rope_length-1)), n_folds)
        for i in inds:
            self.physics.data.qpos[i] = np.random.uniform(-1, 1)
        self.physics.forward()

        x = np.random.uniform(x_bounds[0], x_bounds[1])
        y = np.random.uniform(y_bounds[0],y_bounds[1])
        z = np.random.uniform(0, 0.0)
        self._shift_rope(x, y, z)

        jnt_angles = self.get_joint_angles()
        self.physics.step()

        self.physics.data.qpos[1:19] = jnt_angles
        self.physics.forward()


    def is_done(self):
        return super(BaxterRopeEnv, self).is_done()


    def compute_reward(self):
        return 0


class BaxterContinuousRopeEnv(BaxterRopeEnv):
    def __init__(self):
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(8,))
        rope_info = {'length': 10, 'spacing': 0.1, 'radius': 0.01}
        rope = get_deformable_rope(rope_info['length'], 
                                     rope_info['spacing'], 
                                     rope_info['radius'],
                                     (0.5, -0.2, 0.0))
        self._rope_present = True
        self.rope_length = rope_info['length']
        self.rope_sphere_radius = rope_info['radius']
        self.rope_spacing = rope_info['spacing']
        self.rope_info = rope_info
        obs_include = set(['end_effector', 'rope_points', 'rope_joints'])
        super(BaxterRopeEnv, self).__init__(mode='end_effector_pos', 
                                             items=[rope], 
                                             obs_include=obs_include,
                                             im_dims=(IMAGE_WIDTH, IMAGE_HEIGHT),
                                             view=False)



class BaxterDiscreteRopeEnv(BaxterRopeEnv):
    def __init__(self):
        self.action_space = spaces.Discrete(16)
        rope_info = {'length': 10, 'spacing': 0.1, 'radius': 0.01}
        rope = get_deformable_rope(rope_info['length'], 
                                     rope_info['spacing'], 
                                     rope_info['radius'],
                                     (0.5, -0.2, 0.0))
        self._rope_present = True
        self.rope_length = rope_info['length']
        self.rope_sphere_radius = rope_info['radius']
        self.rope_spacing = rope_info['spacing']
        self.rope_info = rope_info
        obs_include = set(['end_effector', 'rope_points', 'rope_joints'])
        super(BaxterRopeEnv, self).__init__(mode='discrete_pos', 
                                             items=[rope], 
                                             obs_include=obs_include, 
                                             im_dims=(IMAGE_WIDTH, IMAGE_HEIGHT),
                                             view=False)
