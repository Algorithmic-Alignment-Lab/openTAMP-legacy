# import matplotlib as mpl
# mpl.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from threading import Thread
import time
import xml.etree.ElementTree as xml

from tkinter import TclError

try:
    from dm_control import render
except:
    from dm_control import _render as render
from dm_control.mujoco import Physics
from dm_control.viewer import gui
from dm_control.viewer import renderer
from dm_control.viewer import runtime
from dm_control.viewer import user_input
from dm_control.viewer import util
from dm_control.viewer import viewer
from dm_control.viewer import views

from gym import spaces
from gym.core import Env

import opentamp
from opentamp.envs import BaxterMJCEnv
from opentamp.util_classes.ik_controller import BaxterIKController
from opentamp.util_classes.mjc_xml_utils import *
from opentamp.util_classes import transform_utils as T


BASE_VEL_XML = os.getcwd() + '/opentamp'+'/robot_info/hsr_model.xml'
ENV_XML = os.getcwd() + '/opentamp'+'/robot_info/current_hsr_env.xml'


MUJOCO_JOINT_ORDER = ["slide_x", "slide_y", "arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint", "hand_l_proximal_joint", "hand_r_proximal_joint"]


_MAX_FRONTBUFFER_SIZE = 2048
_CAM_WIDTH = 200
_CAM_HEIGHT = 150

GRASP_THRESHOLD = np.array([0.05, 0.05, 0.025]) # np.array([0.01, 0.01, 0.03])
MJC_TIME_DELTA = 0.002
MJC_DELTAS_PER_STEP = int(1. // MJC_TIME_DELTA)
N_CONTACT_LIMIT = 12

CTRL_MODES = ['joint_angle', 'end_effector', 'end_effector_pos', 'discrete_pos']
MUJOCO_MODEL_X_OFFSET = 0
MUJOCO_MODEL_Z_OFFSET = 0


class HSRMJCEnv(BaxterMJCEnv):
    metadata = {'render.modes': ['human', 'rgb_array', 'depth'], 'video.frames_per_second': 67}

    def __init__(self, mode='end_effector', obs_include=[], items=[], include_files=[], cloth_info=None, im_dims=(_CAM_WIDTH, _CAM_HEIGHT), max_iter=250, view=False):
        assert mode in CTRL_MODES, 'Env mode must be one of {0}'.format(CTRL_MODES)
        self.ctrl_mode = mode
        self.active = True

        self.cur_time = 0.
        self.prev_time = 0.

        self.use_viewer = view
        self.use_glew = 'MUJOCO_GL' not in os.environ or os.environ['MUJOCO_GL'] != 'osmesa'
        self.obs_include = obs_include
        self.include_files = include_files
        self._joint_map_cache = {}
        self._ind_cache = {}
        self._cloth_present = cloth_info is not None
        if self._cloth_present:
            self.cloth_width = cloth_info['width']
            self.cloth_length = cloth_info['length']
            self.cloth_sphere_radius = cloth_info['radius']
            self.cloth_spacing = cloth_info['spacing']

        self.im_wid, self.im_height = im_dims
        self.items = items
        self._item_map = {item[0]: item for item in items}
        self._set_obs_info(obs_include)
        self._load_model()

        self._ikcontrol = BaxterIKController(lambda: self.get_arm_joint_angles())

        # Start joints with grippers pointing downward

        self.action_inds = {
            ('hsr', 'pose'): np.array([0,1]),
            ('hsr', 'arm'): np.array(list(range(2,7))),
            ('hsr', 'gripper'): np.array([7]),
        }

        self._max_iter = max_iter
        self._cur_iter = 0

        if view:
            self._launch_viewer(_CAM_WIDTH, _CAM_HEIGHT)
        else:
            self._viewer = None


    def _load_model(self):
        generate_xml(BASE_VEL_XML, ENV_XML, self.items, self.include_files)
        self.physics = Physics.from_xml_path(ENV_XML)


    def _set_obs_info(self, obs_include):
        self._obs_inds = {}
        self._obs_shape = {}
        ind = 0
        if 'overhead_image' in obs_include or not len(obs_include):
            self._obs_inds['overhead_image'] = (ind, ind+3*self.im_wid*self.im_height)
            self._obs_shape['overhead_image'] = (self.im_height, self.im_wid, 3)
            ind += 3*self.im_wid*self.im_height

        if 'pos' in obs_include or not len(obs_include):
            self._obs_inds['pos'] = (ind, ind+2)
            self._obs_shape['pos'] = (2,)
            ind += 2

        if 'joints' in obs_include or not len(obs_include):
            self._obs_inds['joints'] = (ind, ind+7)
            self._obs_shape['joints'] = (7,)
            ind += 7

        if 'end_effector' in obs_include or not len(obs_include):
            self._obs_inds['end_effector'] = (ind, ind+8)
            self._obs_shape['end_effector'] = (8,)
            ind += 8

        for item, xml, info in self.items:
            if item in obs_include or not len(obs_include):
                self._obs_inds[item] = (ind, ind+3) # Only store 3d Position
                self._obs_shape[item] = (3,)
                ind += 3

        self.dO = ind
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(ind,), dtype='float32')
        return ind


    def get_obs(self, obs_include=None):
        obs = np.zeros(self.dO)
        if obs_include is None:
            obs_include = self.obs_include

        if not len(obs_include) or 'overhead_image' in obs_include:
            pixels = self.render(height=self.im_height, width=self.im_wid, camera_id=0, view=False)
            inds = self._obs_inds['overhead_image']
            obs[inds[0]:inds[1]] = pixels.flatten()

        if not len(obs_include) or 'pos' in obs_include:
            pos = self.get_base_pos()
            inds = self._obs_inds['pos']
            obs[inds[0]:inds[1]] = pos

        if not len(obs_include) or 'joints' in obs_include:
            jnts = self.get_joint_angles()
            inds = self._obs_inds['joints']
            obs[inds[0]:inds[1]] = jnts

        if not len(obs_include) or 'end_effector' in obs_include:
            inds = self._obs_inds['end_effector']
            obs[inds[0]:inds[1]] = np.r_[self.get_ee_pos(), 
                                         self.get_ee_rot(),
                                         self.get_grip_jnts()[0]]

        for item in self.items:
            if not len(obs_include) or item[0] in obs_include:
                inds = self._obs_inds[item[0]]
                obs[inds[0]:inds[1]] = self.get_item_pos(item[0])

        return np.array(obs)


    def get_ee_pos(self, mujoco_frame=True):
        model = self.physics.model
        hand_ind = model.name2id('hand_site', 'site')
        pos = self.physics.data.xpos[hand_ind]
        if not mujoco_frame:
            pos[2] -= MUJOCO_MODEL_Z_OFFSET
            pos[0] -= MUJOCO_MODEL_X_OFFSET
        return pos 


    def get_ee_rot(self):
        model = self.physics.model
        hand_ind = model.name2id('hand_site', 'site')
        return self.physics.data.xquat[hand_ind].copy()


    def get_joint_angles(self):
        return self.physics.data.qpos[:9].copy()


    def get_arm_joint_angles(self):
        inds = [2,3,4,5,6]
        return self.physics.data.qpos[inds]


    def get_grip_jnts(self):
        inds = [7,8]
        return self.physics.data.qpos[inds]


    def get_base_pos(self, mujoco_frame=True):
        if 'hsr' in self._ind_cache:
            ind = self._ind_cache['hsr']
        else:
            ind = self.physics.model.name2id('hsr')
        pos = self.physics.model.body_pos[ind].copy()
        if not mujoco_frame:
            pos[0] -= MUJOCO_MODEL_X_OFFSET
            pos[2] -= MUJOCO_MODEL_Z_OFFSET
        return pos


    def _get_joints(self, act_index):
        if act_index in self._joint_map_cache:
            return self._joint_map_cache[act_index]

        res = []
        for name, attr in self.action_inds:
            inds = self.action_inds[name, attr]
            # Actions have a single gripper command, but MUJOCO uses two gripper joints
            if act_index in inds:
                if attr == 'gripper':
                    res = [('hand_l_proximal_joint', 1), ('hand_r_proximal_joint', 1)]
                elif attr == 'arm':
                    arm_ind = inds.tolist().index(act_index)
                    res = [(MUJOCO_JOINT_ORDER[arm_ind], 1)]

        self._joint_map_cache[act_index] = res
        return res


    def _calc_ik(self, pos, quat, check_limits=True):
        arm_jnts = self.get_arm_joint_angles()
        grip_jnts = self.get_grip_jnts()
        cmd = {'dpos': pos+np.array([0,0,MUJOCO_MODEL_Z_OFFSET]), 'rotation': [quat[1], quat[2], quat[3], quat[0]]}
        jnt_cmd = self._ikcontrol.joint_positions_for_eef_command(cmd, use_right)

        if jnt_cmd is None:
            print('Cannot complete action; ik will cause unstable control')
            return arm_jnts

        return jnt_cmd


    def step(self, action, mode=None, obs_include=None, debug=False):
        if mode is None:
            mode = self.ctrl_mode

        cmd = np.zeros((8))
        abs_cmd = np.zeros((8))

        r_grip = 0
        l_grip = 0

        if mode == 'joint_angle':
            for i in range(len(action)):
                jnts = self._get_joints(i)
                for jnt in jnts:
                    cmd_angle = jnt[1] * action[i]
                    ind = MUJOCO_JOINT_ORDER.index(jnt[0])
                    abs_cmd[ind] = cmd_angle
            r_grip = action[7]
            l_grip = action[15]

        elif mode == 'end_effector':
            cur_ee_pos = self.get_ee_pos()
            cur_ee_rot = self.get_ee_rot()
            target_ee_pos = cur_ee_pos + action[2:5]
            target_ee_pos[2] -= MUJOCO_MODEL_Z_OFFSET
            target_ee_rot = action[5:8]

            cmd = self._calc_ik(target_ee_pos, 
                                target_ee_rot)

            abs_cmd[:2] = action[:2]
            abs_cmd[2:7] = cmd
            grip = action[8]

        elif mode == 'end_effector_pos':
            cur_ee_pos = self.get_ee_pos()
            cur_ee_rot = self.get_ee_rot()
            target_ee_pos = cur_ee_pos + action[2:5]
            target_ee_pos[2] -= MUJOCO_MODEL_Z_OFFSET
            target_ee_rot = START_EE[5:9]

            cmd = self._calc_ik(target_ee_pos, 
                                target_ee_rot)

            abs_cmd[:2] = action[:2]
            abs_cmd[2:7] = cmd
            grip = action[8]

        elif mode == 'discrete_pos':
            raise NotImplementedError
            return self.get_obs(), \
                   self.compute_reward(), \
                   False, \
                   {}

        for t in range(MJC_DELTAS_PER_STEP / 4):
            # error = abs_cmd - self.physics.data.qpos[1:19]
            # cmd = 7e1 * error
            cmd = abs_cmd
            cmd[7] = grip
            self.physics.set_control(cmd)
            self.physics.step()

        return self.get_obs(obs_include=obs_include), \
               self.compute_reward(), \
               self.is_done(), \
               {}


    def sim_from_plan(self, plan, t):
        model  = self.physics.model
        xpos = model.body_pos.copy()
        xquat = model.body_quat.copy()
        param = list(plan.params.values())

        for param_name in plan.params:
            param = plan.params[param_name]
            if param.is_symbol(): continue
            if param._type != 'Robot':
                if param.name in self._ind_cache:
                    param_ind = self._ind_cache[param.name]
                else:
                    try:
                        param_ind = model.name2id(param.name, 'body')
                    except:
                        param_ind = -1
                    self._ind_cache[param.name] = -1
                if param_ind == -1: continue

                pos = param.pose[:, t]
                xpos[param_ind] = pos + np.array([MUJOCO_MODEL_X_OFFSET, 0, MUJOCO_MODEL_Z_OFFSET])
                if hasattr(param, 'rotation'):
                    rot = param.rotation[:, t]
                    mat = OpenRAVEBody.transform_from_obj_pose([0,0,0], rot)[:3,:3]
                    xquat[param_ind] = openravepy.quatFromRotationMatrix(mat)

        self.physics.data.xpos[:] = xpos[:]
        self.physics.data.xquat[:] = xquat[:]
        model.body_pos[:] = xpos[:]
        model.body_quat[:] = xquat[:]

        hsr = plan.params['hsr']
        self.physics.data.qpos[:2] = hsr.pose[:2, t]
        self.physics.data.qpos[2:7] = hsr.arm[:, t]
        self.physics.data.qpos[7] = hsr.gripper[:, t]

        self.physics.forward()
