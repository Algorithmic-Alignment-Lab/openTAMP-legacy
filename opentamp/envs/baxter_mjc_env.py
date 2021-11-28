# import matplotlib as mpl
# mpl.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from threading import Thread
import time
import traceback
import sys
import xml.etree.ElementTree as xml

from tkinter import TclError


import pybullet as P

from dm_control.mujoco import Physics
from dm_control.rl.control import PhysicsError
from dm_control.viewer import runtime
from dm_control.viewer import user_input
from dm_control.viewer import util

from gym import spaces
from gym.core import Env

import opentamp
from opentamp.envs import MJCEnv
from opentamp.util_classes.ik_controller import BaxterIKController
from opentamp.util_classes.mjc_xml_utils import *
from opentamp.util_classes import transform_utils as T
from core.util_classes.robots import *

BASE_VEL_XML = opentamp.__path__._last_parent_path[1] + '/opentamp'+'/robot_info/baxter_model.xml'
# ENV_XML = opentamp.__path__._last_parent_path[1] + '/opentamp'+'/robot_info/current_baxter_env.xml'
SPECIFIC_ENV_XML = opentamp.__path__._last_parent_path[1] + '/opentamp'+'/local/current_baxter_{0}.xml'

MUJOCO_JOINT_ORDER = ['right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2', 'right_gripper_l_finger_joint', 'right_gripper_r_finger_joint',\
                      'left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2', 'left_gripper_l_finger_joint', 'left_gripper_r_finger_joint']
                      

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


BAXTER_GAINS = {
    'left_s0': (5000., 0.01, 2.5),
    'left_s1': (5000., 50., 50.),
    'left_e0': (4000., 15., 1.),
    'left_e1': (1500, 30, 1.),
    'left_w0': (500, 10, 0.01),
    'left_w1': (500, 0.1, 0.01),
    'left_w2': (1000, 0.1, 0.01),
    'left_gripper_l_finger_joint': (1000, 0.1, 0.01),
    'left_gripper_r_finger_joint': (1000, 0.1, 0.01),

    'right_s0': (5000., 0.01, 2.5),
    'right_s1': (5000., 50., 50.),
    'right_e0': (4000., 15., 1.),
    'right_e1': (1500, 30, 1.),
    'right_w0': (500, 10, 0.01),
    'right_w1': (500, 0.1, 0.01),
    'right_w2': (1000, 0.1, 0.01),
    'right_gripper_l_finger_joint': (1000, 0.1, 0.01),
    'right_gripper_r_finger_joint': (1000, 0.1, 0.01),
}
ERROR_COEFF = 1e2
OPEN_VAL = 50
CLOSE_VAL = -50

_MAX_FRONTBUFFER_SIZE = 2048
_CAM_WIDTH = 200
_CAM_HEIGHT = 150

GRASP_THRESHOLD = np.array([0.05, 0.05, 0.025]) # np.array([0.01, 0.01, 0.03])
# MJC_TIME_DELTA = 0.002
# MJC_DELTAS_PER_STEP = int(1. // MJC_TIME_DELTA)
N_CONTACT_LIMIT = 12

# START_EE = [0.6, -0.5, 0.7, 0, 0, 1, 0, 0.6, 0.5, 0.7, 0, 0, 1, 0]
# START_EE = [0.6, -0.5, 0.9, 0, 0, 1, 0, 0.6, 0.5, 0.9, 0, 0, 1, 0]
START_EE = [0.6, -0.4, 0.9, 0, 1, 0, 0, 0.6, 0.4, 0.9, 0, 1, 0, 0]
DOWN_QUAT = [0, 0, 1, 0]
ALT_DOWN_QUAT = [0, 0.535, 0.845, 0]

CTRL_MODES = ['joint_angle', 'end_effector', 'end_effector_pos', 'discrete_pos', 'discrete']
DISCRETE_DISP = 0.02 # How far to move for each discrete action choice


class BaxterMJCEnv(MJCEnv):
    # metadata = {'render.modes': ['human', 'rgb_array', 'depth'], 'video.frames_per_second': 67}

    # def __init__(self, mode='end_effector', obs_include=[], items=[], include_files=[], include_items=[], im_dims=(_CAM_WIDTH, _CAM_HEIGHT), sim_freq=25, timestep=0.002, max_iter=250, view=False):
    #     assert mode in CTRL_MODES, 'Env mode must be one of {0}'.format(CTRL_MODES)
    #     self.ctrl_mode = mode
    #     self.active = True

    #     self.cur_time = 0.
    #     self.prev_time = 0.
    #     self.timestep = timestep
    #     self.sim_freq = sim_freq

    #     self.use_viewer = view
    #     self.use_glew = 'MUJOCO_GL' not in os.environ or os.environ['MUJOCO_GL'] != 'osmesa'
    #     self.obs_include = obs_include
    #     self._joint_map_cache = {}
    #     self._ind_cache = {}

    #     self.im_wid, self.im_height = im_dims
    #     self.items = items
    #     self._item_map = {item[0]: item for item in items}
    #     self.include_files = include_files
    #     self.include_items = include_items
    #     self._set_obs_info(obs_include)

    #     self._load_model()
    #     self._init_control_info()

    #     self._max_iter = max_iter
    #     self._cur_iter = 0

    #     if view:
    #         self._launch_viewer(_CAM_WIDTH, _CAM_HEIGHT)
    #     else:
    #         self._viewer = None


    def _init_control_info(self):
        self.ctrl_data = {}
        for joint in BAXTER_GAINS:
            self.ctrl_data[joint] = {
                'prev_err': 0.,
                'cp': 0.,
                'cd': 0.,
                'ci': 0.,
            }

        self.ee_ctrl_data = {}
        for joint in BAXTER_GAINS:
            self.ee_ctrl_data[joint] = {
                'prev_err': 0.,
                'cp': 0.,
                'cd': 0.,
                'ci': 0.,
            }

        
        if not P.getConnectionInfo()['isConnected']:
            P.connect(P.DIRECT)

        self.geom = Baxter()
        self.geom.setup()
        self._jnt_inds = {}
        for key, jnts in self.geom.jnt_names.items():
            self._jnt_inds[key] = [self.physics.model.name2id(jnt, 'joint') for jnt in jnts]

        # Start joints with grippers pointing downward
        self.physics.data.qpos[1:8] = self._calc_ik(START_EE[:3], START_EE[3:7], 'right', False)
        self.physics.data.qpos[10:17] = self._calc_ik(START_EE[7:10], START_EE[10:14], 'left', False)
        self.physics.forward()

        self.action_inds = {
            ('baxter', 'rArmPose'): np.array(list(range(7))),
            ('baxter', 'rGripper'): np.array([7]),
            ('baxter', 'lArmPose'): np.array(list(range(8, 15))),
            ('baxter', 'lGripper'): np.array([15]),
        }


    def _load_model(self):
        xmlpath = SPECIFIC_ENV_XML.format(self.xmlid)
        generate_xml(BASE_VEL_XML, xmlpath, self.items, self.include_files, self.include_items, timestep=self.timestep)
        self.physics = Physics.from_xml_path(xmlpath)


    # def _launch_viewer(self, width, height, title='Main'):
    #     self._matplot_view_thread = None
    #     if self.use_glew:
    #         self._renderer = renderer.NullRenderer()
    #         self._render_surface = None
    #         self._viewport = renderer.Viewport(width, height)
    #         self._window = gui.RenderWindow(width, height, title)
    #         self._viewer = viewer.Viewer(
    #             self._viewport, self._window.mouse, self._window.keyboard)
    #         self._viewer_layout = views.ViewportLayout()
    #         self._viewer.render()
    #     else:
    #         self._viewer = None
    #         self._matplot_im = None
    #         self._run_matplot_view()


    # def _reload_viewer(self):
    #     if self._viewer is None or not self.use_glew: return

    #     if self._render_surface:
    #       self._render_surface.free()

    #     if self._renderer:
    #       self._renderer.release()

    #     self._render_surface = render.Renderer(
    #         max_width=_MAX_FRONTBUFFER_SIZE, max_height=_MAX_FRONTBUFFER_SIZE)
    #     self._renderer = renderer.OffScreenRenderer(
    #         self.physics.model, self._render_surface)
    #     self._renderer.components += self._viewer_layout
    #     self._viewer.initialize(
    #         self.physics, self._renderer, touchpad=False)
    #     self._viewer.zoom_to_scene()


    # def _render_viewer(self, pixels):
    #     if self.use_glew:
    #         with self._window._context.make_current() as ctx:
    #             ctx.call(
    #                 self._window._update_gui_on_render_thread, self._window._context.window, pixels)
    #         self._window._mouse.process_events()
    #         self._window._keyboard.process_events()
    #     else:
    #         if self._matplot_im is not None:
    #             self._matplot_im.set_data(pixels)
    #             plt.draw()


    # def _run_matplot_view(self):
    #     self._matplot_view_thread = Thread(target=self._launch_matplot_view)
    #     self._matplot_view_thread.daemon = True
    #     self._matplot_view_thread.start()


    # def _launch_matplot_view(self):
    #     try:
    #         self._matplot_im = plt.imshow(self.render(view=False))
    #         plt.show()
    #     except TclError:
    #         print('\nCould not find display to launch viewer (this does not affect the ability to render images)\n')


    def _set_obs_info(self, obs_include):
        self._obs_inds = {}
        self._obs_shape = {}
        ind = 0
        if 'overhead_image' in obs_include or not len(obs_include):
            self._obs_inds['overhead_image'] = (ind, ind+3*self.im_wid*self.im_height)
            self._obs_shape['overhead_image'] = (self.im_height, self.im_wid, 3)
            ind += 3*self.im_wid*self.im_height

        if 'forward_image' in obs_include or not len(obs_include):
            self._obs_inds['forward_image'] = (ind, ind+3*self.im_wid*self.im_height)
            self._obs_shape['forward_image'] = (self.im_height, self.im_wid, 3)
            ind += 3*self.im_wid*self.im_height

        if 'right_image' in obs_include or not len(obs_include):
            self._obs_inds['right_image'] = (ind, ind+3*self.im_wid*self.im_height)
            self._obs_shape['right_image'] = (self.im_height, self.im_wid, 3)
            ind += 3*self.im_wid*self.im_height

        if 'left_image' in obs_include or not len(obs_include):
            self._obs_inds['left_image'] = (ind, ind+3*self.im_wid*self.im_height)
            self._obs_shape['left_image'] = (self.im_height, self.im_wid, 3)
            ind += 3*self.im_wid*self.im_height

        if 'joints' in obs_include or not len(obs_include):
            self._obs_inds['joints'] = (ind, ind+18)
            self._obs_shape['joints'] = (18,)
            ind += 18

        if 'end_effector' in obs_include or not len(obs_include):
            self._obs_inds['end_effector'] = (ind, ind+16)
            self._obs_shape['end_effector'] = (16,)
            ind += 16

        for item, xml, info in self.items:
            if item in obs_include or not len(obs_include):
                self._obs_inds[item] = (ind, ind+3) # Only store 3d Position
                self._obs_shape[item] = (3,)
                ind += 3

        self.dO = ind
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(ind,), dtype='float32')
        return ind


    def get_obs(self, obs_include=None, view=False):
        obs = np.zeros(self.dO)
        if obs_include is None:
            obs_include = self.obs_include

        if self.load_render:
            if not len(obs_include) or 'overhead_image' in obs_include:
                pixels = self.render(height=self.im_height, width=self.im_wid, camera_id=0, view=view)
                view = False
                inds = self._obs_inds['overhead_image']
                obs[inds[0]:inds[1]] = pixels.flatten()

            if not len(obs_include) or 'forward_image' in obs_include:
                pixels = self.render(height=self.im_height, width=self.im_wid, camera_id=1, view=view)
                view = False
                inds = self._obs_inds['forward_image']
                obs[inds[0]:inds[1]] = pixels.flatten()

            if not len(obs_include) or 'right_image' in obs_include:
                pixels = self.render(height=self.im_height, width=self.im_wid, camera_id=2, view=view)
                view = False
                inds = self._obs_inds['right_image']
                obs[inds[0]:inds[1]] = pixels.flatten()

            if not len(obs_include) or 'left_image' in obs_include:
                pixels = self.render(height=self.im_height, width=self.im_wid, camera_id=3, view=view)
                view = False
                inds = self._obs_inds['left_image']
                obs[inds[0]:inds[1]] = pixels.flatten()

        if not len(obs_include) or 'joints' in obs_include:
            jnts = self.get_joint_angles()
            inds = self._obs_inds['joints']
            obs[inds[0]:inds[1]] = jnts

        if not len(obs_include) or 'end_effector' in obs_include:
            grip_jnts = self.get_gripper_joint_angles()
            inds = self._obs_inds['end_effector']
            obs[inds[0]:inds[1]] = np.r_[self.get_right_ee_pos(), 
                                         self.get_right_ee_rot(),
                                         grip_jnts[0],
                                         self.get_left_ee_pos(),
                                         self.get_left_ee_rot(),
                                         grip_jnts[1]]

        for item in self.items:
            if not len(obs_include) or item[0] in obs_include:
                inds = self._obs_inds[item[0]]
                obs[inds[0]:inds[1]] = self.get_item_pos(item[0])

        return np.array(obs)


    # def get_obs_types(self):
    #     return self._obs_inds.keys()


    # def get_obs_inds(self, obs_type):
    #     if obs_type not in self._obs_inds:
    #         raise KeyError('{0} is not a valid observation for this environment. Valid options: {1}'.format(obs_type, self.get_obs_types()))
    #     return self._obs_inds[obs_type]


    # def get_obs_shape(self, obs_type):
    #     if obs_type not in self._obs_inds:
    #         raise KeyError('{0} is not a valid observation for this environment. Valid options: {1}'.format(obs_type, self.get_obs_types()))
    #     return self._obs_shape[obs_type]


    # def get_obs_data(self, obs, obs_type):
    #     if obs_type not in self._obs_inds:
    #         raise KeyError('{0} is not a valid observation for this environment. Valid options: {1}'.format(obs_type, self.get_obs_types()))
    #     return obs[self._obs_inds[obs_type]].reshape(self._obs_shape[obs_type])


    def get_arm_section_inds(self, section_name):
        inds = self.get_obs_inds('joints')
        if section_name == 'lArmPose':
            return inds[9:16]
        if section_name == 'lGripper':
            return inds[16:]
        if section_name == 'rArmPose':
            return inds[:7]
        if section_name == 'rGripper':
            return inds[7:8]


    def get_left_ee_pos(self, mujoco_frame=True):
        model = self.physics.model
        ind = model.name2id('left_gripper', 'body')
        pos = self.physics.data.xpos[ind].copy()
        if not mujoco_frame:
            pos[2] -= MUJOCO_MODEL_Z_OFFSET
            pos[0] -= MUJOCO_MODEL_X_OFFSET
        return pos 


    def get_right_ee_pos(self, mujoco_frame=True):
        model = self.physics.model
        ind = model.name2id('right_gripper', 'body')
        pos = self.physics.data.xpos[ind].copy()
        if not mujoco_frame:
            pos[2] -= MUJOCO_MODEL_Z_OFFSET
            pos[0] -= MUJOCO_MODEL_X_OFFSET
        return pos 


    def get_left_ee_rot(self, mujoco_frame=True):
        model = self.physics.model
        ind = model.name2id('left_gripper', 'body')
        quat = self.physics.data.xquat[ind].copy()
        return quat 


    def get_right_ee_rot(self, mujoco_frame=True):
        model = self.physics.model
        ind = model.name2id('right_gripper', 'body')
        quat = self.physics.data.xquat[ind].copy()
        return quat


    def get_item_pos(self, name, mujoco_frame=True, rot=False):
        if name.find('ee_pos') >= 0:
            if name.find('left') >= 0:
                return self.get_left_ee_pos(mujoco_frame)
            elif name.find('right') >= 0:
                return self.get_right_ee_pos(mujoco_frame)
        
        if name.find('ee_quat') >= 0:
            if name.find('left') >= 0:
                return self.get_left_ee_rot(mujoco_frame)
            elif name.find('right') >= 0:
                return self.get_right_ee_rot(mujoco_frame)

        return super(BaxterMJCEnv, self).get_item_pos(name, mujoco_frame, rot)


    def get_item_rot(self, name, mujoco_frame=True, to_euler=False):
        if name.find('ee_quat') >= 0:
            if name.find('left') >= 0:
                rpt = self.get_left_ee_rot(mujoco_frame)
            elif name.find('right') >= 0:
                rot = self.get_right_ee_rot(mujoco_frame)
            if to_euler: rot = T.quaternion_to_euler(rot, order='xyzw')
            return rot
        
        return super(BaxterMJCEnv, self).get_item_rot(name, mujoco_frame, to_euler)

    #def get_item_pos(self, name, mujoco_frame=True):
    #    model = self.physics.model
    #    try:
    #        ind = model.name2id(name, 'joint')
    #        adr = model.jnt_qposadr[ind]
    #        pos = self.physics.data.qpos[adr:adr+3].copy()
    #    except Exception as e:
    #        try:
    #            item_ind = model.name2id(name, 'body')
    #            pos = self.physics.data.xpos[item_ind].copy()
    #        except:
    #            item_ind = -1

    #    if not mujoco_frame:
    #        pos[2] -= MUJOCO_MODEL_Z_OFFSET
    #        pos[0] -= MUJOCO_MODEL_X_OFFSET
    #    return pos


    #def set_item_pos(self, name, pos, mujoco_frame=True, forward=True):
    #    if not mujoco_frame:
    #        pos = [pos[0]+MUJOCO_MODEL_X_OFFSET, pos[1], pos[2]+MUJOCO_MODEL_Z_OFFSET]

    #    item_type = 'joint'
    #    try:
    #        ind = self.physics.model.name2id(name, 'joint')
    #        adr = self.physics.model.jnt_qposadr[ind]
    #        old_pos = self.physics.data.qpos[adr:adr+3]
    #        self.physics.data.qpos[adr:adr+3] = pos
    #    except Exception as e:
    #        try:
    #            ind = self.physics.model.name2id(name, 'body')
    #            old_pos = self.physics.data.xpos[ind]
    #            self.physics.data.xpos[ind] = pos
    #            # self.physics.model.body_pos[ind] = pos
    #            # old_pos = self.physics.model.body_pos[ind]
    #            item_type = 'body'
    #        except:
    #            item_type = 'unknown'

    #    if forward:
    #        self.physics.forward()
    #    # try:
    #    #     self.physics.forward()
    #    # except PhysicsError as e:
    #    #     print e
    #    #     traceback.print_exception(*sys.exc_info())
    #    #     print '\n\n\n\nERROR IN SETTING {0} POSE.\nPOSE TYPE: {1}.\nRESETTING SIMULATION.\n\n\n\n'.format(name, item_type)
    #    #     qpos = self.physics.data.qpos.copy()
    #    #     xpos = self.physics.data.xpos.copy()
    #    #     if item_type == 'joint':
    #    #         qpos[adr:adr+3] = old_pos
    #    #     elif item_type == 'body':
    #    #         xpos[ind] = old_pos
    #    #     self.physics.reset()
    #    #     self.physics.data.qpos[:] = qpos[:]
    #    #     self.physics.data.xpos[:] = xpos[:]
    #    #     self.physics.forward()


    ## def get_pos_from_label(self, label, mujoco_frame=True):
    ##     if label in self._item_map:
    ##         return self.get_item_pos(label, mujoco_frame)
    ##     return None


    def get_joint_angles(self):
        return self.physics.data.qpos[1:19].copy()


    def get_arm_joint_angles(self):
        inds = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16]
        return self.physics.data.qpos[inds].copy()


    def set_arm_joint_angles(self, jnts):
        inds = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16]
        self.physics.data.qpos[inds] = jnts
        self.physics.data.qvel[inds] = 0
        self.physics.data.qacc[inds] = 0
        self.physics.forward()


    def set_gripper_joint_angles(self, jnts):
        inds = [8, 17]
        self.physics.data.qpos[inds] = jnts
        self.physics.data.qvel[inds] = 0
        self.physics.data.qacc[inds] = 0
        self.physics.forward()


    def get_gripper_joint_angles(self):
        inds = [8, 17]
        return self.physics.data.qpos[inds]


    def _get_joints(self, act_index):
        if act_index in self._joint_map_cache:
            return self._joint_map_cache[act_index]

        res = []
        for name, attr in self.action_inds:
            inds = self.action_inds[name, attr]
            # Actions have a single gripper command, but MUJOCO uses two gripper joints
            if act_index in inds:
                if attr == 'lGripper':
                    res = [('left_gripper_l_finger_joint', 1), ('left_gripper_r_finger_joint', -1)]
                elif attr == 'rGripper':
                    res = [('right_gripper_r_finger_joint', 1), ('right_gripper_l_finger_joint', -1)]
                elif attr == 'lArmPose':
                    arm_ind = inds.tolist().index(act_index)
                    res = [(MUJOCO_JOINT_ORDER[9+arm_ind], 1)]
                elif attr == 'rArmPose':
                    arm_ind = inds.tolist().index(act_index)
                    res = [(MUJOCO_JOINT_ORDER[arm_ind], 1)]

        self._joint_map_cache[act_index] = res
        return res


    def get_action_meanings(self):
        # For discrete action mode
        return ['NOOP', 'RIGHT_EE_FORWARD', 'RIGHT_EE_BACK', 'RIGHT_EE_LEFT', 'RIGHT_EE_RIGHT',
                'RIGHT_EE_UP', 'RIGHT_EE_DOWN', 'RIGHT_EE_OPEN', 'RIGHT_EE_CLOSE',
                'LEFT_EE_FORWARD', 'LEFT_EE_BACK', 'LEFT_EE_LEFT', 'LEFT_EE_RIGHT',
                'LEFT_EE_UP', 'LEFT_EE_DOWN', 'LEFT_EE_OPEN', 'LEFT_EE_CLOSE']


    def move_right_gripper_forward(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[0] = DISCRETE_DISP
        return self.step(act, mode='end_effector_pos')


    def move_right_gripper_backward(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[0] = -DISCRETE_DISP
        return self.step(act, mode='end_effector_pos')


    def move_right_gripper_left(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[1] = DISCRETE_DISP
        return self.step(act, mode='end_effector_pos')


    def move_right_gripper_right(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[1] = -DISCRETE_DISP
        return self.step(act, mode='end_effector_pos')


    def move_right_gripper_up(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[2] = DISCRETE_DISP
        return self.step(act, mode='end_effector_pos')


    def move_right_gripper_down(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[2] = -DISCRETE_DISP
        return self.step(act, mode='end_effector_pos')

    def open_right_gripper(self):
        act = np.zeros(8)
        act[3] = DISCRETE_DISP
        return self.step(act, mode='end_effector_pos')


    def close_right_gripper(self):
        act = np.zeros(8)
        act[3] = 0
        return self.step(act, mode='end_effector_pos')


    def move_left_gripper_forward(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[4] = DISCRETE_DISP
        return self.step(act, mode='end_effector_pos')


    def move_left_gripper_backward(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[4] = -DISCRETE_DISP
        return self.step(act, mode='end_effector_pos')


    def move_left_gripper_left(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[5] = DISCRETE_DISP
        return self.step(act, mode='end_effector_pos')


    def move_left_gripper_right(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[5] = -DISCRETE_DISP
        return self.step(act, mode='end_effector_pos')


    def move_left_gripper_up(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[6] = DISCRETE_DISP
        return self.step(act, mode='end_effector_pos')


    def move_left_gripper_down(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[6] = -DISCRETE_DISP
        return self.step(act, mode='end_effector_pos')

    def open_left_gripper(self):
        act = np.zeros(8)
        act[7] = 0.02
        return self.step(act, mode='end_effector_pos')


    def close_left_gripper(self):
        act = np.zeros(8)
        act[7] = 0
        return self.step(act, mode='end_effector_pos')


    def _step_joint(self, joint, error):
        ctrl_data = self.ctrl_data[joint]
        gains = BAXTER_GAINS[joint]
        dt = MJC_TIME_DELTA
        de = error - ctrl_data['prev_err']
        ctrl_data['cp'] = error
        ctrl_data['cd'] = de / dt
        ctrl_data['ci'] += error * dt
        ctrl_data['prev_err'] = error
        return gains[0] * ctrl_data['cp'] + \
               gains[1] * ctrl_data['cd'] + \
               gains[2] * ctrl_data['ci']


    def _clip_joint_angles(self, r_jnts, r_grip, l_jnts, l_grip):
        DOF_limits = self._ikbody.env_body.GetDOFLimits()
        left_DOF_limits = (DOF_limits[0][2:9]+0.001, DOF_limits[1][2:9]-0.001)
        right_DOF_limits = (DOF_limits[0][10:17]+0.001, DOF_limits[1][10:17]-0.001)

        if r_grip[0] < 0:
            r_grip[0] = 0
        if r_grip[0] > 0.02:
            r_grip[0] = 0.02
        if l_grip[0] < 0:
            l_grip[0] = 0
        if l_grip[0] > 0.02:
            l_grip[0] = 0.02

        for i in range(7):
            if l_jnts[i] < left_DOF_limits[0][i]:
                l_jnts[i] = left_DOF_limits[0][i]
            if l_jnts[i] > left_DOF_limits[1][i]:
                l_jnts[i] = left_DOF_limits[1][i]
            if r_jnts[i] < right_DOF_limits[0][i]:
                r_jnts[i] = right_DOF_limits[0][i]
            if r_jnts[i] > right_DOF_limits[1][i]:
                r_jnts[i] = right_DOF_limits[1][i]


    def _calc_ik(self, pos, quat, arm, check_limits=True):
        lb, ub = self.geom.get_arm_bnds()
        ranges = (np.array(ub) - np.array(lb)).tolist()
        jnt_ids = sorted(self.geom.get_free_inds())
        jnts = P.getJointStates(self.geom.id, jnt_ids)
        rest_poses = []
        arm_inds = self.geom.get_arm_inds(arm)
        arm_jnts = self.geom.jnt_names[arm]
        cur_jnts = self.get_joints(arm_jnts, vec=True)
        for ind, jnt_id in enumerate(jnt_ids):
            if jnt_id in arm_inds:
                rest_poses.append(cur_jnts[arm_inds.index(jnt_id)])
            else:
                rest_poses.append(jnts[ind][0])
        manip_id = self.geom.get_ee_link(arm)
        damp = (0.1 * np.ones(len(jnt_ids))).tolist()
        joint_pos = P.calculateInverseKinematics(self.geom.id,
                                                 manip_id,
                                                 pos,
                                                 quat,
                                                 lowerLimits=lb,
                                                 upperLimits=ub,
                                                 jointRanges=ranges,
                                                 restPoses=rest_poses,
                                                 jointDamping=damp,
                                                 maxNumIterations=128)
        inds = list(self.geom.get_free_inds(arm))
        joint_pos = np.array(joint_pos)[inds].tolist()
        return joint_pos

    #def _calc_ik(self, pos, quat, use_right=True, check_limits=True):
    #    arm_jnts = self.get_arm_joint_angles()
    #    grip_jnts = self.get_gripper_joint_angles()

    #    cmd = {'dpos': pos+np.array([0,0,MUJOCO_MODEL_Z_OFFSET]), 'rotation': [quat[1], quat[2], quat[3], quat[0]]}
    #    jnt_cmd = self._ikcontrol.joint_positions_for_eef_command(cmd, use_right)

    #    if use_right:
    #        if jnt_cmd is None:
    #            print('Cannot complete action; ik will cause unstable control')
    #            return arm_jnts[:7]
    #    else:
    #        if jnt_cmd is None:
    #            print('Cannot complete action; ik will cause unstable control')
    #            return arm_jnts[7:]

    #    return jnt_cmd


    #def _check_ik(self, pos, quat=DOWN_QUAT, use_right=True):
    #    cmd = {'dpos': pos+np.array([0,0,MUJOCO_MODEL_Z_OFFSET]), 'rotation': [quat[1], quat[2], quat[3], quat[0]]}
    #    jnt_cmd = self._ikcontrol.joint_positions_for_eef_command(cmd, use_right)

    #    return jnt_cmd is not None


    def step(self, action, mode=None, obs_include=None, gen_obs=True, view=False, debug=False):
        start_t = time.time()
        if mode is None:
            mode = self.ctrl_mode

        cmd = np.zeros((18))
        abs_cmd = np.zeros((18))

        r_grip = 0
        l_grip = 0

        cur_left, cur_right = self.get_attr('baxter', 'left'), self.get_attr('baxter', 'right')
        if mode == 'joint_angle':
            if type(action) is dict:
                left = cur_left + action.get('left', np.zeros(7))
                right = cur_right + action.get('right', np.zeros(7))
                r_grip = action.get('right_gripper', 0)
                l_grip = action.get('left_gripper', 0)
                abs_cmd[:7] = right
                abs_cmd[9:16] = left
            else:
                for i in range(len(action)):
                    jnts = self._get_joints(i)
                    for jnt in jnts:
                        cmd_angle = jnt[1] * action[i]
                        ind = MUJOCO_JOINT_ORDER.index(jnt[0])
                        abs_cmd[ind] = cmd_angle
                r_grip = action[7]
                l_grip = action[15]

        elif mode == 'end_effector':
            # Action Order: ee_right_pos, ee_right_quat, ee_right_grip, ee_left_pos, ee_left_quat, ee_left_grip
            cur_right_ee_pos = self.get_right_ee_pos()
            cur_right_ee_rot = self.get_right_ee_rot()
            cur_left_ee_pos = self.get_left_ee_pos()
            cur_left_ee_rot = self.get_left_ee_rot()
 
            if type(action) is dict:
                right_ee_cmd, left_ee_cmd = action['right_ee_pos'], action['left_ee_pos']
                right_ee_rot, left_ee_rot = action['right_ee_rot'], action['left_ee_rot']
                r_grip, l_grip = action['right_gripper'], action['left_gripper']
            else:
                right_ee_cmd, left_ee_cmd = action[:3], action[8:11]
                right_ee_rot, left_ee_rot = action[3:7], action[11:15]
                r_grip, l_grip = action[7], action[11]

            target_right_ee_pos = cur_right_ee_pos + right_ee_cmd 
            target_right_ee_rot = right_ee_rot # cur_right_ee_rot + action[3:7]
            target_left_ee_pos = cur_left_ee_pos + left_ee_cmd
            target_left_ee_rot = left_ee_rot # cur_left_ee_rot + action[11:15]

            # target_right_ee_rot /= np.linalg.norm(target_right_ee_rot)
            # target_left_ee_rot /= np.linalg.norm(target_left_ee_rot)

            right_cmd = self._calc_ik(target_right_ee_pos, 
                                      target_right_ee_rot, 
                                      'right')

            left_cmd = self._calc_ik(target_left_ee_pos, 
                                     target_left_ee_rot, 
                                     'left')

            abs_cmd[:7] = right_cmd
            abs_cmd[9:16] = left_cmd
            r_grip = action[7]
            l_grip = action[15]

        elif mode == 'end_effector_pos':
            # Action Order: ee_right_pos, ee_right_quat, ee_right_grip, ee_left_pos, ee_left_quat, ee_left_grip
            cur_right_ee_pos = self.get_right_ee_pos()
            cur_left_ee_pos = self.get_left_ee_pos()

            if type(action) is dict:
                right_ee_cmd, left_ee_cmd = action.get('right_ee_pos', (0,0,0)), action.get('left_ee_pos', (0,0,0))
                r_grip, l_grip = action.get('right_gripper', 0), action.get('left_gripper', 0)
            else:
                right_ee_cmd, left_ee_cmd = action[:3], action[4:7]
                r_grip, l_grip = action[3], action[7]

            target_right_ee_pos = cur_right_ee_pos + right_ee_cmd
            target_right_ee_rot = START_EE[3:7]
            target_left_ee_pos = cur_left_ee_pos + left_ee_cmd
            target_left_ee_rot = START_EE[10:14]


            right_cmd = self._calc_ik(target_right_ee_pos, 
                                      target_right_ee_rot, 
                                      'right')

            left_cmd = self._calc_ik(target_left_ee_pos, 
                                     target_left_ee_rot, 
                                     'left')

            abs_cmd[:7] = right_cmd
            abs_cmd[9:16] = left_cmd

        elif mode == 'discrete_pos':
            if action == 1: return self.move_right_gripper_forward()
            if action == 2: return self.move_right_gripper_backward()
            if action == 3: return self.move_right_gripper_left()
            if action == 4: return self.move_right_gripper_right()
            if action == 5: return self.move_right_gripper_up()
            if action == 6: return self.move_right_gripper_down()
            if action == 7: return self.open_right_gripper()
            if action == 8: return self.close_right_gripper()

            if action == 9: return self.move_left_gripper_forward()
            if action == 10: return self.move_left_gripper_backward()
            if action == 11: return self.move_left_gripper_left()
            if action == 12: return self.move_left_gripper_right()
            if action == 13: return self.move_left_gripper_up()
            if action == 14: return self.move_left_gripper_down()
            if action == 15: return self.open_left_gripper()
            if action == 16: return self.close_left_gripper()
            return self.get_obs(view=view), \
                   self.compute_reward(), \
                   False, \
                   {}

        for t in range(self.sim_freq): # range(int(1/(4*self.timestep))):
            error = abs_cmd - self.physics.data.qpos[1:19]
            cmd = ERROR_COEFF * error
            # cmd[cmd > 0.25] = 0.25
            # cmd[cmd < -0.25] = -0.25
            cmd[7] = OPEN_VAL if r_grip > 0.0165 else CLOSE_VAL
            cmd[8] = cmd[7]
            cmd[16] = OPEN_VAL if l_grip > 0.0165 else CLOSE_VAL
            cmd[17] = cmd[16]

            # cmd[7] = 0.03 if r_grip > 0.0165 else -0.01
            # cmd[8] = -cmd[7]
            # cmd[16] = 0.03 if l_grip > 0.0165 else -0.01
            # cmd[17] = -cmd[16]
            cur_state = self.physics.data.qpos.copy()
            self.physics.set_control(cmd)
            try:
                self.physics.step()
            except PhysicsError as e:
                traceback.print_exception(*sys.exc_info())
                print('\n\nERROR IN PHYSICS SIMULATION; RESETTING ENV.\n\n')
                self.physics.reset()
                self.physics.data.qpos[:] = cur_state[:]
                self.physics.forward()

        # self.render(camera_id=1, view=True)
        if not gen_obs and not view: return
        return self.get_obs(obs_include=obs_include, view=view), \
               self.compute_reward(), \
               self.is_done(), \
               {}


    # def compute_reward(self):
    #     return 0


    # def is_done(self):
    #     return self._cur_iter >= self._max_iter


    # def render(self, mode='rgb_array', height=0, width=0, camera_id=0,
    #            overlays=(), depth=False, scene_option=None, view=True):
    #     # Make friendly with dm_control or gym interface
    #     depth = depth or mode == 'depth_array'
    #     view = view or mode == 'human'
    #     if height == 0: height = self.im_height
    #     if width == 0: width = self.im_wid

    #     pixels = self.physics.render(height, width, camera_id, overlays, depth, scene_option)
    #     if view and self.use_viewer:
    #         self._render_viewer(pixels)

    #     return pixels


    def reset(self):
        # self._cur_iter = 0
        # self.physics.reset()
        # self._reload_viewer()
        # self.ctrl_data = {}
        # self.cur_time = 0.
        # self.prev_time = 0.
        self._cur_iter = 0
        self.physics.data.qpos[1:8] = self._calc_ik(START_EE[:3], START_EE[3:7], 'right', False)
        self.physics.data.qpos[10:17] = self._calc_ik(START_EE[7:10], START_EE[10:14], 'left', False)

        obs = super(BaxterMJCEnv, self).reset()
        for joint in BAXTER_GAINS:
            self.ctrl_data[joint] = {
                'prev_err': 0.,
                'cp': 0.,
                'cd': 0.,
                'ci': 0.,
            }
        return obs


    @classmethod
    def init_from_plan(cls, plan, view=True):
        items = []
        for p in list(plan.params.values()):
            if p.is_symbol(): continue
            param_xml = get_param_xml(p)
            if param_xml is not None:
                items.append(param_xml)
        return cls.__init__(view, items)


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

        baxter = plan.params['baxter']
        self.physics.data.qpos[1:8] = baxter.rArmPose[:, t]
        self.physics.data.qpos[8] = baxter.rGripper[:, t]
        self.physics.data.qpos[9] = -baxter.rGripper[:, t]
        self.physics.data.qpos[10:17] = baxter.lArmPose[:, t]
        self.physics.data.qpos[17] = baxter.lGripper[:, t]
        self.physics.data.qpos[18] = -baxter.lGripper[:, t]

        self.physics.forward()


    def mp_state_from_sim(self, plan):
        X = np.zeros(plan.symbolic_bound)
        for param_name, attr_name in plan.state_inds:
            inds = plan.state_inds[param_name, attr_name]
            if param_name in plan.params:
                param = plan.params[param_name]
                if param_name == 'baxter':
                    pass
                elif not param.is_symbol():
                    if attr_name == 'pose':
                        X[inds] = self.get_item_pos(param_name)
                    elif attr_name == 'rotation':
                        X[inds] = self.get_item_rot(param_name, convert_to_euler=True)




    def jnt_ctrl_from_plan(self, plan, t):
        baxter = plan.params['baxter']
        lArmPose = baxter.lArmPose[:, t]
        lGripper = baxter.lGripper[:, t]
        rArmPose = baxter.rArmPose[:, t]
        rGripper = baxter.rGripper[:, t]
        ctrl = np.r_[rArmPose, rGripper, -rGripper, lArmPose, lGripper, -lGripper]
        return self.step(joint_angles=ctrl)


    def run_plan(self, plan):
        self.reset()
        obs = []
        for t in range(plan.horizon):
            obs.append(self.jnt_ctrl_from_plan(plan, t))

        return obs


    def _move_to(self, pos, gripper1, gripper2, left=True, view=False):
        observations = [self.get_obs(view=False)]
        if not self._check_ik(pos, quat=DOWN_QUAT, use_right=not left):
            return observations

        limit1 = np.array([0.01, 0.01, 0.035])
        limit2 = np.array([0.005, 0.005, 0.01])
        ee_above = pos + np.array([0.0, 0, 0.2])
        ee_above[2] = np.minimum(ee_above[2], 0.6)

        inds = ([[4,5,6]], 7) if left else ([[0,1,2]], 3)
        aux_inds = ([[0,1,2]], 3) if left else ([[4,5,6]], 7)
        ee_pos = self.get_left_ee_pos() if left else self.get_right_ee_pos()
        aux_ee_pos = [0.6, -0.5, 0.2] if left else [0.6, 0.5, 0.2]
        end_ee_pos = [0.6, 0.5, 0.2] if left else [0.6, -0.5, 0.2]

        gripper_angle = self.get_gripper_joint_angles()[1] if left else self.get_gripper_joint_angles()[0]

        max_iter = 20
        cur_iter = 0
        while np.any(np.abs(ee_pos - ee_above) > limit1) or np.abs(gripper_angle < gripper1*0.015):
            next_ee_cmd = np.minimum(np.maximum(ee_above - ee_pos, -0.2*np.ones((3,))), 0.2*np.ones((3,)))
            # next_ee_cmd = ee_above - ee_pos
            # next_ee_cmd[2] += 0.03
            next_ee_cmd[0] = next_ee_cmd[0] if ee_above[0] > 0.5 else next_ee_cmd[0]
            next_cmd = np.zeros((8,))
            next_cmd[inds[0]] = next_ee_cmd
            next_cmd[inds[1]] = gripper1

            cur_aux_ee_pos = self.get_right_ee_pos() if left else self.get_left_ee_pos()
            next_cmd[aux_inds[0]] = aux_ee_pos - cur_aux_ee_pos

            obs, _, _, _ = self.step(next_cmd, mode='end_effector_pos', view=view)
            # observations.append((next_cmd, obs))
            observations.append(obs)
            ee_pos = self.get_left_ee_pos() if left else self.get_right_ee_pos()
            gripper_angle = self.get_gripper_joint_angles()[1] if left else self.get_gripper_joint_angles()[0]
            cur_iter += 1
            if cur_iter > max_iter and np.all(np.abs(ee_pos - ee_above)[:2] < 0.05): break
            if cur_iter > 2*max_iter: break

        next_cmd = np.zeros((8,))
        next_cmd[inds[1]] = gripper1
        obs, _, _, _ = self.step(next_cmd, mode='end_effector_pos', view=view)
        # observations.append((next_cmd, obs))
        observations.append(obs)

        max_iter = 15
        cur_iter = 0
        ee_pos = self.get_left_ee_pos() if left else self.get_right_ee_pos()
        gripper_angle = self.get_gripper_joint_angles()[1] if left else self.get_gripper_joint_angles()[0]
        while np.any(np.abs(ee_pos - pos) > limit2):
            next_ee_cmd = np.minimum(np.maximum(pos - ee_pos, -0.05*np.ones((3,))), 0.05*np.ones((3,)))
            # next_ee_cmd = pos - ee_pos
            next_ee_cmd[2] += 0.01
            next_ee_cmd[0] = next_ee_cmd[0] - 0.01 if pos[0] > 0.5 else next_ee_cmd[0] - 0.01
            next_cmd = np.zeros((8,))
            next_cmd[inds[0]] = next_ee_cmd
            next_cmd[inds[1]] = gripper1

            cur_aux_ee_pos = self.get_right_ee_pos() if left else self.get_left_ee_pos()
            next_cmd[aux_inds[0]] = aux_ee_pos - cur_aux_ee_pos

            obs, _, _, _ = self.step(next_cmd, mode='end_effector_pos', view=view)
            # observations.append((next_cmd, obs))
            observations.append(obs)
            ee_pos = self.get_left_ee_pos() if left else self.get_right_ee_pos()
            gripper_angle = self.get_gripper_joint_angles()[1] if left else self.get_gripper_joint_angles()[0]
            cur_iter += 1
            if cur_iter > max_iter: break


        next_cmd = np.zeros((8,))
        next_cmd[inds[1]] = gripper2
        obs, _, _, _ = self.step(next_cmd, mode='end_effector_pos', view=view)
        # observations.append((next_cmd, obs))
        observations.append(obs)

        cur_iter = 0
        max_iter = 5
        ee_pos = self.get_left_ee_pos() if left else self.get_right_ee_pos()
        gripper_angle = self.get_gripper_joint_angles()[1] if left else self.get_gripper_joint_angles()[0]
        while np.any(np.abs(ee_pos - ee_above) > limit1):
            next_ee_cmd = np.minimum(np.maximum(ee_above - ee_pos, -0.1*np.ones((3,))), 0.1*np.ones((3,)))
            # next_ee_cmd = ee_above - ee_pos
            next_cmd = np.zeros((8,))
            next_cmd[inds[0]] = next_ee_cmd
            next_cmd[inds[1]] = gripper2

            cur_aux_ee_pos = self.get_right_ee_pos() if left else self.get_left_ee_pos()
            next_cmd[aux_inds[0]] = aux_ee_pos - cur_aux_ee_pos

            obs, _, _, _ = self.step(next_cmd, mode='end_effector_pos', view=view)
            # observations.append((next_cmd, obs))
            observations.append(obs)
            ee_pos = self.get_left_ee_pos() if left else self.get_right_ee_pos()
            gripper_angle = self.get_gripper_joint_angles()[1] if left else self.get_gripper_joint_angles()[0]
            cur_iter += 1
            if cur_iter > max_iter: break


        # cur_iter = 0
        # max_iter = 10
        # ee_pos = self.get_left_ee_pos() if left else self.get_right_ee_pos()
        # gripper_angle = self.get_gripper_joint_angles()[1] if left else self.get_gripper_joint_angles()[0]
        # while np.any(np.abs(ee_pos - end_ee_pos) > limit1):
        #     next_ee_cmd = np.minimum(np.maximum(end_ee_pos - ee_pos, -0.1*np.ones((3,))), 0.1*np.ones((3,)))
        #     # next_ee_cmd = ee_above - ee_pos
        #     next_cmd = np.zeros((8,))
        #     next_cmd[inds[0]] = next_ee_cmd
        #     next_cmd[inds[1]] = gripper2

        #     cur_aux_ee_pos = self.get_right_ee_pos() if left else self.get_left_ee_pos()
        #     next_cmd[aux_inds[0]] = aux_ee_pos - cur_aux_ee_pos

        #     obs, _, _, _ = self.step(next_cmd, mode='end_effector_pos', view=view)
        #     # observations.append((next_cmd, obs))
        #     observations.append(obs)
        #     ee_pos = self.get_left_ee_pos() if left else self.get_right_ee_pos()
        #     gripper_angle = self.get_gripper_joint_angles()[1] if left else self.get_gripper_joint_angles()[0]
        #     cur_iter += 1
        #     if cur_iter > max_iter: break

        return observations


    # def move_left_to_grasp(self, item_name, view=False):
    #     item_pos = self.get_item_pos(item_name)
    #     return self._move_to(item_pos, 1, 0, True, view)


    def move_left_to_grasp(self, pos, view=False):
        return self._move_to(pos, 1, 0, True, view)


    def move_left_to_place(self, target_pos, view=False):
        return self._move_to(target_pos, 0, 1, True, view)


    def move_left_to(self, pos1, pos2, reset_arm=True, view=True):
        if pos1[1] < -0.2 or pos2[1] < -0.2 or pos1[1] > 0.8 or pos2[1] > 0.8:
            return [self.get_obs(view=False)]

        if not (self._check_ik(pos1, quat=DOWN_QUAT, use_right=False) and \
                self._check_ik(pos2, quat=DOWN_QUAT, use_right=False)):
            return [self.get_obs(view=False)]

        self.physics.data.qpos[8] = 0.03
        self.physics.data.qpos[9] = -0.03
        self.physics.data.qpos[17] = 0.03
        self.physics.data.qpos[18] = -0.03
        self.physics.forward()
        start_jnts = self.get_arm_joint_angles()

        obs1 = self.move_left_to_grasp(pos1, view)
        obs2 = self.move_left_to_place(pos2, view)

        if reset_arm:
            self.set_arm_joint_angles(start_jnts)
            self.physics.forward()

        return np.r_[obs1, obs2]


    # def move_right_to_grasp(self, item_name, view=False):
    #     item_pos = self.get_item_pos(item_name)
    #     return self._move_to(item_pos, 1, 0, False, view)


    def move_right_to_grasp(self, pos, view=False):
        return self._move_to(pos, 1, 0, False, view)


    def move_right_to_place(self, target_pos, view=False):
        return self._move_to(target_pos, 0, 1, False, view)


    def move_right_to(self, pos1, pos2, reset_arm=True, view=True):
        if pos1[1] > 0.2 or pos2[1] > 0.2 or pos1[1] < -0.8 or pos2[1] < -0.8:
            return [self.get_obs(view=False)]
        if not (self._check_ik(pos1, quat=DOWN_QUAT, use_right=True) and \
                self._check_ik(pos2, quat=DOWN_QUAT, use_right=True)):
            return [self.get_obs(view=False)]

        self.physics.data.qpos[8] = 0.03
        self.physics.data.qpos[9] = -0.03
        self.physics.data.qpos[17] = 0.03
        self.physics.data.qpos[18] = -0.03
        self.physics.forward()
        start_jnts = self.get_arm_joint_angles()

        obs1 = self.move_right_to_grasp(pos1, view)
        obs2 = self.move_right_to_place(pos2, view)
        if reset_arm:
            self.set_arm_joint_angles(start_jnts)
            self.physics.forward()

        return np.r_[obs1, obs2]


    # def close(self):
    #     self.active = False
    #     if self._viewer is not None and self.use_glew:
    #         self._viewer.close()
    #         self._viewer = None
    #     self.physics.free()


    # def seed(self, seed=None):
    #     pass


    # def list_joint_info(self):
    #     for i in range(self.physics.model.njnt):
    #         print('\n')
    #         print('Jnt ', i, ':', self.physics.model.id2name(i, 'joint'))
    #         print('Axis :', self.physics.model.jnt_axis[i])
    #         print('Dof adr :', self.physics.model.jnt_dofadr[i])
    #         body_id = self.physics.model.jnt_bodyid[i]
    #         print('Body :', self.physics.model.id2name(body_id, 'body'))
    #         print('Parent body :', self.physics.model.id2name(self.physics.model.body_parentid[body_id], 'body'))
