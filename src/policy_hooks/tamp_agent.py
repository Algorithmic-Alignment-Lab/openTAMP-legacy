""" This file defines an agent for the MuJoCo simulator environment. """
import copy

import numpy as np

import xml.etree.ElementTree as xml

import openravepy
from openravepy import RaveCreatePhysicsEngine

from mujoco_py import mjcore, mjconstants
from mujoco_py.mjlib import mjlib

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise
from gps.agent.config import AGENT
from gps.sample.sample import Sample

import core.util_classes.baxter_constants as const
import core.util_classes.items as items
from core.util_classes.openrave_body import OpenRAVEBody
from policy_hooks.action_utils import action_durations
from policy_hooks.baxter_controller import BaxterMujocoController
from policy_hooks.policy_solver_utils import STATE_ENUM, OBS_ENUM, ACTION_ENUM, NOISE_ENUM
import policy_hooks.policy_solver_utils as utils

'''
Mujoco specific
'''
BASE_POS_XML = '../models/baxter/mujoco/baxter_mujoco_pos.xml'
BASE_MOTOR_XML = '../models/baxter/mujoco/baxter_mujoco.xml'
ENV_XML = 'policy_hooks/current_env.xml'

MUJOCO_TIME_DELTA = 0.002

MUJOCO_JOINT_ORDER = ['left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_e2', 'left_w0', 'left_w1', 'left_w2', 'left_gripper_l_finger_joint', 'left_gripper_r_finger_joint'\
                      'right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2', 'right_gripper_l_finger_joint', 'right_gripper_r_finger_joint']

MUJOCO_MODEL_Z_OFFSET = -0.686

# TODO: Handle mapping Mujoco state to TAMPy state
# TODO: Split this into two agents for the different simulators (Mujooc & Bullet).
class LaundryWorldMujocoAgent(Agent):
    def __init__(self, hyperparams):
        config = copy.deepcopy(AGENT)
        config.update(hyperparams)
        self._hyperparams = config

        self.plan = self._hyperparams['plan']
        self.active_ts, self.params = utils.get_plan_traj_info(plan)
        self.solver = self._hyperparams['solver']
        self.x0s = self._hyperparams['x0s']
        self.sim = 'mujoco'
        self.viewer = None
        self.pos_model = self.setup_mujoco_model(self.plan, motor=False, view=True)
        self.motor_model = self.setup_mujoco_model(self.plan)
        self.controller = BaxterMujocoController(self.model, pos_gains=2.5e2, vel_gains=1e1)
        self.demonstrations = np.ones((len(self.plans))) * self._hyperparams['demonstrations']

        Agent.__init__(self, config)

    def _generate_xml(self, plan, motor=True):
        '''
            Search a plan for cloths, tables, and baskets to create an XML in MJCF format
        '''
        base_xml = xml.parse(BASE_MOTOR_XML) if motor else xml.parse(BASE_POS_XML)
        root = base_xml.getroot()
        worldbody = root.find('worldbody')
        active_ts = (0, plan.horizon+1)
        params = plan.params.values()

        # real_t = np.sum([t for t in plan.time[:, active_ts[0]:active_ts[1]+1]])
        # root.append(xml.fromstring(options_xml.format(MUJOCO_TIME_DELTA, real_t/MUJOCO_TIME_DELTA)))

        for param in params:
            if param.is_symbol(): continue
            if param._type == 'Cloth':
                height = param.geom.height
                radius = param.geom.radius
                x, y, z = param.pose[:, active_ts[0]]
                cloth_body = xml.SubElement(worldbody, 'body', {'name': param.name, 'pos': "{} {} {}".format(x,y,z+MUJOCO_MODEL_Z_OFFSET), 'euler': "0 0 0"})
                cloth_geom = xml.SubElement(cloth_body, 'geom', {'name':param.name, 'type':'cylinder', 'size':"{} {}".format(radius, height), 'rgba':"0 0 1 1", 'friction':'1 1 1'})
            # We might want to change this; in Openrave we model tables as hovering box so there's no easy translation to Mujoco
            elif param._type == 'Obstacle': 
                length = param.geom.dim[0]
                width = param.geom.dim[1]
                thickness = param.geom.dim[2]
                x, y, z = param.pose[:, active_ts[0]]
                table_body = xml.SubElement(worldbody, 'body', {'name': param.name, 'pos': "{} {} {}".format(x, y, z+MUJOCO_MODEL_Z_OFFSET), 'euler': "0 0 0"})
                table_geom = xml.SubElement(table_body, 'geom', {'name':param.name, 'type':'box', 'size':"{} {} {}".format(length, width, thickness)})
            elif param._type == 'Basket':
                x, y, z = param.pose[:, active_ts[0]]
                yaw, pitch, roll = param.rotation[:, active_ts[0]]
                basket_body = xml.SubElement(worldbody, 'body', {'name':param.name, 'pos':"{} {} {}".format(x, y, z+MUJOCO_MODEL_Z_OFFSET), 'euler':'{} {} {}'.format(roll, pitch, yaw)})
                basket_intertial = xml.SubElement(basket_body, 'inertial', {'pos':"0 0 0", 'mass':"1", 'diaginertia':"2 1 1"})
                basket_geom = xml.SubElement(basket_body, 'geom', {'name':param.name, 'type':'mesh', 'mesh': "laundry_basket"})
        base_xml.write(ENV_XML)

    def setup_mujoco_model(self, plan, motor=True, view=False):
        '''
            Create the Mujoco model and intiialize the viewer if desired
        '''
        self._generate_xml(plan, motor)
        model = mjcore.MjModel(ENV_XML)
        if view:
            self.viewer = mjviewer.MjViewer()
            self.viewer.start()
            self.viewer.set_model(model)
            self.viewer.loop_once()
        return model

    # def run_ee_policy(self, cond, policy, noise):
    #     '''
    #         Run the policy (for end effector control) in simulation
    #     '''
    #     plan = self.plans[cond]
    #     baxter = plan.params[p'baxter']
    #     x0 = self.x0[conds]
    #     if plan in self._traj_info_cache:
    #         active_ts, params = self._traj_info_cache[plan]
    #     else:
    #         active_ts, params = utils.get_plan_traj_info(plan)
    #         self._traj_info_cache[plan] = (active_ts, params)
    #     self._set_simulator_state(x0, plan)
    #     trajectory_state = np.zeros((plan.dX, plan.T))
        
    #     x = x0.copy()
    #     for ts in range(active_ts[0], active_ts[1]):
    #         obs = self._get_obs(cond, t)
    #         u = policy.act(x, obs, noise[:, ts - active_ts[0]], ts - active_ts[0])

    #         right_ee_vec = u[:7]
    #         left_ee_vec = u[7:]

    #         right_grip = const.GRIPPER_OPEN_VALUE if right_ee_vec[6] > 0.5 else const.GRIPPER_CLOSE_VALUE
    #         left_grip = const.GRIPPER_OPEN_VALUE if left_ee_vec[6] > 0.5 else const.GRIPPER_CLOSE_VALUE

    #         cur_right_joints = x[plan.state_inds[(param.name, 'rArmPose')]]
    #         cur_left_joints = x[plan.state_inds[(param.name, 'lArmPose')]]

    #         baxter.set_dof('lArmPose':cur_left_joints, 'rArmPose':cur_right_joints, 'lGripper':left_grip, 'rGripper':right_grip)
    #         ee_right = robot.openrave_body.env_body.GetLink('right_gripper').GetTransform()
    #         ee_left = robot.openrave_body.env_body.GetLink('left_gripper').GetTransform()

    #         next_right_poses = baxter.openrave_body.get_ik_from_pose(right_ee_vec[:3], right_ee_vec[3:6], 'right_arm')
    #         next_left_poses = baxter.openrave_body.get_ik_from_pose(left_ee_vec[:3], left_ee_vec[3:6], 'left_arm')

    #         next_right_pose = utils.closest_arm_pose(next_right_poses, x[plan.state_inds[('baxter', 'rArmPose')]])
    #         next_left_pose = utils.closest_arm_pose(next_left_poses, x[plan.state_inds[('baxter', 'lArmPose')]])

    #         while(np.any(np.abs(cur_right_joints - next_right_pose) > 0.01) or np.any(np.abs(cur_left_joints - next_left_pose) > 0.01)):
    #             self.pos_model.data.ctrl = np.r_[next_right_pose, right_grip, -right_grip, next_left_pose, left_grip, -left_grip]
    #             self.pos_model.step()
    #             cur_right_joints = self.pos_model.data.qpos[1:8]
    #             cur_left_joints = self.pos_model.data.qpos[10:17]

    #         x = self._get_simulator_state()
    #         trajectory_state[:, ts] = x
    #     return trajectory_state

    def _set_simulator_state(self, x, plan):
        '''
            Set the simulator to the state of the specified condition, except for the robot
        '''
        model  = self.motor_model
        xpos = model.data.xpos.copy()
        xquat = model.data.xquat.copy()
        param = plan.params.values()

        for param in params:
            if param._type != 'Robot' and (param.name, 'pose') in plan.state_inds and (param.name, 'rotation') in plan.state_inds:
                param_ind = mjlib.mj_name2id(model.ptr, mjconstants.mjOBJ_BODY, param.name)
                if param_ind == -1: continue
                pos = x[plan.state_inds[(param.name, 'pose')]]
                rot = x[plan.state_inds[(param.name, 'rotation')]]
                xpos[param_ind] = pos + np.array([0, 0, MUJOCO_MODEL_Z_OFFSET])
                xquat[param_ind] = openravepy.quatFromRotationMatrix(OpenRAVEBody.transform_from_obj_pose(pos, rot)[:3,:3])

        model.data.xpos = xpos
        model.data.xquat = xquat
        model.data.qpos = self._baxter_to_mujoco(x, plan.state_inds)

    def _baxter_to_mujoco(self, x, x_inds):
        return np.r_[0, x[x_inds['baxter', 'rArmPose']], x[x_inds['baxter', 'rGripper']], -x[x_inds['baxter', 'rGripper']], x[x_inds['baxter', 'lArmPose']], x[x_inds['baxter', 'lGripper']], -x[x_inds['baxter', 'lGripper']]]

    def _get_simulator_state(self, x_inds, dx):
        X = np.zeros((dX,))

        for param in x_inds.keys():
            if param._type != "Robot" and not param.is_symbol():
                param_ind = self.pos_model.body_names.index(param.name)
                X[x_inds[param.name, 'pose']] = self.pos_model.data.xpos[param_ind].flatten()
                quat = self.pos_model.data.xquat[param_ind].flatten()
                rotation = [np.atan2(2*(quat[0]*quat[1]+quat[2]*quat[3]), 1-2*(quat[1]**2+quat[2]**2)), np.arcsin(2*(quat[0]*quat[2] - q[3]*quat[1])), \
                            np.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 1-2*(quat[2]**2+quat[3]**2))]

                X[x_inds[param.name, 'rotation']] = rotation

            elif param._type == "Robot":
                robot_name = param.name
                robot_pose = self.pos_model.data.xpos[self.pos_model.body_names.index('baxter')]
                right_arm = self.pos_model.data.qpos[1:8]
                x[('baxter', 'rArmPose')] = right_arm
                right_grip = const.GRIPPER_OPEN_VALUE if self.pos_model.data.qpos[8] > 0 else const.GRIPPER_CLOSE_VALUE
                x[('baxter', 'rGripper')] = right_grip

                left_arm = self.pos_model.data.qpos[10:17]
                x[('baxter', 'lArmPose')] = left_arm
                left_grip = const.GRIPPER_OPEN_VALUE if self.pos_model.data.qpos[17] > 0 else const.GRIPPER_CLOSE_VALUE
                x[('baxter', 'lGripper')] - left_grip

        return X
    
    def _get_obs(self, cond, t):
        o_t = np.zeros((self.plan.symbolic_bound))
        return o_t

    def sample(self, policy, condition, verbose=False, save=True, noisy=False):
        '''
            Take a sample for an entire trajectory for the given condition
        '''
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))

        if self.demonstrations[condition]:
            sample = self._sample_joint_angle_trajectory(condition, noise)
            save = True
            self.demonstrations[condition] -= 1
        else:
            self.sample_joint_trajectory_policy(self, policy, condition, noise)

        if save:
            self._samples[condition].append(sample)
        return sample

    def _sample_joint_angle_trajectory(self, condition):
        sample = Sample(self)

        # Initialize the plan for the given condition
        x0 = self.x0s[condition]
        utils.set_params_attrs(self.params, self.plan.state_inds, x0[0], 0)

        first_act = self.plan.actions[x0[1][0]]
        last_act = self.plan.actions[x0[1][1]]

        init_t = first_act.active_timesteps[0]
        final_t = last_act.active_timesteps[1]

        first_act.params[1] = plan.params['robot_init_pose'] # Cloth world specific

        self.solver._backtrack_solve(self.plans[condition], anum=x0[1][0], amax=x0[1][1])

        for ts in range(first_act.active_timesteps[0], last_act.active_timesteps[1]):
            U = np.zeros(self.plan.dU)
            utils.fill_vector(params, self.plan.action_inds, U, ts+1)
            sample.set(ACTION_ENUM, np.r_[left_vec, right_vec], ts - first_act.active_timesteps[0])
            X = np.zeros((self.plan.dX,))
            utils.fill_vector(params, self.plan.state_inds, X, ts)
            sample.set(STATE_ENUM, X, ts-init_t)
            sample.set(OBS_ENUM, X, ts-init_t)
            sample.set(NOISE_ENUM, np.zeros((self.dU)), ts-init_t)

        return sample

    def sample_joint_trajectory_policy(self, pol, cond, noise):
        sample = Sample(self)
        x0 = self.x0s[cond]
        utils.set_params_attrs(self.params, self.plan.state_inds, x0[0], 0)
        self._set_simulator_state(x0, self.plan)
        obs = x0

        first_act = self.plan.actions[x0[1][0]]
        last_act = self.plan.actions[x0[1][1]]

        init_t = first_act.active_timesteps[0]
        final_t = last_act.active_timesteps[1]

        X = x0[self.plan.symbolic_bound]
        for ts in range(0, final_t-init_t):
            U = policy.act(X, obs, ts, noise[ts, :])
            sample.set(STATE_ENUM, X, ts)
            sample.set(OBS_ENUM, X, ts)
            sample.set(ACTION_ENUM, U, ts)
            sample.set(NOISE_ENUM, noise[ts, :], ts)
            self.run_traj_policy_step(U)
            X = self._get_simulator_state(plan.state_inds, plan.dX)[self.plan.symbolic_bound]
            obs = X

        return sample

    def run_traj_policy_step(self, u, u_inds):
        next_right_pose = u[u_inds['baxter', 'rArmPose']]
        next_left_pose = u[u_inds['baxter', 'lArmPose']]
        cur_right_joints = self.pos_model.data.qpos[1:8]
        cur_left_joints = self.pos_model.data.qpos[10:17]
        self.pos_model.data.ctrl = np.r_[next_right_pose, u[u_inds['baxter', 'rGripper']], -u[u_inds['baxter', 'rGripper']], next_left_pose, u[u_inds['baxter', 'lGripper']], -u[u_inds['baxter', 'lGripper']]]
        while(np.any(np.abs(cur_right_joints - next_right_pose) > 0.05) or np.any(np.abs(cur_left_joints - next_left_pose) > 0.05)):
            self.pos_model.step()
            cur_right_joints = self.pos_model.data.qpos[1:8]
            cur_left_joints = self.pos_model.data.qpos[10:17]

    # def run_policy_traj_step(self, U):
    #     pass

    # def _sample_ee_trajectory(self, condition, noise):
    #     sample = Sample(self)
    #     plan = self.plans[condition]
    #     if plan in self._traj_info_cache:
    #         active_ts, params = self._traj_info_cache[plan]
    #     else:
    #         active_ts, params = utils.get_plan_traj_info(plan)
    #         self._traj_info_cache[plan] = (active_ts, params)

    #     t0 = active_ts[0]
    #     baxter = plan.params['baxter']
    #     actions = filter(plan.actions, lambda a: a.active_timesteps[0] >= t0)

    #     for ts in range(active_ts[0], active_ts[1]):
    #         baxter.set_dof('lArmPose': baxter.lArmPose[:, ts], 'rArmPose': baxter.rArmPose[:, ts])
    #         ee_right = robot.openrave_body.env_body.GetLink('right_gripper').GetTransform()
    #         ee_left = robot.openrave_body.env_body.GetLink('left_gripper').GetTransform()

    #         baxter.set_dof('lArmPose': baxter.lArmPose[:, ts+1], 'rArmPose': baxter.rArmPose[:, ts+1])
    #         ee_right_next = robot.openrave_body.env_body.GetLink('right_gripper').GetTransform()
    #         ee_left_next = robot.openrave_body.env_body.GetLink('left_gripper').GetTransform()

    #         right_rot = OpenRAVEBody._ypr_from_rot_matrix(ee_right[:3.:3])
    #         left_rot = OpenRAVEBody._ypr_from_rot_matrix(ee_left[:3,:3])
    #         next_right_rot = OpenRAVEBody._ypr_from_rot_matrix(ee_right_next[:3.:3])
    #         next_left_rot = OpenRAVEBody._ypr_from_rot_matrix(ee_left_next[:3,:3])

    #         right_open = 0 if plan.params['baxter'].lGripper[:, ts+1] < const.GRIPPER_OPEN_VALUE else 1
    #         left_open = 0 if plan.params['baxter'].rGripper[:, ts+1] < const.GRIPPER_OPEN_VALUE else 1

    #         # right_vec = np.r_[ee_right_next[:3,3] - ee_right[:3,3], next_right_rot - right_rot, right_open]
    #         # left_vec = np.r_[ee_left_next[:3,3] - ee_left[:3,3], next_left_rot - left_rot, left_open]
    #         right_vec = np.r_[ee_right_next[:3,3], next_right_rot, right_open]
    #         left_vec = np.r_[ee_left_next[:3,3], next_left_rot, left_open]


    #         sample.set(ACTION_ENUM, np.r_[left_vec, right_vec], ts - active_ts[0])
    #         X = np.zeros((plan.dX,))
    #         utils.fill_vector(params, plan.state_inds, X, ts)
    #         sample.set(STATE_ENUM, X, ts - active_ts[0])
    #         sample.set(NOISE_ENUM, noise, ts - active_ts[0])

    #     return sample


    # def _sample_joint_efforts(self, condition, noise):
    #     sample = Sample(self)
    #     plan = self.plans[condition]
    #     if plan in self._traj_info_cache:
    #         active_ts, params = self._traj_info_cache[plan]
    #     else:
    #         active_ts, params = utils.get_plan_traj_info(plan)
    #         self._traj_info_cache[plan] = (active_ts, params)

    #     t0 = active_ts[0]
    #     baxter = plan.params['baxter']
    #     actions = filter(plan.actions, lambda a: a.active_timesteps[0] >= t0)

    #     for ts in range(active_ts[0], active_ts[1]):
    #         U = np.zeros(plan.dU)
    #         utils.fill_vector(params, plan.action_inds, U, ts)
    #         sample.set(ACTION_ENUM, np.r_[left_vec, right_vec], ts - active_ts[0])
    #         X = np.zeros((plan.dX,))
    #         utils.fill_vector(params, plan.state_inds, X, ts)
    #         sample.set(STATE_ENUM, X, ts - active_ts[0])
    #         sample.set(NOISE_ENUM, noise, ts - active_ts[0])

    #     return sample
    #             