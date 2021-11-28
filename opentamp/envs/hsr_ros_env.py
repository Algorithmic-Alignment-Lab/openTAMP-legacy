import main
import numpy as np
import rospy
import sys
import time
import traceback


from std_msgs.msg import Float32MultiArray

from pma import hl_solver, hsr_solver
from core.parsing import parse_domain_config, parse_problem_config
from core.util_classes.hsr_prob_gen import save_prob
import core.util_classes.hsr_constants as const

import opentamp
from opentamp.envs import HSRMJCEnv
from opentamp.util_classes.mjc_xml_utils import *
from opentamp.util_classes import transform_utils as T


_MAX_FRONTBUFFER_SIZE = 2048
_CAM_WIDTH = 200
_CAM_HEIGHT = 150

class HSRRosEnv(HSRMJCEnv):
    metadata = {'render.modes': ['human', 'rgb_array', 'depth'], 'video.frames_per_second': 67}

    def __init__(self, path_to_tampy=None, mode='end_effector', obs_include=[], items=[], include_files=[], include_items=[], im_dims=(_CAM_WIDTH, _CAM_HEIGHT), sim_freq=25, timestep=0.002, max_iter=250, view=False):
        super(HSRRosEnv, self).__init__(mode, obs_include, items, include_files, include_items, im_dims, sim_freq, timestep, max_iter, view)

        if path_to_tampy is not None:
            self._init_solver(path_to_tampy)

        self.mp_problems = []
        self.state_updates = []
        self._end = False

        rospy.init_node('HSR_Mujoco')
        self._init_publishers()
        self._init_subscribers()
        self.run()


    @classmethod
    def load_config(cls, config):
        path_to_tampy = config.get("tampy_path", None)
        mode = config.get("mode", "joint_angle")
        obs_include = config.get("obs_include", [])
        items = config.get("items", [])
        include_files = config.get("include_files", [])
        include_items = config.get("include_items", [])
        im_dims = config.get("image_dimensions", (_CAM_WIDTH, _CAM_HEIGHT))
        sim_freq = config.get("sim_freq", 25)
        ts = config.get("mjc_timestep", 0.002)
        view = config.get("view", False)
        max_iter = config.get("max_iterations", 250)
        return cls(path_to_tampy, mode, obs_include, items, include_files, include_items, im_dims, sim_freq, ts, max_iter, view)


    def run(self):
        while not self._end:
            limit = 100
            i = 0

            while len(self.state_updates) and i < limit:
                item, field, msg = self.state_updates.pop()
                if field == 'pos':
                    self.set_item_pos(item, msg.data)
                elif field == 'rot':
                    self.set_item_rot(item, msg.data)

            while len(self.mp_problems):
                try:
                    msg = self.mp_problems.pop()
                    self.plan_to_move(msg.data[:3], msg.data[3], msg.data[4:7], msg.data[7])
                except Exception as e:
                    traceback.print_exception(*sys.exc_info())

            time.sleep(0.01)

        self.close()


    def _init_publishers(self):
        self.obs_publishers = {}
        self.cam_publishers = {}
        self.item_publishers = {}

        for obs_type in self.obs_include:
            self.obs_publishers[obs_type] = rospy.Publisher(obs_type, Float32MultiArray, queue_size=1)
            if 'camera' in obs_type:
                self.cam_publishers[obs_type] = rospy.Publisher(obs_type, Float32MultiArray, queue_size=1)

        for item in self.item_names:
            self.item_publishers[item+'_pos'] = rospy.Publisher(item+'_pos', Float32MultiArray, queue_size=1)
            self.item_publishers[item+'_rot'] = rospy.Publisher(item+'_rot', Float32MultiArray, queue_size=1)


    def _init_subscribers(self):
        self.item_subscribers = {}
        for item in self.item_names:
            self.item_subscribers[item+'_pos_cmd'] = rospy.Subscriber(item+'_pos_cmd', Float32MultiArray, lambda m: self.state_updates.append((item, 'pos', m)))
            self.item_subscribers[item+'_rot_cmd'] = rospy.Subscriber(item+'_rot_cmd', Float32MultiArray, lambda m: self.state_updates.append((item, 'rot', m)))
        self.move_item_subscriber = rospy.Subscriber('move_item', Float32MultiArray, self.store_prob)


    def store_prob(self, msg):
        self.mp_problems.append(msg)


    def _init_solver(self, path_to_tampy):
        self.solver = hsr_solver.HSRSolver()
        domain_fname = path_to_tampy+'/domains/hsr_domain/hsr.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        self.domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        self.hls = hl_solver.FFSolver(d_c)
        obs_sizes = self.get_geom_dimensions()
        obs_pos = self.get_geom_positions()
        # obs_rot = self.get_geom_rotations(use_euler=True)
        obstacles = [('obs{0}'.format(i), obs_sizes[i].tolist(), obs_pos[i].tolist(), [0, 0, 0]) for i in range(len(obs_sizes))]
        save_prob(path_to_tampy+'/domains/hsr_domain/hsr_probs/hsr_sim_prob.prob', obstacles)
        p_c = main.parse_file_to_dict(path_to_tampy+'/domains/hsr_domain/hsr_probs/hsr_sim_prob.prob')
        self.problem = parse_problem_config.ParseProblemConfig.parse(p_c, self.domain)


    def sync_plan_init_state(self, plan):
        plan.params['hsr'].arm[:, 0] = self.get_arm_joint_angles()
        plan.params['hsr'].pose[:2,0] = self.get_base_pos()[:2]
        plan.params['hsr'].pose[2,0] - self.get_base_dir()
        plan.params['robot_init_pose'].arm[:,0] = plan.params['hsr'].arm[:, 0]
        plan.params['robot_init_pose'].value[:,0] = plan.params['hsr'].pose[:, 0]


    def plan_to_move(self, init_pos, init_theta, end_pos, end_theta):
        print('Solving motion plan')
        ll_plan_str = []
        act_num = 0
        ll_plan_str.append('{0}: MOVETO HSR ROBOT_INIT_POSE CAN_GRASP_BEGIN_0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: CAN_GRASP HSR CAN0 CAN0_INIT_TARGET CAN_GRASP_BEGIN_0 CG_EE_0 CAN_GRASP_END_0\n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: MOVEHOLDING_CAN HSR CAN_GRASP_END_0 CAN_PUTDOWN_BEGIN_0 CAN0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: CAN_PUTDOWN HSR CAN0 CAN0_END_TARGET CAN_PUTDOWN_BEGIN_0 CP_EE_0 CAN_PUTDOWN_END_0 \n'.format(act_num))
        act_num += 1
        plan = self.hls.get_plan(ll_plan_str, self.domain, self.problem)
        self.sync_plan_init_state(plan)
        plan.params['can0'].pose[:,0] = init_pos
        plan.params['can0'].rotation[2,0] = init_theta
        plan.params['can0_init_target'].value[:,0] = plan.params['can0'].pose[:,0]
        plan.params['can0_init_target'].rotation[:,0] = plan.params['can0'].rotation[:,0]
        plan.params['can0_end_target'].value[:,0] = end_pos
        plan.params['can0_end_target'].rotation[0,0] = end_theta

        def callback(a):
            return None
        result = self.solver.backtrack_solve(plan, callback = callback, verbose=False, n_resamples=2)
        self.run_plan(plan)


    def run_plan(self, plan):
        for t in range(plan.horizon):
            hsr = plan.params['hsr']
            grip = -50 if hsr.gripper[:, t] < 0.6 else 50
            action = np.r_[hsr.pose[:,t], hsr.arm[:,t], grip]
            self.step(action, mode="joint_angle")
        print(self.physics.data.qpos)
        print(self.physics.data.ctrl)
        import ipdb; ipdb.set_trace()


    def step(self, action, mode=None):
        obs, _, _, _ = super(HSRRosEnv, self).step(action, mode)
        for obs_type in self.obs_include:
            obs_data = self.get_obs_data(obs, obs_type)
            obs_msg = Float32MultiArray(data=obs_data.flatten().tolist())
            self.obs_publishers[obs_type].publish(obs_msg)
            if 'camera' in obs_type:
                fovy, pos, mat = self.get_camera_info(obs_type)
                obs_msg = Float32MultiArray(data=np.r_[fovy, pos, mat].tolist())
                self.cam_publishers[obs_type].publish(obs_msg)

        for item in self.item_names:
            pos = self.get_item_pos(item)
            quat = self.get_item_rot(item)
            pos_msg = Float32MultiArray(data=pos.flatten().tolist())
            quat_msg = Float32MultiArray(data=quat.flatten().tolist())
            self.item_publishers[item+'_pos'].publish(pos_msg)
            self.item_publishers[item+'_rot'].publish(quat_msg)

        if self.use_viewer:
            self.render(camera_id=1, view=True)

        return obs, _, _, _
