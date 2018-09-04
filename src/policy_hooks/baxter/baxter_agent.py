import copy

import cPickle as pickle

import ctypes

import numpy as np

import xml.etree.ElementTree as xml

import openravepy
from openravepy import RaveCreatePhysicsEngine


# from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise
from gps.agent.config import AGENT
#from gps.sample.sample import Sample
from gps.sample.sample_list import SampleList

import core.util_classes.baxter_constants as const
import core.util_classes.items as items
from core.util_classes.namo_predicates import dsafe
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes.viewer import OpenRAVEViewer
from core.util_classes.plan_hdf5_serialization import PlanSerializer, PlanDeserializer

from policy_hooks.agent import Agent
from policy_hooks.sample import Sample
from policy_hooks.utils.policy_solver_utils import *
import policy_hooks.utils.policy_solver_utils as utils
from policy_hooks.utils.tamp_eval_funcs import *
from policy_hooks.baxter.sorting_prob import *


class BaxterSortingAgent(Agent):
    def __init__(self, hyperparams):
        Agent.__init__(self, hyperparams)
        # Note: All plans should contain identical sets of parameters
        self.plans = self._hyperparams['plans']
        self.task_list = self._hyperparams['task_list']
        self.task_durations = self._hyperparams['task_durations']
        self.task_encoding = self._hyperparams['task_encoding']
        # self._samples = [{task:[] for task in self.task_encoding.keys()} for _ in range(self._hyperparams['conditions'])]
        self._samples = {task: [] for task in self.task_list}
        self.state_inds = self._hyperparams['state_inds']
        self.action_inds = self._hyperparams['action_inds']
        self.dX = self._hyperparams['dX']
        self.dU = self._hyperparams['dU']
        self.symbolic_bound = self._hyperparams['symbolic_bound']
        self.solver = self._hyperparams['solver']
        self.num_cans = self._hyperparams['num_cans']
        self.init_vecs = self._hyperparams['x0']
        self.x0 = [x[:self.symbolic_bound] for x in self.init_vecs]
        self.targets = self._hyperparams['targets']
        self.target_dim = self._hyperparams['target_dim']
        self.target_inds = self._hyperparams['target_inds']
        self.target_vecs = []
        for condition in range(len(self.x0)):
            target_vec = np.zeros((self.target_dim,))
            for target_name in self.targets[condition]:
                if (target_name, 'value') in self.target_inds:
                    target_vec[self.target_inds[target_name, 'value']] = self.targets[condition][target_name]
            self.target_vecs.append(target_vec)
        self.targ_list = self.targets[0].keys()
        self.obj_list = self._hyperparams['obj_list']

        self._get_hl_plan = self._hyperparams['get_hl_plan']
        self.env = self._hyperparams['env']
        self.openrave_bodies = self._hyperparams['openrave_bodies']

        self.current_cond = 0
        self.cond_global_pol_sample = [None for _ in  range(len(self.x0))] # Samples from the current global policy for each condition
        self.initial_opt = True
        self.stochastic_conditions = self._hyperparams['stochastic_conditions']

        self.hist_len = self._hyperparams['hist_len']
        self.traj_hist = None
        self.reset_hist()

        self.optimal_samples = {task: [] for task in self.task_list}
        self.optimal_state_traj = [[] for _ in range(len(self.plans))]
        self.optimal_act_traj = [[] for _ in range(len(self.plans))]

        self.task_paths = []

        self.get_plan = self._hyperparams['get_plan']

        self.in_left_grip = -1
        self.in_right_grip = -1


    def get_samples(self, task):
        samples = []
        for batch in self._samples[task]:
            samples.append(SampleList(batch))

        return samples
        

    def add_sample_batch(self, samples, task):
        self._samples[task].append(samples)


    def clear_samples(self, keep_prob=0., keep_opt_prob=1.):
        for task in self.task_list:
            n_keep = int(keep_prob * len(self._samples[task]))
            self._samples[task] = random.sample(self._samples[task], n_keep)

            n_opt_keep = int(keep_opt_prob * len(self.optimal_samples[task]))
            self.optimal_samples[task] = random.sample(self.optimal_samples[task], n_opt_keep)


    def reset_sample_refs(self):
        for task in self.task_list:
            for batch in self._samples[task]:
                for sample in batch:
                    sample.set_ref_X(np.zeros((sample.T, self.symbolic_bound)))
                    sample.set_ref_U(np.zeros((sample.T, self.dU)))


    def add_task_paths(self, paths):
        self.task_paths.extend(paths)


    def get_task_paths(self):
        return copy.copy(self.task_paths)


    def clear_task_paths(self, keep_prob=0.):
        n_keep = int(keep_prob * len(self.task_paths))
        self.task_paths = random.sample(self.task_paths, n_keep)


    def reset_hist(self, hist=[]):
        if not len(hist):
            hist = np.zeros((self.hist_len, self.dU)).tolist()
        self.traj_hist = hist


    def get_hist(self):
        return copy.deepcopy(self.traj_hist)


    def sample(self, policy, condition, save_global=False, verbose=False, noisy=False):
        raise NotImplementedError


    def sample_task(self, policy, condition, x0, task, use_prim_obs=False, save_global=False, verbose=False, use_base_t=True, noisy=True):
        task = tuple(task)
        plan = self.plans[task[:2]]
        for (param, attr) in self.state_inds:
            if plan.params[param].is_symbol(): continue
            getattr(plan.params[param], attr)[:,0] = x0[self.state_inds[param, attr]]

        base_t = 0
        self.T = plan.horizon
        sample = Sample(self)
        sample.init_t = 0

        target_vec = np.zeros((self.target_dim,))

        set_params_attrs(plan.params, plan.state_inds, x0, 0)
        for target_name in self.targets[condition]:
            target = plan.params[target_name]
            target.value[:,0] = self.targets[condition][target.name]
            if (target.name, 'value') in self.target_inds:
                target_vec[self.target_inds[target.name, 'value']] = target.value[:,0]

        # self.traj_hist = np.zeros((self.hist_len, self.dU)).tolist()

        if noisy:
            noise = 1e1 * generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))

        for t in range(0, self.T):
            X = np.zeros((plan.symbolic_bound))
            fill_vector(plan.params, plan.state_inds, X, t)

            sample.set(STATE_ENUM, X.copy(), t)
            if OBS_ENUM in self._hyperparams['obs_include']:
                sample.set(OBS_ENUM, im.copy(), t)
            sample.set(NOISE_ENUM, noise[t], t)
            sample.set(TRAJ_HIST_ENUM, np.array(self.traj_hist).flatten(), t)
            task_vec = np.zeros((len(self.task_list)), dtype=np.float32)
            task_vec[self.task_list.index(task[0])] = 1.
            sample.set(TASK_ENUM, task_vec, t)
            sample.set(TARGETS_ENUM, target_vec.copy(), t)

            obj_vec = np.zeros((len(self.obj_list)), dtype='float32')
            targ_vec = np.zeros((len(self.targ_list)), dtype='float32')
            obj_vec[self.obj_list.index(task[1])] = 1.
            targ_vec[self.targ_list.index(task[2])] = 1.
            sample.set(OBJ_ENUM, obj_vec, t)
            sample.set(TARG_ENUM, targ_vec, t)
            sample.task = task[0]
            sample.condition = condition

            if use_prim_obs:
                obs = sample.get_prim_obs(t=t)
            else:
                obs = sample.get_obs(t=t)

            U = policy.act(sample.get_X(t=t), obs, t, noise[t])
            sample.set(ACTION_ENUM, U.copy(), t)
            if np.any(np.isnan(U)):
                U[np.isnan(U)] = 0
                # import ipdb; ipdb.set_trace()
            
            self.traj_hist.append(U)
            while len(self.traj_hist) > self.hist_len:
                self.traj_hist.pop(0)

            self.run_policy_step(U, X, self.plans[task[:2]], t)

        return sample


    def run_policy_step(self, u, x, plan, t):
        u_inds = self.action_inds
        x_inds = self.state_inds
        in_gripper = False

        
        if t < plan.horizon - 1:
            self._clip_joint_angles(u, plan)
            for param, attr in u_inds:
                getattr(plan.params[param], attr)[:, t+1] = u[u_inds[param, attr]]

            plan.params['baxter'].openrave_body.set_dof({
                'lArmPose': plan.params['baxter'].lArmPose[:, t],
                'lGripper': plan.params['baxter'].lGripper[:, t],
                'rArmPose': plan.params['baxter'].rArmPose[:, t],
                'rGripper': plan.params['baxter'].rGripper[:, t],
            })
            l_pose = plan.params['baxter'].openrave_body.env_body.GetLink('left_gripper').GetTransformPose()[4:]
            r_pose = plan.params['baxter'].openrave_body.env_body.GetLink('right_gripper').GetTransformPose()[4:]

            for param in plan.params.values():
                if param._type == 'Can':
                    l_pose = plan.params['baxter'].openrave_body.env_body.Get
                    l_dist = l_pose - plan.params[param.name].pose[:, t]
                    r_dist = r_pose - plan.params[param.name].pose[:, t]

                    if plan.params['baxter'].lGripper[0, t] < 0.01 and np.all(np.abs(l_dist) < 0.02):
                        param.pose[:, t+1] = l_pose
                    elif plan.params['baxter'].rGripper[0, t] < 0.01 and np.all(np.abs(r_dist) < 0.02):
                        param.pose[:, t+1] = r_pose
                    elif param._type == 'Can':
                        param.pose[:, t+1] = param.pose[:, t]

        return True


    def _clip_joint_angles(self, u, plan):
        DOF_limits = plan.params['baxter'].openrave_body.env_body.GetDOFLimits()
        left_DOF_limits = (DOF_limits[0][2:9]+0.000001, DOF_limits[1][2:9]-0.000001)
        right_DOF_limits = (DOF_limits[0][10:17]+0.000001, DOF_limits[1][10:17]-0.00001)
        left_joints = u[plan.action_inds['baxter', 'lArmPose']]
        left_grip = u[plan.action_inds['baxter', 'lGripper']]
        right_joints = u[plan.action_inds['baxter', 'rArmPose']]
        right_grip = u[plan.action_inds['baxter', 'rGripper']]

        if left_grip[0] < 0:
            left_grip[0] = 0.015
        elif left_grip[0] > 0.02:
            left_grip[0] = 0.02

        if right_grip[0] < 0:
            right_grip[0] = 0.015
        elif right_grip[0] > 0.02:
            right_grip[0] = 0.02

        for i in range(7):
            if left_joints[i] < left_DOF_limits[0][i]:
                left_joints[i] = left_DOF_limits[0][i]
            if left_joints[i] > left_DOF_limits[1][i]:
                left_joints[i] = left_DOF_limits[1][i]
            if right_joints[i] < right_DOF_limits[0][i]:
                right_joints[i] = right_DOF_limits[0][i]
            if right_joints[i] > right_DOF_limits[1][i]:
                right_joints[i] = right_DOF_limits[1][i]

        u[plan.action_inds['baxter', 'lArmPose']] = left_joints
        u[plan.action_inds['baxter', 'lGripper']] = left_grip
        u[plan.action_inds['baxter', 'rArmPose']] = right_joints
        u[plan.action_inds['baxter', 'rGripper']] = right_grip


    def set_nonopt_attrs(self, plan, task):
        plan.dX, plan.dU, plan.symbolic_bound = self.dX, self.dU, self.symbolic_bound
        plan.state_inds, plan.action_inds = self.state_inds, self.action_inds


    def sample_optimal_trajectory(self, state, task, condition, traj_mean=[], fixed_targets=[]):
        exclude_targets = []
        success = False

        if len(fixed_targets):
            targets = fixed_targets
            obj = fixed_targets[0]
            targ = fixed_targets[1]
        else:
            task_distr, obj_distr, targ_distr = self.prob_func(sample.get_prim_obs(t=0))
            obj = self.plans.values()[0].params[self.obj_list[np.argmax(obj_distr)]]
            targ = self.plans.values()[0].params[self.targ_list[np.argmax(targ_distr)]]
            targets = [obj, targ]
            # targets = get_next_target(self.plans.values()[0], state, task, self.targets[condition], sample_traj=traj_mean)

        failed_preds = []
        iteration = 0
        while not success and targets[0] != None:
            if iteration > 0 and not len(fixed_targets):
                 targets = get_next_target(self.plans.values()[0], state, task, self.targets[condition], sample_traj=traj_mean, exclude=exclude_targets)

            iteration += 1
            if targets[0] is None:
                break

            plan = self.plans[task, targets[0].name] 
            set_params_attrs(plan.params, plan.state_inds, state, 0)
            for param_name in plan.params:
                param = plan.params[param_name]
                if param._type == 'Can' and '{0}_init_target'.format(param_name) in plan.params:
                    plan.params['{0}_init_target'.format(param_name)].value[:,0] = plan.params[param_name].pose[:,0]

            for target in self.targets[condition]:
                plan.params[target].value[:,0] = self.targets[condition][target]

            if targ.name in self.targets[condition]:
                plan.params['{0}_end_target'.format(obj.name)].value[:,0] = self.targets[condition][targ.name]

            if task == 'grasp':
                plan.params[targ.name].value[:,0] = plan.params[obj.name].pose[:,0]
            
            plan.params['robot_init_pose'].lArmPose[:,0] = plan.params['baxter'].lArmPose[:,0]
            plan.params['robot_init_pose'].lGripper[:,0] = plan.params['baxter'].lGripper[:,0]
            plan.params['robot_init_pose'].rArmPose[:,0] = plan.params['baxter'].rArmPose[:,0]
            plan.params['robot_init_pose'].rGripper[:,0] = plan.params['baxter'].rGripper[:,0]
            # plan.params['robot_end_pose'].lArmPose[:,0] = plan.params['baxter'].lArmPose[:,-1]
            # plan.params['robot_end_pose'].lGripper[:,0] = plan.params['baxter'].lGripper[:,-1]
            # plan.params['robot_end_pose'].rArmPose[:,0] = plan.params['baxter'].rArmPose[:,-1]
            # plan.params['robot_end_pose'].rGripper[:,0] = plan.params['baxter'].rGripper[:,-1]
            # if task == 'grasp':
            #     plan.params['robot_end_pose'].value[:,0] = plan.params[targets[1].name].value[:,0] - [0, dist+0.2]
            # self.env.SetViewer('qtcoin')
            success = self.solver._backtrack_solve(plan, n_resamples=5, traj_mean=traj_mean)
            # try:
            #     success = self.solver._backtrack_solve(plan, n_resamples=5, traj_mean=traj_mean)
            #     # viewer = OpenRAVEViewer._viewer if OpenRAVEViewer._viewer is not None else OpenRAVEViewer(plan.env)
            #     # import ipdb; ipdb.set_trace()
            #     # if task == 'putdown':
            #     #     import ipdb; ipdb.set_trace()
            #     # self.env.SetViewer('qtcoin')
            #     # import ipdb; ipdb.set_trace()
            # except Exception as e:
            #     print e
            #     # self.env.SetViewer('qtcoin')
            #     # import ipdb; ipdb.set_trace()
            #     success = False

            if not len(failed_preds):
                failed_preds = [(pred, targets[0], targets[1]) for negated, pred, t in plan.get_failed_preds(tol=1e-3)]
            exclude_targets.append(targets[0].name)

            if len(fixed_targets):
                break

        if not success:
            task_vec = np.zeros((len(self.task_list)), dtype=np.float32)
            task_vec[self.task_list.index(task)] = 1.
            obj_vec = np.zeros((len(self.obj_list)), dtype='float32')
            targ_vec = np.zeros((len(self.targ_list)), dtype='float32')
            obj_vec[self.obj_list.index(targets[0].name)] = 1.
            targ_vec[self.targ_list.index(targets[1].name)] = 1.
            target_vec = np.zeros((self.target_dim,))
            set_params_attrs(plan.params, plan.state_inds, state, 0)
            for target_name in self.targets[condition]:
                target = plan.params[target_name]
                target.value[:,0] = self.targets[condition][target.name]
                target_vec[self.target_inds[target.name, 'value']] = target.value[:,0]

            sample = Sample(self)
            sample.set(STATE_ENUM, state.copy(), 0)
            sample.set(TASK_ENUM, task_vec, 0)
            sample.set(OBJ_ENUM, obj_vec, 0)
            sample.set(TARG_ENUM, targ_vec, 0)
            sample.set(TRAJ_HIST_ENUM, np.array(self.traj_hist).flatten(), 0)
            sample.set(TARGETS_ENUM, target_vec, 0)
            sample.condition = condition
            sample.task = task
            return sample, failed_preds, success

        class optimal_pol:
            def act(self, X, O, t, noise):
                U = np.zeros((plan.dU), dtype=np.float32)
                if t < plan.horizon - 1:
                    fill_vector(plan.params, plan.action_inds, U, t+1)
                else:
                    fill_vector(plan.params, plan.action_inds, U, t)
                return U

        sample = self.sample_task(optimal_pol(), condition, state, [task, targets[0].name, targets[1].name], noisy=False)
        self.optimal_samples[task].append(sample)
        return sample, failed_preds, success


    def get_hl_plan(self, state, condition, failed_preds):
        return self._get_hl_plan(state, self.targets[condition], self.plans.values()[0].params, self.state_inds, failed_preds)


    def update_targets(self, targets, condition):
        self.targets[condition] = targets


    def get_sample_constr_cost(self, sample):
        obj = self.plans.values()[0].params[self.obj_list[np.argmax(sample.get(OBJ_ENUM, t=0))]]
        targ = self.plans.values()[0].params[self.targ_list[np.argmax(sample.get(TARG_ENUM, t=0))]]
        targets = [obj, targ]
        # targets = get_next_target(self.plans.values()[0], sample.get(STATE_ENUM, t=0), sample.task, self.targets[sample.condition])
        plan = self.plans[sample.task, targets[0].name]
        for t in range(sample.T):
            set_params_attrs(plan.params, plan.state_inds, sample.get(STATE_ENUM, t=t), t)

        for param_name in plan.params:
            param = plan.params[param_name]
            if param._type == 'Cloth' and '{0}_init_target'.format(param_name) in plan.params:
                plan.params['{0}_init_target'.format(param_name)].value[:,0] = plan.params[param_name].pose[:,0]

        for target in targets:
            if target.name in self.targets[sample.condition]:
                plan.params[target.name].value[:,0] = self.targets[sample.condition][target.name]

        plan.params['robot_init_pose'].lArmPose[:,0] = plan.params['baxter'].lArmPose[:,0]
        plan.params['robot_init_pose'].lGripper[:,0] = plan.params['baxter'].lGripper[:,0]
        plan.params['robot_init_pose'].rArmPose[:,0] = plan.params['baxter'].rArmPose[:,0]
        plan.params['robot_init_pose'].rGripper[:,0] = plan.params['baxter'].rGripper[:,0]

        return check_constr_violation(plan, exclude=['BaxterRobotAt'])


    def replace_conditions(self, conditions, keep=(0.2, 0.5)):
        self.targets = []
        for i in range(conditions):
            self.targets.append(get_end_targets(self.num_cans))
        self.init_vecs = get_random_initial_state_vec(self.num_cans, self.targets, self.dX, self.state_inds, conditions)
        self.x0 = [x[:self.symbolic_bound] for x in self.init_vecs]
        self.target_vecs = []
        for condition in range(len(self.x0)):
            target_vec = np.zeros((self.target_dim,))
            for target_name in self.targets[condition]:
                if (target_name, 'value') in self.target_inds:
                    target_vec[self.target_inds[target_name, 'value']] = self.targets[condition][target_name]
            self.target_vecs.append(target_vec)

        if keep != (1., 1.):
            self.clear_samples(*keep)
