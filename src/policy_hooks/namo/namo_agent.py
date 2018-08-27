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
from policy_hooks.namo.sorting_prob import *


class NAMOSortingAgent(Agent):
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


    def sample_task(self, policy, condition, x0, task, save_global=False, verbose=False, use_base_t=True, noisy=True):
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

            U = policy.act(sample.get_X(t=t), sample.get_obs(t=t), t, noise[t])
            robot_start = X[plan.state_inds['pr2', 'pose']]
            robot_vec = U[plan.action_inds['pr2', 'pose']] - robot_start
            if np.sum(np.abs(robot_vec)) != 0 and np.linalg.norm(robot_vec) < 0.005:
                U[plan.action_inds['pr2', 'pose']] = robot_start + 0.1 * robot_vec / np.linalg.norm(robot_vec)
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
            for param, attr in u_inds:
                getattr(plan.params[param], attr)[:, t+1] = u[u_inds[param, attr]]

            for param in plan.params.values():
                if param._type == 'Can':
                    dist = plan.params['pr2'].pose[:, t] - plan.params[param.name].pose[:, t]
                    radius1 = param.geom.radius
                    radius2 = plan.params['pr2'].geom.radius
                    grip_dist = radius1 + radius2 + dsafe
                    if plan.params['pr2'].gripper[0, t] > 0.2 and np.abs(dist[0]) < 0.05 and np.abs(grip_dist + dist[1]) < 0.05:
                        param.pose[:, t+1] = plan.params['pr2'].pose[:, t+1] + [0, grip_dist+dsafe]
                    elif param._type == 'Can':
                        param.pose[:, t+1] = param.pose[:, t]

        return True


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
            
            plan.params['robot_init_pose'].value[:,0] = plan.params['pr2'].pose[:,0]
            dist = plan.params['pr2'].geom.radius + targets[0].geom.radius + dsafe
            plan.params['robot_end_pose'].value[:,0] = plan.params[targets[1].name].value[:,0] - [0, dist]
            if task == 'grasp':
                plan.params['robot_end_pose'].value[:,0] = plan.params[targets[1].name].value[:,0] - [0, dist+0.2]
            # self.env.SetViewer('qtcoin')
            try:
                success = self.solver._backtrack_solve(plan, n_resamples=5, traj_mean=traj_mean)
                # viewer = OpenRAVEViewer._viewer if OpenRAVEViewer._viewer is not None else OpenRAVEViewer(plan.env)
                # import ipdb; ipdb.set_trace()
                # if task == 'putdown':
                #     import ipdb; ipdb.set_trace()
                # self.env.SetViewer('qtcoin')
                # import ipdb; ipdb.set_trace()
            except Exception as e:
                print e
                # self.env.SetViewer('qtcoin')
                # import ipdb; ipdb.set_trace()
                success = False

            if not len(failed_preds):
                failed_preds = [(pred, targets[0], targets[1]) for negated, pred, t in plan.get_failed_preds(tol=1e-3)]
            exclude_targets.append(targets[0].name)

            if len(fixed_targets):
                break


        if not success:
            return None, failed_preds

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
        return sample, failed_preds


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
            if param._type == 'Can' and '{0}_init_target'.format(param_name) in plan.params:
                plan.params['{0}_init_target'.format(param_name)].value[:,0] = plan.params[param_name].pose[:,0]

        for target in targets:
            if target.name in self.targets[sample.condition]:
                plan.params[target.name].value[:,0] = self.targets[sample.condition][target.name]

        plan.params['robot_init_pose'].value[:,0] = plan.params['pr2'].pose[:,0]
        dist = plan.params['pr2'].geom.radius + targets[0].geom.radius + dsafe
        plan.params['robot_end_pose'].value[:,0] = plan.params[targets[1].name].value[:,0] - [0, dist]

        return check_constr_violation(plan)