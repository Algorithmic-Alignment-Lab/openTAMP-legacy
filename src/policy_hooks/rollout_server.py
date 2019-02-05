from datetime import datetime
import numpy as np
import sys
import time

from numba import cuda
import rospy
from scipy.cluster.vq import kmeans2 as kmeans
from std_msgs.msg import Float32MultiArray, String

from gps.sample.sample_list import SampleList

from tamp_ros.msg import *
from tamp_ros.srv import *

from policy_hooks.utils.policy_solver_utils import *


class DummyPolicy:
    def __init__(self, task, policy_call):
        self.task = task
        self.policy_call = policy_call

    def act(self, x, obs, t, noise):
        return self.policy_call(x, obs, t, noise, self.task)


class DummyPolicyOpt:
    def __init__(self, update, prob):
        self.update = update
        self.prob = prob


class RolloutServer(object):
    def __init__(self, hyperparams):
        self.id = hyperparams['id']
        rospy.init_node('rollout_server_'+str(self.id))
        self.mcts = hyperparams['mcts']
        self.prim_dims = hyperparams['prim_dims']
        self.agent = hyperparams['agent']['type'](hyperparams['agent'])
        for m in self.mcts:
            m.value_func = self.value_call
            m.prob_func = self.primitive_call
            m.agent = self.agent
        self.alg_map = hyperparams['alg_map']
        self.task_list = self.agent.task_list
        self.traj_opt_steps = hyperparams['traj_opt_steps']
        self.num_samples = hyperparams['num_samples']
        for mcts in self.mcts:
            mcts.num_samples = self.num_samples
        self.num_rollouts = hyperparams['num_rollouts']
        self.stopped = False

        self.updaters = {task: rospy.Publisher(task+'_update', PolicyUpdate, queue_size=50) for task in self.alg_map}
        self.updaters['value'] = rospy.Publisher('value_update', PolicyUpdate, queue_size=50)
        self.updaters['primitive'] = rospy.Publisher('primitive_update', PolicyUpdate, queue_size=50)
        self.mp_subcriber = rospy.Subscriber('motion_plan_result_'+str(self.id), MotionPlanResult, self.sample_mp)
        self.async_plan_publisher = rospy.Publisher('motion_plan_prob', MotionPlanProblem, queue_size=5)
        self.hl_subscriber = rospy.Subscriber('hl_result_'+str(self.id), HLPlanResult, self.update_hl)
        self.hl_publisher = rospy.Publisher('hl_prob', HLProblem, queue_size=10)
        self.weight_subscriber = rospy.Subscriber('tf_weights', String, self.store_weights, queue_size=1, buff_size=2**20)
        self.stop = rospy.Subscriber('terminate', String, self.end)

        for alg in self.alg_map.values():
            alg.policy_opt = DummyPolicyOpt(self.update, self.prob)
        self.n_optimizers = hyperparams['n_optimizers']
        self.waiting_for_opt = {}
        self.sample_queue = []
        self.current_id = 0
        self.cur_step = 0
        self.opt_samples = {task: [] for task in self.task_list}
        self.max_sample_queue = hyperparams['max_sample_queue']
        self.max_opt_sample_queue = hyperparams['max_opt_sample_queue']
        self.early_stop_prob = hyperparams['mcts_early_stop_prob']
        self.run_hl = hyperparams['run_hl'] if 'run_hl' in hyperparams else False
        self.opt_prob = hyperparams['opt_prob'] if 'opt_prob' in hyperparams else 0.1

        self.policy_proxies = {task: rospy.ServiceProxy(task+'_policy_act', PolicyAct, persistent=True) for task in self.task_list}
        self.value_proxy = rospy.ServiceProxy('qvalue', QValue, persistent=True)
        self.primitive_proxy = rospy.ServiceProxy('primitive', Primitive, persistent=True)
        self.prob_proxies = {task: rospy.ServiceProxy(task+'_policy_prob', PolicyProb, persistent=True) for task in self.task_list}
        self.mp_proxies = {mp_id: rospy.ServiceProxy('motion_planner_'+str(mp_id), MotionPlan, persistent=True) for mp_id in range(self.n_optimizers)}

        self.use_local = hyperparams['use_local']
        if self.use_local:
            hyperparams['policy_opt']['weight_dir'] = hyperparams['weight_dir'] + '_trained'
            hyperparams['policy_opt']['scope'] = None
            self.policy_opt = hyperparams['policy_opt']['type'](
                hyperparams['policy_opt'], 
                hyperparams['dO'],
                hyperparams['dU'],
                hyperparams['dPrimObs'],
                hyperparams['dValObs'],
                hyperparams['prim_bounds']
            )

        self.traj_centers = hyperparams['n_traj_centers']

        self.time_log = 'tf_saved/'+hyperparams['weight_dir']+'/timing_info.txt'
        self.log_timing = hyperparams['log_timing']


    def end(self, msg):
        self.stopped = True
        rospy.signal_shutdown('Received signal to terminate.')

    def update(self, obs, mu, prc, wt, task, rollout_len=0):
        msg = PolicyUpdate()
        msg.obs = obs.flatten()
        msg.mu = mu.flatten()
        msg.prc = prc.flatten()
        msg.wt = wt.flatten()
        msg.dO = self.agent.dO
        msg.dPrimObs = self.agent.dPrim
        msg.dValObs = self.agent.dVal
        msg.dU = mu.shape[-1]
        msg.n = len(mu)
        msg.rollout_len = mu.shape[1] if rollout_len < 1 else rollout_len
        self.updaters[task].publish(msg)

    def store_weights(self, msg):
        if self.use_local:
            self.policy_opt.deserialize_weights(msg.data)

    def policy_call(self, x, obs, t, noise, task):
        # print 'Entering policy call:', datetime.now()
        if self.use_local:
            if 'control' in self.policy_opt.task_map:
                if self.policy_opt.task_map['control']['policy'].scale is None:
                    return self.alg_map[task].cur[0].traj_distr.act(x.copy(), obs.copy(), t, noise)
                return self.policy_opt.task_map['control']['policy'].act(x.copy(), obs.copy(), t, noise)
            else:
                if self.policy_opt.task_map[task]['policy'].scale is None:
                    return self.alg_map[task].cur[0].traj_distr.act(x.copy(), obs.copy(), t, noise)
                return self.policy_opt.task_map[task]['policy'].act(x.copy(), obs.copy(), t, noise)

        rospy.wait_for_service(task+'_policy_act', timeout=10)
        req = PolicyActRequest()
        req.obs = obs
        req.noise = noise
        req.task = task
        resp = self.policy_proxies[task](req)
        # print 'Leaving policy call:', datetime.now()
        return np.array(resp.act)

    def value_call(self, obs):
        # print 'Entering value call:', datetime.now()
        if self.use_local:
            return self.policy_opt.value(obs)

        rospy.wait_for_service('qvalue', timeout=10)
        req = QValueRequest()
        req.obs = obs
        resp = self.value_proxy(req)
        # print 'Leaving value call:', datetime.now()
        return np.array(resp.value)

    def primitive_call(self, prim_obs):
        # print 'Entering primitive call:', datetime.now()
        if self.use_local:
            return self.policy_opt.task_distr(prim_obs)

        rospy.wait_for_service('primitive', timeout=10)
        req = PrimitiveRequest()
        req.prim_obs = prim_obs
        resp = self.primitive_proxy(req)
        # print 'Leaving primitive call:', datetime.now()
        return np.array(resp.task_distr), np.array(resp.obj_distr), np.array(resp.targ_distr)

    def prob(self, obs, task):
        # print 'Entering prob call:', datetime.now()
        if self.use_local:
            return self.policy_opt.prob(obs, task)

        rospy.wait_for_service(task+'_policy_prob', timeout=10)
        req_obs = []
        for i in range(len(obs)):
            next_line = Float32MultiArray()
            next_line.data = obs[i]
            req_obs.append(next_line)
        req = PolicyProbRequest()
        req.obs = req_obs
        req.task = task
        resp = self.prob_proxies[task](req)
        # print 'Leaving prob call:', datetime.now()
        return np.array([resp.mu[i].data for i in range(len(resp.mu))]), np.array([resp.sigma[i].data for i in range(len(resp.sigma))]), [], []

    def update_hl(self, msg):
        cond = msg.cond
        mcts = self.mcts[cond]
        path_to = eval(msg.path_to)
        samples = []
        for step in msg.steps:
            traj = np.array([step.traj[i].data for i in range(len(step.traj))])
            success = step.success
            task = step.task
            opt_sample = self.agent.sample_optimal_trajectory(traj[0], task, cond, traj, traj_mean=[])
            samples.append(opt_sample)
            self.store_opt_sample(opt_sample, -1)
            path_to.append((task))
        mcts.update_vals(path_to, msg.success)
        self.update_qvalue(samples)
        self.update_primitive(samples)
        if msg.success:
            self.early_stop_prob *= 0.975
        else:
            self.early_stop_prob *= 1.025

    def motion_plan(self, x, task, condition, traj_mean, targets):
        mp_id = np.random.randint(0, self.n_optimizers)
        mean = []
        for i in range(len(mean)):
            next_line = Float32MultiArray()
            next_line.data = traj_mean[i].tolist()
            mean.append(next_line)

        req = MotionPlanRequest()
        req.state = x.flatten()
        req.task = task
        req.obj = targets[0].name
        req.targ = targets[1].name
        req.condition = condition
        req.mean = mean

        resp = self.mp_proxies[mp_id](req)
        failed = eval(resp.failed)
        success = resp.success
        traj = np.array([resp.traj[i].data for i in range(len(resp.traj))])


    def store_for_opt(self, samples):
        self.waiting_for_opt[self.current_id] = samples
        self.sample_queue.append(self.current_id)
        self.current_id += 1
        while len(self.sample_queue) > self.max_sample_queue:
            del self.waiting_for_opt[self.sample_queue[0]]
            del self.sample_queue[0]


    def store_opt_sample(self, sample, plan_id):
        if plan_id in self.waiting_for_opt:
            samples = self.waiting_for_opt[plan_id]
            del self.waiting_for_opt[plan_id]
        else:
            samples = []

        for s in samples:
            s.set_ref_X(sample.get_ref_X())
        self.opt_samples[sample.task].append((sample, samples))
        while len(self.opt_samples[sample.task]) > self.max_opt_sample_queue:
            del self.opt_samples[sample.task][0]


    def sample_mp(self, msg):
        print 'Sampling optimal trajectory for rollout server {0}.'.format(self.id)
        plan_id = msg.plan_id
        traj = np.array([msg.traj[i].data for i in range(len(msg.traj))])
        success = msg.success
        task = eval(msg.task)
        condition = msg.cond
        if success:
            opt_sample = self.agent.sample_optimal_trajectory(traj[0], task, condition, traj, traj_mean=[])
            self.store_opt_sample(opt_sample, plan_id)


    def choose_mp_problems(self, samples):
        Xs = samples.get_X()[:,:,self.agent._x_data_idx[STATE_ENUM]]
        flat_Xs = Xs.reshape((Xs.shape[0], np.prod(Xs.shape[1:])))
        centroids, labels = kmeans(flat_Xs, k=self.traj_centers, minit='points')
        probs = []
        for c in range(len(centroids)):
            centroid = centroids[c]
            traj_mean = centroid.reshape(Xs.shape[1:])
            probs.append([traj_mean, []])

        for i in range(len(samples)):
            probs[labels[i]][1].append(samples[i])

        probs = filter(lambda p: len(p[1]), probs)

        for p in probs:
            p[1] = SampleList(p[1])


        return probs


    def send_mp_problem(self, centroid, s_list):
        next_sample = s_list[0]
        state = next_sample.get_X(t=0)
        task = next_sample.task
        cond = next_sample.condition
        traj_mean = []
        for t in range(next_sample.T):
            next_line = Float32MultiArray()
            next_line.data = centroid[t]
            traj_mean.append(next_line)
        prob = MotionPlanProblem()
        prob.state = state
        prob.task = str(task)
        prob.cond = cond
        prob.traj_mean = traj_mean
        prob.prob_id = self.current_id
        prob.solver_id = np.random.randint(0, self.n_optimizers)
        prob.server_id = self.id
        self.store_for_opt(s_list)

        if self.alg_map[next_sample.task_name].policy_prior.gmm.sigma is None:
            prob.use_prior = False
        else:
            gmm = self.alg_map[next_sample.task_name].mp_policy_prior.gmm
            prob.use_prior = True
            prob.mu = gmm.mu.flatten()
            prob.sigma = gmm.sigma.flatten()
            prob.logmass = gmm.logmass.flatten()
            prob.mass = gmm.mass.flatten()
            prob.N = len(gmm.mu)
            prob.K = len(gmm.mass)
            prob.Do = gmm.sigma.shape[1]

        print 'Sending motion plan problem to server {0}.'.format(prob.solver_id)
        self.async_plan_publisher.publish(prob)


    def step(self):
        print '\n\nTaking tree search step.\n\n'
        self.cur_step += 1
        rollout_policies = {task: DummyPolicy(task, self.policy_call) for task in self.agent.task_list}

        start_time = time.time()
        for mcts in self.mcts:
            val = mcts.run(self.agent.x0[mcts.condition], self.num_rollouts, use_distilled=False, new_policies=rollout_policies, debug=True)
            if self.run_hl and val > 0:
                init_state = self.agent.x0[mcts.condition]
                prob = HLProblem()
                prob.server_id = self.id
                prob.solver_id = np.random.randint(0, self.n_optimizers)
                prob.init_state = init_state.tolist()
                prob.cond = mcts.condition
                gmms = {}
                for task in self.task_list:
                    gmm = self.alg_map[task].mp_policy_prior.gmm
                    if gmm.sigma is None: continue
                    gmms[task] = {}
                    gmms[task]['mu'] = gmm.mu.flatten()
                    gmms[task]['sigma'] = gmm.sigma.flatten()
                    gmms[task]['logmass'] = gmm.logmass.flatten()
                    gmms[task]['mass'] = gmm.mass.flatten()
                    gmms[task]['N'] = len(gmm.mu)
                    gmms[task]['K'] = len(gmm.mass)
                    gmms[task]['Do'] = gmm.sigma.shape[1]
                prob.gmms = str(gmms)

                path = mcts.simulate(init_state, early_stop_prob=self.early_stop_prob)
                path_tuples = []
                for step in path:
                    path_tuples.append(step.task)
                prob.path_to = str(path_tuples)
                self.hl_publisher.publish(prob)

        end_time = time.time()

        sample_lists = {task: self.agent.get_samples(task) for task in self.task_list}
        self.agent.clear_samples(keep_prob=0.1, keep_opt_prob=0.2)
        all_samples = []

        n_probs = 0
        start_time_2 = time.time()
        for task in sample_lists:
            for s_list in sample_lists[task]:
                if np.random.uniform() < self.opt_prob:
                    all_samples.extend(s_list._samples)
                    probs = self.choose_mp_problems(s_list)
                    n_probs += len(probs)
                    for p in probs:
                        self.send_mp_problem(*p)
        end_time_2 = time.time()

        if self.log_timing:
            with open(self.time_log, 'a+') as f:
                f.write('Generated {0} problems from {1} conditions with {2} rollouts per condition.\n'.format(n_probs, len(self.mcts), self.num_rollouts))
                f.write('Time to complete: {0}\n'.format(end_time-start_time))
                f.write('Time to select problems through kmeans and send to supervised learner: {0}\n\n'.format(end_time_2-start_time_2))

        path_samples = []
        for path in self.agent.get_task_paths():
            path_samples.extend(path)

        self.update_primitive(path_samples)
        self.update_qvalue(all_samples)

        start_time = time.time()
        for task in self.agent.task_list:
            if len(self.opt_samples[task]):
                sample_lists[task] = self.alg_map[task].iteration(self.opt_samples[task], reset=False)

        # for step in range(self.traj_opt_steps-1):
        #     for task in self.agent.task_list:
        #         try:
        #             sample_lists[task] = self.alg_map[task].iteration(sample_lists[task], self.opt_samples[task], reset=True)
        #             if len(sample_lists[task]):
        #                 sample_lists[task] = self.agent.resample(sample_lists[task], rollout_policies[task], self.n_samples)
        #             else:
        #                 continue
        #             self.agent._samples[task] = sample_lists[task]
        #         except:
        #             traceback.print_exception(*sys.exc_info())
        # self.agent.reset_sample_refs()
        end_time = time.time()
        if self.log_timing:
            with open(self.time_log, 'a+') as f:
                f.write('Time to update algorithms for {0} iterations on data: {1}\n\n'.format(self.traj_opt_steps, end_time-start_time))

        print '\n\nFinished tree search step.\n\n'

    def run(self):
        while not self.stopped:
            self.step()

    def update_qvalue(self, samples, first_ts_only=False):
        dV, dO = 2, self.agent.dVal

        obs_data, tgt_mu = np.zeros((0, dO)), np.zeros((0, dV))
        tgt_prc, tgt_wt = np.zeros((0, dV, dV)), np.zeros((0))
        for sample in samples:
            if not hasattr(sample, 'success'): continue
            for t in range(sample.T):
                obs = [sample.get_val_obs(t=t)]
                mu = [sample.success]
                prc = [np.eye(dV)]
                wt = [10. / (t+1)]
                tgt_mu = np.concatenate((tgt_mu, mu))
                tgt_prc = np.concatenate((tgt_prc, prc))
                tgt_wt = np.concatenate((tgt_wt, wt))
                obs_data = np.concatenate((obs_data, obs))
                if first_ts_only: break

        if len(tgt_mu):
            self.update(obs_data, tgt_mu, tgt_prc, tgt_wt, 'value', 1)

    def update_primitive(self, samples):
        dP, dO = self.agent.dPrimOut, self.agent.dPrim
        # Compute target mean, cov, and weight for each sample.
        obs_data, tgt_mu = np.zeros((0, dO)), np.zeros((0, dP))
        tgt_prc, tgt_wt = np.zeros((0, dP, dP)), np.zeros((0))
        for sample in samples:
            for t in range(sample.T):
                obs = [sample.get_prim_obs(t=t)]
                mu = [np.concatenate([sample.get(enum, t=t) for enum in self.prim_dims])]
                prc = [np.eye(dP)]
                wt = [1.] # [np.exp(-sample.task_cost)]
                tgt_mu = np.concatenate((tgt_mu, mu))
                tgt_prc = np.concatenate((tgt_prc, prc))
                tgt_wt = np.concatenate((tgt_wt, wt))
                obs_data = np.concatenate((obs_data, obs))

        if len(tgt_mu):
            self.update(obs_data, tgt_mu, tgt_prc, tgt_wt, 'primitive', 1)
