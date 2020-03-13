"""
Defines utility functions for planning in the sorting domain
"""
import copy
from collections import OrderedDict
import itertools
import numpy as np
import random
import time

from core.internal_repr.plan import Plan
from core.util_classes.namo_predicates import dsafe
from pma.hl_solver import FFSolver
from policy_hooks.utils.load_task_definitions import get_tasks, plan_from_str
from policy_hooks.utils.policy_solver_utils import *
import policy_hooks.utils.policy_solver_utils as utils

NUM_OBJS = 8
NUM_TARGS = 4
SORT_CLOSET = False
USE_PERTURB = False
OPT_MCTS_FEEDBACK = True

prob_file = "../domains/namo_domain/namo_probs/sort_closet_prob_{0}.prob".format(NUM_OBJS)
domain_file = "../domains/namo_domain/namo.domain"
mapping_file = "policy_hooks/namo/sorting_task_mapping_3"
pddl_file = "../domains/namo_domain/sorting_domain_3.pddl"

descriptor = 'namo_{0}_obj_sort_closet_{1}_perturb_{2}_feedback_to_tree_{3}'.format(NUM_OBJS, SORT_CLOSET, USE_PERTURB, OPT_MCTS_FEEDBACK)

# END_TARGETS = [(0., 5.8), 
#            (0., 5.), 
#            (0., 4.), 
#            (2., -2.), 
#            (0., -2.),
#            (4., 0.),
#            (-4, 0.),
#            (4., -2.),
#            (-4., -2.),
#            (-2., -2.)]

END_TARGETS =[(0., 5.8), (0., 5.), (0., 4.)] if SORT_CLOSET else []
END_TARGETS.extend([(2., 1.5), 
                   (1., 1.5),
                   (-1., 1.5),
                   (-2, 1.5),
                   (-2.8, 1.5),
                   (2.8, 1.5),
                   (3.6, 1.5),
                   (-3.6, 1.5),
                   (4.4, 1.5),
                   (-4.4, 1.5)
                   ])

possible_can_locs = [(0, 57), (0, 50), (0, 43), (0, 35)] if SORT_CLOSET else []
MAX_Y = 25 if SORT_CLOSET else 10
# possible_can_locs.extend(list(itertools.product(range(-45, 45, 2), range(-35, MAX_Y, 2))))
possible_can_locs.extend(list(itertools.product(range(-45, 46, 2), range(-50, MAX_Y, 2))))

# for i in range(-25, 25):
for i in range(-12, 13):
    for j in range(-50, 11):
        if (i, j) in possible_can_locs:
            possible_can_locs.remove((i, j))
            
for i in range(len(possible_can_locs)):
    loc = list(possible_can_locs[i])
    loc[0] *= 0.1
    loc[1] *= 0.1
    possible_can_locs[i] = tuple(loc)

if not SORT_CLOSET:
    for target in END_TARGETS:
        if target in possible_can_locs:
            possible_can_locs.remove(target)


def get_prim_choices():
    out = OrderedDict({})
    out[utils.TASK_ENUM] = get_tasks(mapping_file).keys()
    out[utils.OBJ_ENUM] = ['can{0}'.format(i) for i in range(NUM_OBJS)]
    out[utils.TARG_ENUM] = ['middle_target', 
                            'left_target_1', 
                            # 'left_target_2', 
                            'right_target_1',
                            # 'right_target_2'
                            ] + ['can{0}_end_target'.format(i) for i in range(NUM_OBJS)]
    return out


def get_vector(config):
    state_vector_include = {
        'pr2': ['pose', 'gripper'] ,
    }
    for i in range(NUM_OBJS):
        state_vector_include['can{0}'.format(i)] = ['pose']

    action_vector_include = {
        'pr2': ['pose', 'gripper']
    }

    target_vector_include = {
        'can{0}_end_target'.format(i): ['value'] for i in range(NUM_OBJS)
    }
    target_vector_include['middle_target'] = ['value']
    target_vector_include['left_target_1'] = ['value']
    # target_vector_include['left_target_2'] = ['value']
    target_vector_include['right_target_1'] = ['value']
    # target_vector_include['right_target_2'] = ['value']

    return state_vector_include, action_vector_include, target_vector_include


def get_random_initial_state_vec(config, plans, dX, state_inds, conditions):
    # Information is track by the environment
    x0s = []
    for i in range(conditions):
        x0 = np.zeros((dX,))
        x0[state_inds['pr2', 'pose']] = np.random.uniform([-1, -5], [1, 1]) # [0, -2]
        can_locs = copy.deepcopy(END_TARGETS) if SORT_CLOSET else copy.deepcopy(possible_can_locs)
        # can_locs = copy.deepcopy(END_TARGETS)
        locs = []
        ide = np.random.uniform(1e4)
        while len(locs) < NUM_OBJS:
            random.shuffle(can_locs)
            #print('gen for', ide)
            valid = [1 for _ in range(len(can_locs))]
            for j in range(NUM_OBJS):
                for n in range(len(can_locs)):
                    if valid[n]:
                        locs.append(can_locs[n])
                        valid[n] = 0
                        for m in range(len(can_locs)):
                            if not valid[m] or n==m: continue
                            if np.linalg.norm(np.array(can_locs[n]) - np.array(can_locs[m])) < 1.5:
                                valid[m] = 0
                        break

        can_locs = locs
        for j in range(NUM_OBJS):
            x0[state_inds['can{0}'.format(j), 'pose']] = can_locs[j]
        print(x0)
        x0s.append(x0)
    return x0s

def parse_hl_plan(hl_plan):
    plan = []
    for i in range(len(hl_plan)):
        action = hl_plan[i]
        act_params = action.split()
        task = act_params[1].lower()
        next_params = [p.lower() for p in act_params[2:]]
        plan.append((task, next_params))
    return plan

# def get_plans():
#     tasks = get_tasks(mapping_file)
#     prim_options = get_prim_choices()
#     plans = {}
#     openrave_bodies = {}
#     env = None
#     for task in tasks:
#         next_task_str = copy.deepcopy(tasks[task])
#         plan = plan_from_str(next_task_str, prob_file, domain_file, env, openrave_bodies)

#         for i in range(len(prim_options[utils.OBJ_ENUM])):
#             for j in range(len(prim_options[utils.TARG_ENUM])):
#                 plans[(tasks.keys().index(task), i, j)] = plan
#         if env is None:
#             env = plan.env
#             for param in plan.params.values():
#                 if not param.is_symbol() and param.openrave_body is not None:
#                     openrave_bodies[param.name] = param.openrave_body
#     return plans, openrave_bodies, env

def get_plans():
    tasks = get_tasks(mapping_file)
    prim_options = get_prim_choices()
    plans = {}
    openrave_bodies = {}
    env = None
    for task in tasks:
        next_task_str = copy.deepcopy(tasks[task])
        for i in range(len(prim_options[utils.OBJ_ENUM])):
            for j in range(len(prim_options[utils.TARG_ENUM])):
                obj = prim_options[utils.OBJ_ENUM][i]
                targ = prim_options[utils.TARG_ENUM][j]
                new_task_str = []
                for step in next_task_str:
                    new_task_str.append(step.format(obj, targ))
                plan = plan_from_str(new_task_str, prob_file, domain_file, env, openrave_bodies)
                plans[(tasks.keys().index(task), i, j)] = plan
                if env is None:
                    env = plan.env
                    for param in plan.params.values():
                        if not param.is_symbol() and param.openrave_body is not None:
                            openrave_bodies[param.name] = param.openrave_body
    return plans, openrave_bodies, env

def get_end_targets(num_cans=NUM_OBJS, num_targs=NUM_OBJS, targs=None, randomize=False):
    target_map = {}
    inds = np.random.permutation(range(num_targs))
    for n in range(num_cans):
        if n > num_targs and targs is not None:
            target_map['can{0}_end_target'.format(n)] = np.array(targs[n])
        else:
            if randomize:
                ind = inds[n]
            else:
                ind = n

            target_map['can{0}_end_target'.format(n)] = np.array(END_TARGETS[ind])

    target_map['middle_target'] = np.array([0., 0.])
    target_map['left_target_1'] = np.array([-1., 0.])
    target_map['right_target_1'] = np.array([1., 0.])
    # target_map['left_target_2'] = np.array([-2., 0.])
    # target_map['right_target_2'] = np.array([2., 0.])
    return target_map

# CODE FROM OLDER VERSION OF PROB FILE BELOW THIS



def get_sorting_problem(can_locs, targets, pr2, grasp, failed_preds=[]):
    hl_plan_str = "(define (problem sorting_problem)\n"
    hl_plan_str += "(:domain sorting_domain)\n"

    hl_plan_str += "(:objects"
    for can in can_locs:
        hl_plan_str += " {0} - Can".format(can)

    for target in targets:
        hl_plan_str += " {0} - Target".format(target)

    hl_plan_str += ")\n"

    hl_plan_str += parse_initial_state(can_locs, targets, pr2, grasp, failed_preds)

    goal_state = {}
    goal_str = "(:goal (and"
    for i in range(len(can_locs)):
        goal_str += " (CanAtTarget can{0} can{1}_end_target)".format(i, i)
        goal_state["(CanAtTarget can{0} can{1}_end_target)".format(i, i)] = True

    goal_str += "))\n"

    hl_plan_str += goal_str

    hl_plan_str += "\n)"
    return hl_plan_str, goal_state

def parse_initial_state(can_locs, targets, pr2, grasp, failed_preds=[]):
    hl_init_state = "(:init "
    for can1 in can_locs:
        loc1 = can_locs[can1]
        t1 = targets[can1+'_end_target']
        if loc1[1] < 3.5: continue
        for can2 in can_locs:
            if can2 == can1: continue
            loc2 = can_locs[can2]
            t2 = targets[can2+'_end_target']
            if loc2[1] < 3.5: continue
            if loc1[1] < loc2[1]:
                hl_init_state += " (CanObstructs {0} {1})".format(can1, can2)
                hl_init_state += " (WaitingOnCan {0} {1})".format(can2, can1)
            else:
                hl_init_state += " (CanObstructs {0} {1})".format(can2, can1)
                hl_init_state += " (WaitingOnCan {0} {1})".format(can1, can2)

    for t1 in targets:
        loc1 = targets[t1]
        if loc1[1] < 3.5 or np.abs(loc1[0]) > 0.5: continue
        for t2 in targets:
            if t1 == t2: continue
            loc2 = targets[t2]
            if loc2[1] < 3.5 or np.abs(loc2[0]) > 0.5: continue
            if loc2[1] > loc1[1]:
                hl_init_state += " (InFront {0} {1})".format(t1, t2)



    for can in can_locs:
        loc = can_locs[can]

        hl_init_state += " (CanInReach {0})".format(can)

        closest_target = None
        closest_dist = np.inf
        for target in targets:
            if targets[target][1] > 3.5 \
               and np.abs(targets[target][0]) < 0.5 \
               and loc[1] > 3.5 \
               and loc[1] < targets[target][1] \
               and np.abs(loc[0]) < 0.5 \
               and target[3] != can[3]:
                hl_init_state += " (CanObstructsTarget {0} {1})".format(can, target)

            dist = np.sum((targets[target] - loc)**2)
            if dist < closest_dist:
                closest_dist = dist
                closest_target = target

        if closest_dist < 0.001:
            hl_init_state += " (CanAtTarget {0} {1})".format(can, closest_target)

        if np.all(np.abs(loc - pr2 + grasp) < 0.2):
            hl_init_state += " (CanInGripper {0})".format(can)

        if np.all(np.abs(loc - pr2 + grasp) < 1.0):
            hl_init_state += " (NearCan {0})".format(can)

    # Only mark the closest obstruction; it needs to be cleared first.
    for pred in failed_preds:
        if pred[0].get_type().lower() == 'obstructs':
            if " (CanObstructsTarget {0} {1})".format(pred[0].c.name, pred[2].name) not in hl_init_state:
                if pred[0].c.name != pred[1].name:
                    hl_init_state += " (CanObstructs {0} {1})".format(pred[0].c.name, pred[1].name)
                hl_init_state += " (CanObstructsTarget {0} {1})".format(pred[0].c.name, pred[2].name)
                break

        if pred[0].get_type().lower() == 'obstructsholding':
            if " (CanObstructsTarget {0} {1})".format(pred[0].obstr.name, pred[2].name) not in hl_init_state:
                if pred[0].obstr.name != pred[1].name:
                    hl_init_state += " (CanObstructs {0} {1})".format(pred[0].obstr.name, pred[1].name)
                hl_init_state += " (CanObstructsTarget {0} {1})".format(pred[0].obstr.name, pred[2].name)
                break


    hl_init_state += ")\n"
    # print hl_init_state
    return hl_init_state

def get_hl_plan(prob, plan_id):
    with open(pddl_file, 'r+') as f:
        domain = f.read()
    hl_solver = FFSolver(abs_domain=domain)
    return hl_solver._run_planner(domain, prob, 'namo_{0}'.format(plan_id))

def parse_hl_plan(hl_plan):
    plan = []
    for i in range(len(hl_plan)):
        action = hl_plan[i]
        act_params = action.split()
        task = act_params[1].lower()
        next_params = [p.lower() for p in act_params[2:]]
        plan.append((task, next_params))
    return plan

def hl_plan_for_state(state, targets, plan_id, param_map, state_inds, failed_preds=[]):
    can_locs = {}

    for param_name in param_map:
        param = param_map[param_name]
        if param_map[param_name]._type == 'Can':
            can_locs[param.name] = state[state_inds[param.name, 'pose']]

    prob, goal = get_sorting_problem(can_locs, targets, state[state_inds['pr2', 'pose']], param_map['grasp0'].value[:,0], failed_preds)
    hl_plan = get_hl_plan(prob, plan_id)
    if hl_plan == Plan.IMPOSSIBLE:
        # print 'Impossible HL plan for {0}'.format(prob)
        return []
    return parse_hl_plan(hl_plan)

def get_ll_plan_str(hl_plan, num_cans):
    tasks = get_tasks(mapping_file)
    ll_plan_str = []
    actions_per_task = []
    last_pose = "ROBOT_INIT_POSE"
    region_targets = {}
    for i in range(len(hl_plan)):
        action = hl_plan[i]
        act_params = action.split()
        next_task_str = copy.deepcopy(tasks[act_params[1].lower()])
        can = act_params[2].lower()
        if len(act_params) > 3:
            target = act_params[3].upper()
        else:
            target = "CAN{}_INIT_TARGET".format(can[-1])

        for j in range(len(next_task_str)):
            next_task_str[j]= next_task_str[j].format(can, target)
        ll_plan_str.extend(next_task_str)
        actions_per_task.append((len(next_task_str), act_params[1].lower()))
    return ll_plan_str, actions_per_task

def get_plan(num_cans):
    cans = ["Can{0}".format(i) for i in range(num_cans)]
    can_locs = get_random_initial_can_locations(num_cans)
    end_targets = get_can_end_targets(num_cans)
    prob, goal_state = get_sorting_problem(can_locs, end_targets)
    hl_plan = get_hl_plan(prob)
    ll_plan_str, actions_per_task = get_ll_plan_str(hl_plan, num_cans)
    plan = plan_from_str(ll_plan_str, prob_file.format(num_cans), domain_file, None, {})
    for i in range(len(can_locs)):
        plan.params['can{0}'.format(i)].pose[:,0] = can_locs[i]
        plan.params['can{0}_init_target'.format(i)].value[:,0] = plan.params['can{0}'.format(i)].pose[:,0]
        plan.params['can{0}_end_target'.format(i)].value[:, 0] = end_targets[i]

    task_timesteps = []
    cur_act = 0
    for i in range(len(hl_plan)):
        num_actions = actions_per_task[i][0]
        final_t = plan.actions[cur_act+num_actions-1].active_timesteps[1]
        task_timesteps.append((final_t, actions_per_task[i][1]))
        cur_act += num_actions

    plan.task_breaks = task_timesteps
    return plan, task_timesteps, goal_state


def fill_random_initial_configuration(plan):
    for param in plan.params:
        if plan.params[param]._Type == "Can":
            next_pos = random.choice(possible_can_locs)
            plan.params[param].pose[:,0] = next_pos

def get_random_initial_can_locations(num_cans):
    locs = []
    stop = False
    while not len(locs):
        locs = []
        for _ in range(num_cans):
            next_loc = random.choice(possible_can_locs)
            start = time.time()
            while len(locs) and np.any(np.abs(np.array(locs)[:,:2]-next_loc[:2]) < 0.6):
                next_loc = random.choice(possible_can_locs)
                if time.time() - start > 10:
                    locs = []
                    start = time.time()
                    stop = True
                    break

            if stop: break
            locs.append(next_loc)

    def compare_locs(a, b):
        if b[0] > a[0]: return 1
        if b[0] < a[0]: return -1
        if b[1] > a[1]: return 1
        if b[1] < a[1]: return -1
        return 0

    # locs.sort(compare_locs)

    return locs

def sorting_state_encode(state, plan, targets, task=(None, None, None)):
    pred_list = []
    for param_name in plan.params:
        param = plan.params[param_name]
        if param._type == 'Can':
            for target_name in targets:
                pred_list.append('CanAtTarget {0} {1}'.format(param_name, target_name))

    state_encoding = dict(zip(pred_list, range(len(pred_list))))
    hl_state = np.zeros((len(pred_list)))
    for param_name in plan.params:
        if plan.params[param_name]._type != 'Can': continue
        for target_name in targets:
            if np.all(np.abs(state[plan.state_inds[param_name, 'pose']] - targets[target_name]) < 0.1):
                hl_state[state_encoding['CanAtTarget {0} {1}'.format(param_name, target_name)]] = 1

    if task[0] is not None:
            for target_name in targets:
                hl_state[state_encoding['CanAtTarget {0} {1}'.format(task[1], target_name)]] = 0
            hl_state[state_encoding['CanAtTarget {0} {1}'.format(task[1], task[2])]] = 1

    return tuple(hl_state)

def get_plan_for_task(task, targets, num_cans, env, openrave_bodies):
    tasks = get_tasks(mapping_file)
    next_task_str = copy.deepcopy(tasks[task])
    for j in range(len(next_task_str)):
        next_task_str[j]= next_task_str[j].format(*targets)

    return plan_from_str(next_task_str, prob_file.format(num_cans), domain_file, env, openrave_bodies)
