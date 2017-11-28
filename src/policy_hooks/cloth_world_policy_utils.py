from core.parsing import parse_domain_config, parse_problem_config
from pma import hl_solver, robot_ll_solver

import numpy as np

import unittest, time, main


# BASKET_POSE = [0.7, 0.35, 0.875]
# BASKET_X_RANGE = [0.7, 0.85]
# BASKET_Y_RANGE = [0.7, 0.8]
# CLOTH_INIT_X_RANGE = [0.5, 0.9]
# CLOTH_INIT_Y_RANGE = [-0.15, 0.45]

BASKET_POSE = [0.65, 0.1, 0.875]
BASKET_X_RANGE = [0.6, 0.7]
BASKET_Y_RANGE = [0.075, 0.175]
CLOTH_INIT_X_RANGE = [0.35, 0.7]
CLOTH_INIT_Y_RANGE = [0.55, 0.9]

STEP_DELTA = 16
BASKET_STEP_DELTA = 4
TABLE_POSE = [1.23/2-0.1, 0, 0.97/2-0.375]
TABLE_GEOM = [1.23/2, 2.45/2, 0.97/2] # XYZ
TABLE_TOP = 0.97 - 0.375 + .02
BASKET_HEIGHT_DELTA = 0.02

R_ARM_PUTDOWN_END = [0, -0.25, 0, 0, 0, 0, 0]
L_ARM_PUTDOWN_END = [-0.6, -1.49792454, -0.35878011, 1.63006026, 0.02577696, 1.44332767, -0.17578484] # [-1., -1.11049898, -0.29706795, 1.29338713, 0.13218013, 1.40690655, -0.50397199]

FOUR_CLOTH_LOCATIONS = [
    [[ 0.68 ,  0.39 ,  0.615], [ 0.68 ,  0.75 ,  0.615], [ 0.74 ,  0.63 ,  0.615], [ 0.58 ,  0.71 ,  0.615]],
    [[0.4, 0.6, 0.615], [0.5, 0.4, 0.615], [0.53, 0.7, 0.615], [0.62, 0.42, 0.615]],
    [[0.35, 0.7, 0.615], [0.48, 0.6, 0.615], [0.55, 0.4, 0.615], [0.6, 0.45, 0.615]],
    [[0.35, 0.55, 0.615], [0.41, 0.8, 0.615], [0.43, 0.67, 0.615], [0.59, 0.47, 0.615]],
    [[0.36, 0.7, 0.615], [0.4, 0.68, 0.615], [0.47, 0.46, 0.615], [0.8, 0.7, 0.615]],
    [[0.3, 0.9, 0.615], [0.33, 0.6, 0.615], [0.4, 0.7, 0.615], [0.46, 0.6, 0.615]]
]

FOUR_CLOTH_BASKET_LOCATIONS = [
    [[0.05, 0.05], [0.05, -0.05], [-0.05, -0.05], [-0.05, 0.05]]
]

def generate_cond(num_cloths):
    i = 1
    act_num = 4
    plan_str = [
        '0: MOVETO BAXTER ROBOT_INIT_POSE CLOTH_GRASP_BEGIN_0',
        '1: CLOTH_GRASP BAXTER CLOTH_0 CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0 CLOTH_GRASP_END_0',
        '2: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_0 CLOTH_PUTDOWN_BEGIN_0 CLOTH_0',
        '3: PUT_INTO_BASKET BAXTER CLOTH_0 BASKET CLOTH_TARGET_END_0 INIT_TARGET CLOTH_PUTDOWN_BEGIN_0 CP_EE_0 CLOTH_PUTDOWN_END_0',
    ]

    while i < num_cloths:
        plan_str.append('{0}: MOVETO BAXTER CLOTH_PUTDOWN_END_{1} CLOTH_GRASP_BEGIN_{2}'.format(act_num, i-1, i))
        act_num += 1
        plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH_{1} CLOTH_TARGET_BEGIN_{2} CLOTH_GRASP_BEGIN_{3} CG_EE_{4} CLOTH_GRASP_END_{5}'.format(act_num, i, i, i, i, i))
        act_num += 1
        plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_{1} CLOTH_PUTDOWN_BEGIN_{2} CLOTH_{3}'.format(act_num, i, i, i))
        act_num += 1
        plan_str.append('{0}: PUT_INTO_BASKET BAXTER CLOTH_{1} BASKET CLOTH_TARGET_END_{2} INIT_TARGET CLOTH_PUTDOWN_BEGIN_{3} CP_EE_{4} CLOTH_PUTDOWN_END_{5}'.format(act_num, i, i, i, i, i))
        act_num += 1
        i += 1

    plan_str.append('{0}: MOVETO BAXTER CLOTH_PUTDOWN_END_{1} ROBOT_END_POSE'.format(act_num, i-1))

    domain_fname = '../domains/laundry_domain/laundry.domain'
    d_c = main.parse_file_to_dict(domain_fname)
    domain = parse_domain_config.ParseDomainConfig.parse(d_c)
    hls = hl_solver.FFSolver(d_c)
    p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/cloth_grasp_policy_{0}.prob'.format(num_cloths))
    problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
    # problem.init_state.params['cloth_target_begin_0'].value[:,0] = random_pose
    # problem.init_state.params['cloth_0'].pose[:,0] = random_pose

    plan = hls.get_plan(plan_str, domain, problem)

    basket = plan.params['basket']
    basket_target = plan.params['init_target']
    basket.pose[:,:] = np.array(BASKET_POSE).reshape(3,1)
    basket.rotation[:,:] = [[0], [0], [np.pi/2]]
    basket_target.value[:,:] = np.array(BASKET_POSE).reshape(3,1)
    basket_target.rotation[:,:] = [[0], [0], [np.pi/2]]

    plan.params['table'].pose[:,:] = np.array(TABLE_POSE).reshape(-1,1)
    plan.params['table'].rotation[:,:] = 0

    plan.params['robot_init_pose'].lArmPose[:,0] = L_ARM_PUTDOWN_END # [-0.1, -0.65, 0, 0, 0, 0, 0]
    plan.params['robot_init_pose'].lGripper[:,0] = 0.02
    plan.params['robot_init_pose'].rArmPose[:,0] = R_ARM_PUTDOWN_END # [0.1, -0.65, 0, 0, 0, 0, 0]
    plan.params['robot_init_pose'].rGripper[:,0] = 0.02

    plan.params['robot_end_pose'].lArmPose[:,0] = [1.4, 0.25, 0, 0.25, 0, 0.25, 0]
    plan.params['robot_end_pose'].lGripper[:,0] = 0
    plan.params['robot_end_pose'].rArmPose[:,0] = [-1.4, 0.25, 0, 0.25, 0, 0.25, 0]
    plan.params['robot_end_pose'].rGripper[:,0] = 0

    possible_locs = np.random.choice(range(0, 35*35, STEP_DELTA**2), num_cloths).tolist()

    for c in range(num_cloths):
        next_loc = possible_locs.pop()
        next_x = (next_loc / 35) / 100.0 + CLOTH_INIT_X_RANGE[0]
        next_y = (next_loc % 35) / 100.0 + CLOTH_INIT_Y_RANGE[0]
        plan.params['cloth_{0}'.format(c)].pose[:, 0] = [next_x, next_y, TABLE_TOP]
        plan.params['cloth_{0}'.format(c)].rotation[:, :] = 0
        plan.params['cloth_target_begin_{0}'.format(c)].value[:, 0] = [next_x, next_y, TABLE_TOP]
        plan.params['cloth_target_begin_{0}'.format(c)].rotation[:, :] = 0

    for c in range(num_cloths):
        plan.params['cg_ee_{0}'.format(c)].value[:,:] = np.nan
        plan.params['cg_ee_{0}'.format(c)].rotation[:,:] = np.nan
        
        plan.params['cp_ee_{0}'.format(c)].value[:,:] = np.nan
        plan.params['cp_ee_{0}'.format(c)].rotation[:,:] = np.nan

        plan.params['cloth_grasp_begin_{0}'.format(c)].value[:,:] = np.nan
        plan.params['cloth_grasp_begin_{0}'.format(c)].lArmPose[:,:] = np.nan
        plan.params['cloth_grasp_begin_{0}'.format(c)].lGripper[:,:] = np.nan
        plan.params['cloth_grasp_begin_{0}'.format(c)].rArmPose[:,:] = np.nan
        plan.params['cloth_grasp_begin_{0}'.format(c)].rGripper[:,:] = np.nan

        plan.params['cloth_grasp_end_{0}'.format(c)].value[:,:] = np.nan
        plan.params['cloth_grasp_end_{0}'.format(c)].lArmPose[:,:] = np.nan
        plan.params['cloth_grasp_end_{0}'.format(c)].lGripper[:,:] = np.nan
        plan.params['cloth_grasp_end_{0}'.format(c)].rArmPose[:,:] = np.nan
        plan.params['cloth_grasp_end_{0}'.format(c)].rGripper[:,:] = np.nan

        plan.params['cloth_putdown_begin_{0}'.format(c)].value[:,:] = np.nan
        plan.params['cloth_putdown_begin_{0}'.format(c)].lArmPose[:,:] = np.nan
        plan.params['cloth_putdown_begin_{0}'.format(c)].lGripper[:,:] = np.nan
        plan.params['cloth_putdown_begin_{0}'.format(c)].rArmPose[:,:] = np.nan
        plan.params['cloth_putdown_begin_{0}'.format(c)].rGripper[:,:] = np.nan

        plan.params['cloth_putdown_end_{0}'.format(c)].value[:,:] = np.nan
        plan.params['cloth_putdown_end_{0}'.format(c)].lArmPose[:,:] = np.nan
        plan.params['cloth_putdown_end_{0}'.format(c)].lGripper[:,:] = np.nan
        plan.params['cloth_putdown_end_{0}'.format(c)].rArmPose[:,:] = np.nan
        plan.params['cloth_putdown_end_{0}'.format(c)].rGripper[:,:] = np.nan
        # plan.params['cloth_putdown_end_{0}'.format(c)].lArmPose[:,:] = np.array(L_ARM_PUTDOWN_END).reshape((7,1))
        # plan.params['cloth_putdown_end_{0}'.format(c)].lGripper[:,:] = 0.02
        # plan.params['cloth_putdown_end_{0}'.format(c)].rArmPose[:,:] = np.array(R_ARM_PUTDOWN_END).reshape((7,1))
        # plan.params['cloth_putdown_end_{0}'.format(c)].rGripper[:,:] = 0.02

        plan.params['cloth_target_end_{0}'.format(c)].value[:,:] = np.nan
        plan.params['cloth_target_end_{0}'.format(c)].rotation[:,:] = 0

    plan._determine_free_attrs()

    return plan


def generate_move_cond(num_cloths):
    plan_str = [
        '0: MOVETO BAXTER ROBOT_INIT_POSE CLOTH_PUTDOWN_END_0',
        '1: MOVETO BAXTER CLOTH_PUTDOWN_END_0 ROBOT_END_POSE'
    ]

    domain_fname = '../domains/laundry_domain/laundry.domain'
    d_c = main.parse_file_to_dict(domain_fname)
    domain = parse_domain_config.ParseDomainConfig.parse(d_c)
    hls = hl_solver.FFSolver(d_c)
    p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/cloth_grasp_policy_{0}.prob'.format(num_cloths))
    problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)

    plan = hls.get_plan(plan_str, domain, problem)

    basket = plan.params['basket']
    basket_target = plan.params['init_target']
    basket.pose[:,:] = np.array(BASKET_POSE).reshape(3,1)
    basket.rotation[:,:] = [[0], [0], [np.pi/2]]
    basket_target.value[:,:] = np.array(BASKET_POSE).reshape(3,1)
    basket_target.rotation[:,:] = [[0], [0], [np.pi/2]]

    plan.params['table'].pose[:,:] = np.array(TABLE_POSE).reshape(-1,1)
    plan.params['table'].rotation[:,:] = 0

    plan.params['robot_init_pose'].lArmPose[:,0] = [-0.1, -0.65, 0, 0, 0, 0, 0]
    plan.params['robot_init_pose'].lGripper[:,0] = 0
    plan.params['robot_init_pose'].rArmPose[:,0] = [0.1, -0.65, 0, 0, 0, 0, 0]
    plan.params['robot_init_pose'].rGripper[:,0] = 0

    plan.params['robot_end_pose'].lArmPose[:,0] = [1.4, 0.25, 0, 0.25, 0, 0.25, 0]
    plan.params['robot_end_pose'].lGripper[:,0] = 0
    plan.params['robot_end_pose'].rArmPose[:,0] = [-1.4, 0.25, 0, 0.25, 0, 0.25, 0]
    plan.params['robot_end_pose'].rGripper[:,0] = 0

    for c in range(num_cloths):
        plan.params['cloth_putdown_end_{0}'.format(c)].lArmPose[:,:] = np.array(L_ARM_PUTDOWN_END).reshape((7,1))
        plan.params['cloth_putdown_end_{0}'.format(c)].lGripper[:,:] = 0.02
        plan.params['cloth_putdown_end_{0}'.format(c)].rArmPose[:,:] = np.array(R_ARM_PUTDOWN_END).reshape((7,1))
        plan.params['cloth_putdown_end_{0}'.format(c)].rGripper[:,:] = 0.02

    plan._determine_free_attrs()

    return plan


def get_randomized_initial_state(plan):
    num_cloths = 0
    while 'cloth_{0}'.format(num_cloths) in plan.params:
        num_cloths += 1

    X = np.zeros((plan.dX))

    X[plan.state_inds[('robot_end_pose', 'lArmPose')]] = plan.params['robot_end_pose'].lArmPose.flatten()
    X[plan.state_inds[('robot_end_pose', 'lGripper')]] = plan.params['robot_end_pose'].lGripper
    X[plan.state_inds[('robot_end_pose', 'rArmPose')]] = plan.params['robot_end_pose'].rArmPose.flatten()
    X[plan.state_inds[('robot_end_pose', 'rGripper')]] = plan.params['robot_end_pose'].rGripper

    basket = plan.params['basket']
    basket.pose[:,:] = [[np.random.uniform(BASKET_X_RANGE[0], BASKET_X_RANGE[1])],
                        [np.random.uniform(BASKET_Y_RANGE[0], BASKET_Y_RANGE[1])],
                        [BASKET_POSE[2]]]
    X[plan.state_inds[('basket', 'pose')]] = basket.pose[:,0]
    # X[plan.state_inds[('basket', 'rotation')]] = [0, 0, np.pi/2]
    # X[plan.state_inds[('table', 'pose')]] = TABLE_POSE
    # X[plan.state_inds[('table', 'rotation')]] = [0, 0, 0]
    X[plan.state_inds[('init_target', 'value')]] = basket.pose[:,0]
    X[plan.state_inds[('init_target', 'rotation')]] = [0, 0, np.pi/2]

    num_on_table = np.random.randint(1, num_cloths+1)
    possible_locs = np.random.choice(range(0, 35*35, STEP_DELTA**2), num_on_table).tolist()
    for c in range(num_cloths-1, num_cloths-num_on_table-1, -1):
        next_loc = possible_locs.pop()
        next_x = (next_loc / 35) / 100.0 + CLOTH_INIT_X_RANGE[0]
        next_y = (next_loc % 35) / 100.0 + CLOTH_INIT_Y_RANGE[0]
        X[plan.state_inds[('cloth_{0}'.format(c), 'pose')]] = [next_x, next_y, TABLE_TOP]
        X[plan.state_inds[('cloth_target_begin_{0}'.format(c), 'value')]] = [next_x, next_y, TABLE_TOP]

    possible_basket_locs = np.random.choice(range(0, 150, STEP_DELTA**2), num_cloths-num_on_table).tolist()

    stationary_params = ['basket']#, 'table']
    for c in range(num_cloths-num_on_table):
        next_x = (basket.pose[0,0] - 0.15) + (possible_basket_locs[c] / 10) / 100.0
        next_y = (basket.pose[1,0] - 0.3) + (possible_basket_locs[c] % 10) / 100.0
        X[plan.state_inds[('cloth_{0}'.format(c), 'pose')]] = [next_x, next_y, TABLE_TOP+BASKET_HEIGHT_DELTA]
        X[plan.state_inds[('cloth_target_begin_{0}'.format(c), 'value')]] = [next_x, next_y, TABLE_TOP+BASKET_HEIGHT_DELTA]
        X[plan.state_inds[('cloth_target_end_{0}'.format(c), 'value')]] = [next_x, next_y, TABLE_TOP+BASKET_HEIGHT_DELTA]
        stationary_params.append('cloth_{0}'.format(c))

    discard_actions = (num_cloths - num_on_table) * 4
    return X, [discard_actions, discard_actions+4], stationary_params


def get_randomized_initial_state_move(plan):
    # num_cloths = 0
    # while 'cloth_{0}'.format(num_cloths) in plan.params:
    #     num_cloths += 1

    X = np.zeros((plan.dX))

    robot = plan.params['baxter']
    left_ik_poses = []
    right_ik_poses = []
    while not len(left_ik_poses):
        random_pose = [np.random.uniform(0.35, 0.9), np.random.uniform(0, 1.0), np.random.uniform(0.65, 1.25)]
        random_rot = np.random.uniform(0, np.pi/2, (3,))
        left_ik_poses = robot.openrave_body.get_ik_from_pose(random_pose, random_rot, "left_arm")
    while not len(right_ik_poses):
        random_pose = [np.random.uniform(0.35, 0.9), np.random.uniform(-1.0, 0), np.random.uniform(0.65, 1.25)]
        random_rot = np.random.uniform(0, np.pi/2, (3,))
        right_ik_poses = robot.openrave_body.get_ik_from_pose(random_pose, random_rot, "right_arm")
    left_arm_start = left_ik_poses[0]
    right_arm_start = right_ik_poses[0]

    X[plan.state_inds[('robot_end_pose', 'lArmPose')]] = plan.params['robot_end_pose'].lArmPose.flatten()
    X[plan.state_inds[('robot_end_pose', 'lGripper')]] = plan.params['robot_end_pose'].lGripper
    X[plan.state_inds[('robot_end_pose', 'rArmPose')]] = plan.params['robot_end_pose'].rArmPose.flatten()
    X[plan.state_inds[('robot_end_pose', 'rGripper')]] = plan.params['robot_end_pose'].rGripper

    X[plan.state_inds[('robot_init_pose', 'lArmPose')]] = left_arm_start
    X[plan.state_inds[('robot_init_pose', 'lGripper')]] = 0
    X[plan.state_inds[('robot_init_pose', 'rArmPose')]] = right_arm_start
    X[plan.state_inds[('robot_init_pose', 'rGripper')]] = 0

    X[plan.state_inds[('baxter', 'lArmPose')]] = left_arm_start
    X[plan.state_inds[('baxter', 'lGripper')]] = 0
    X[plan.state_inds[('baxter', 'rArmPose')]] = right_arm_start
    X[plan.state_inds[('baxter', 'rGripper')]] = 0

    basket = plan.params['basket']
    basket.pose[:,:] = [[np.random.uniform(BASKET_X_RANGE[0], BASKET_X_RANGE[1])],
                        [np.random.uniform(BASKET_Y_RANGE[0], BASKET_Y_RANGE[1])],
                        [BASKET_POSE[2]]]
    X[plan.state_inds[('basket', 'pose')]] = basket.pose[:,0]
    # X[plan.state_inds[('basket', 'rotation')]] = [0, 0, np.pi/2]
    # X[plan.state_inds[('table', 'pose')]] = TABLE_POSE
    # X[plan.state_inds[('table', 'rotation')]] = [0, 0, 0]
    X[plan.state_inds[('init_target', 'value')]] = basket.pose[:,0]
    X[plan.state_inds[('init_target', 'rotation')]] = [0, 0, np.pi/2]

    # num_on_table = np.random.randint(1, num_cloths+1)
    # possible_locs = np.random.choice(range(0, 40*60, STEP_DELTA**2), num_on_table).tolist()
    # for c in range(num_cloths-1, num_cloths-num_on_table-1, -1):
    #     next_loc = possible_locs.pop()
    #     next_x = (next_loc / 60) / 100.0 + CLOTH_INIT_X_RANGE[0]
    #     next_y = (next_loc % 60) / 100.0 + CLOTH_INIT_Y_RANGE[0]
    #     X[plan.state_inds[('cloth_{0}'.format(c), 'pose')]] = [next_x, next_y, TABLE_TOP]
    #     X[plan.state_inds[('cloth_target_begin_{0}'.format(c), 'value')]] = [next_x, next_y, TABLE_TOP]

    # possible_basket_locs = np.random.choice(range(0, 144, STEP_DELTA**2), num_cloths-num_on_table).tolist()

    stationary_params = ['basket']#, 'table']
    # for c in range(num_cloths-num_on_table):
    #     next_x = (basket.pose[0,0] - 0.12) + (possible_basket_locs[c] / 24) / 100.0
    #     next_y = (basket.pose[1,0] - 0.12) + (possible_basket_locs[c] % 24) / 100.0
    #     X[plan.state_inds[('cloth_{0}'.format(c), 'pose')]] = [next_x, next_y, TABLE_TOP+BASKET_HEIGHT_DELTA]
    #     X[plan.state_inds[('cloth_target_begin_{0}'.format(c), 'value')]] = [next_x, next_y, TABLE_TOP+BASKET_HEIGHT_DELTA]
    #     X[plan.state_inds[('cloth_target_end_{0}'.format(c), 'value')]] = [next_x, next_y, TABLE_TOP+BASKET_HEIGHT_DELTA]
    #     stationary_params.append('cloth_{0}'.format(c))

    return (X, [0, 1], stationary_params)


def get_randomized_initial_state_multi_step(plan, plan_num):
    num_cloths = 0
    while 'cloth_{0}'.format(num_cloths) in plan.params:
        num_cloths += 1

    X = np.zeros((num_cloths, plan.dX))
    X_0s = []

    success = False
    while not success:
        print 'Searching for initial configuration...'
        X_0s = []

        X[:, plan.state_inds[('robot_end_pose', 'lArmPose')]] = plan.params['robot_end_pose'].lArmPose.flatten()
        X[:, plan.state_inds[('robot_end_pose', 'lGripper')]] = plan.params['robot_end_pose'].lGripper
        X[:, plan.state_inds[('robot_end_pose', 'rArmPose')]] = plan.params['robot_end_pose'].rArmPose.flatten()
        X[:, plan.state_inds[('robot_end_pose', 'rGripper')]] = plan.params['robot_end_pose'].rGripper

        X[:, plan.state_inds[('robot_init_pose', 'lArmPose')]] = plan.params['robot_init_pose'].lArmPose.flatten()
        X[:, plan.state_inds[('robot_init_pose', 'lGripper')]] = plan.params['robot_init_pose'].lGripper
        X[:, plan.state_inds[('robot_init_pose', 'rArmPose')]] = plan.params['robot_init_pose'].rArmPose.flatten()
        X[:, plan.state_inds[('robot_init_pose', 'rGripper')]] = plan.params['robot_init_pose'].rGripper

        X[0, plan.state_inds[('baxter', 'lArmPose')]] = plan.params['robot_init_pose'].lArmPose.flatten()
        X[0, plan.state_inds[('baxter', 'lGripper')]] = plan.params['robot_init_pose'].lGripper
        X[0, plan.state_inds[('baxter', 'rArmPose')]] = plan.params['robot_init_pose'].rArmPose.flatten()
        X[0, plan.state_inds[('baxter', 'rGripper')]] = plan.params['robot_init_pose'].rGripper

        basket = plan.params['basket']
        basket.pose[:,:] = [[np.random.uniform(BASKET_X_RANGE[0], BASKET_X_RANGE[1])],
                            [np.random.uniform(BASKET_Y_RANGE[0], BASKET_Y_RANGE[1])],
                            [BASKET_POSE[2]]]
        X[:, plan.state_inds[('basket', 'pose')]] = basket.pose[:,0]
        X[:, plan.state_inds[('init_target', 'value')]] = basket.pose[:,0]
        X[:, plan.state_inds[('init_target', 'rotation')]] = [0, 0, np.pi/2]

        possible_locs = np.random.choice(range(0, 35*35, STEP_DELTA**2), num_cloths).tolist()
        possible_basket_locs = np.random.choice(range(0, 144, BASKET_STEP_DELTA**2), num_cloths).tolist()

        success = True
        plan.params['table'].openrave_body.set_pose([10,10,10])
        plan.params['basket'].openrave_body.set_pose([-10,10,10])
        for c in range(num_cloths-1, -1, -1):
            next_loc = possible_locs.pop()
            next_x = (next_loc / 35) / 100.0 + CLOTH_INIT_X_RANGE[0]
            next_y = (next_loc % 35) / 100.0 + CLOTH_INIT_Y_RANGE[0]
            X[0, plan.state_inds[('cloth_{0}'.format(c), 'pose')]] = [next_x, next_y, TABLE_TOP]
            X[:, plan.state_inds[('cloth_target_begin_{0}'.format(c), 'value')]] = [next_x, next_y, TABLE_TOP]

            arm_poses = plan.params['baxter'].openrave_body.get_ik_from_pose([next_x, next_y, TABLE_TOP + 0.05], [0, np.pi/2, 0], "left_arm")
            if not len(arm_poses): success = False

            next_x = (basket.pose[0,0] - 0.12) + (possible_basket_locs[c] / 24) / 100.0
            next_y = (basket.pose[1,0] - 0.12) + (possible_basket_locs[c] % 24) / 100.0
            X[:, plan.state_inds[('cloth_target_end_{0}'.format(c), 'value')]] = [next_x, next_y, TABLE_TOP]

            arm_poses = plan.params['baxter'].openrave_body.get_ik_from_pose([next_x, next_y, TABLE_TOP + 0.05], [0, np.pi/2, 0], "left_arm")
            if not len(arm_poses): success = False

            X[:, plan.state_inds[('cloth_putdown_end_{0}'.format(c), 'lArmPose')]] = L_ARM_PUTDOWN_END
            X[:, plan.state_inds[('cloth_putdown_end_{0}'.format(c), 'lGripper')]] = 0.02
            X[:, plan.state_inds[('cloth_putdown_end_{0}'.format(c), 'rArmPose')]] = R_ARM_PUTDOWN_END
            X[:, plan.state_inds[('cloth_putdown_end_{0}'.format(c), 'rGripper')]] = 0.02

        stationary_params = ['basket']
        X_0s.append((X[0], [0, 3], stationary_params, plan_num))

        for i in range(1, num_cloths):
            num_on_table = num_cloths - i

            X[i, plan.state_inds[('baxter', 'lArmPose')]] = L_ARM_PUTDOWN_END
            X[i, plan.state_inds[('baxter', 'lGripper')]] = 0.02
            X[i, plan.state_inds[('baxter', 'rArmPose')]] = R_ARM_PUTDOWN_END
            X[i, plan.state_inds[('baxter', 'rGripper')]] = 0.02

            for c in range(num_cloths-1, num_cloths-num_on_table-1, -1):
                X[i, plan.state_inds[('cloth_{0}'.format(c), 'pose')]] = X[i, plan.state_inds[('cloth_target_begin_{0}'.format(c), 'value')]]

            stationary_params = ['basket']
            for c in range(num_cloths-num_on_table):
                X[i, plan.state_inds[('cloth_{0}'.format(c), 'pose')]] = X[i, plan.state_inds[('cloth_target_end_{0}'.format(c), 'value')]]
                stationary_params.append('cloth_{0}'.format(c))

            discard_actions = (num_cloths - num_on_table) * 4
            X_0s.append((X[i], [i*4, i*4+3], stationary_params, plan_num))

    print "Found initial configuration.\n"
    return X_0s


def get_randomized_initial_state_multi_step_left(plan, plan_num):
    num_cloths = 0
    while 'cloth_{0}'.format(num_cloths) in plan.params:
        num_cloths += 1

    X = np.zeros((num_cloths, plan.dX))
    X_0s = []

    success = False
    while not success:
        print 'Searching for initial configuration...'
        X_0s = []

        X[:, plan.state_inds[('robot_end_pose', 'lArmPose')]] = plan.params['robot_end_pose'].lArmPose.flatten()
        X[:, plan.state_inds[('robot_end_pose', 'lGripper')]] = plan.params['robot_end_pose'].lGripper
        X[:, plan.state_inds[('robot_end_pose', 'rArmPose')]] = np.zeros((7,))
        X[:, plan.state_inds[('robot_end_pose', 'rGripper')]] = 0

        X[:, plan.state_inds[('robot_init_pose', 'lArmPose')]] = plan.params['robot_init_pose'].lArmPose.flatten()
        X[:, plan.state_inds[('robot_init_pose', 'lGripper')]] = plan.params['robot_init_pose'].lGripper
        X[:, plan.state_inds[('robot_init_pose', 'rArmPose')]] = np.zeros((7,))
        X[:, plan.state_inds[('robot_init_pose', 'rGripper')]] = 0

        X[0, plan.state_inds[('baxter', 'lArmPose')]] = plan.params['robot_init_pose'].lArmPose.flatten()
        X[0, plan.state_inds[('baxter', 'lGripper')]] = plan.params['robot_init_pose'].lGripper
        X[0, plan.state_inds[('baxter', 'rArmPose')]] = np.zeros((7,))
        X[0, plan.state_inds[('baxter', 'rGripper')]] = 0

        basket = plan.params['basket']
        basket.pose[:,:] = [[np.random.uniform(BASKET_X_RANGE[0], BASKET_X_RANGE[1])],
                            [np.random.uniform(BASKET_Y_RANGE[0], BASKET_Y_RANGE[1])],
                            [BASKET_POSE[2]]]
        X[:, plan.state_inds[('basket', 'pose')]] = basket.pose[:,0]
        X[:, plan.state_inds[('init_target', 'value')]] = basket.pose[:,0]
        X[:, plan.state_inds[('init_target', 'rotation')]] = [0, 0, np.pi/2]

        possible_locs = np.random.choice(range(0, 35*35, STEP_DELTA), num_cloths).tolist()
        possible_basket_locs = np.random.choice(range(0, 144, BASKET_STEP_DELTA), num_cloths).tolist()

        success = True
        plan.params['table'].openrave_body.set_pose([10,10,10])
        plan.params['basket'].openrave_body.set_pose([-10,10,10])
        for c in range(num_cloths-1, -1, -1):
            next_loc = possible_locs.pop()
            next_x = (next_loc / 35) / 100.0 + CLOTH_INIT_X_RANGE[0]
            next_y = (next_loc % 35) / 100.0 + CLOTH_INIT_Y_RANGE[0]
            X[0, plan.state_inds[('cloth_{0}'.format(c), 'pose')]] = [next_x, next_y, TABLE_TOP]
            X[:, plan.state_inds[('cloth_target_begin_{0}'.format(c), 'value')]] = [next_x, next_y, TABLE_TOP]

            arm_poses = plan.params['baxter'].openrave_body.get_ik_from_pose([next_x, next_y, TABLE_TOP + 0.05], [0, np.pi/2, 0], "left_arm")
            if not len(arm_poses): success = False

            next_x = (basket.pose[0,0] - 0.12) + (possible_basket_locs[c] / 24) / 100.0
            next_y = (basket.pose[1,0] - 0.12) + (possible_basket_locs[c] % 24) / 100.0
            X[:, plan.state_inds[('cloth_target_end_{0}'.format(c), 'value')]] = [next_x, next_y, TABLE_TOP]

            arm_poses = plan.params['baxter'].openrave_body.get_ik_from_pose([next_x, next_y, TABLE_TOP + 0.05], [0, np.pi/2, 0], "left_arm")
            if not len(arm_poses): success = False

            X[:, plan.state_inds[('cloth_putdown_end_{0}'.format(c), 'lArmPose')]] = L_ARM_PUTDOWN_END
            X[:, plan.state_inds[('cloth_putdown_end_{0}'.format(c), 'lGripper')]] = 0.02
            X[:, plan.state_inds[('cloth_putdown_end_{0}'.format(c), 'rArmPose')]] = R_ARM_PUTDOWN_END
            X[:, plan.state_inds[('cloth_putdown_end_{0}'.format(c), 'rGripper')]] = 0.02

        stationary_params = ['basket']
        X_0s.append((X[0], [0, 3], stationary_params, plan_num))

        for i in range(1, num_cloths):
            num_on_table = num_cloths - i

            X[i, plan.state_inds[('baxter', 'lArmPose')]] = L_ARM_PUTDOWN_END
            X[i, plan.state_inds[('baxter', 'lGripper')]] = 0.02
            X[i, plan.state_inds[('baxter', 'rArmPose')]] = R_ARM_PUTDOWN_END
            X[i, plan.state_inds[('baxter', 'rGripper')]] = 0.02

            for c in range(num_cloths-1, num_cloths-num_on_table-1, -1):
                X[i, plan.state_inds[('cloth_{0}'.format(c), 'pose')]] = X[i, plan.state_inds[('cloth_target_begin_{0}'.format(c), 'value')]]

            stationary_params = ['basket']
            for c in range(num_cloths-num_on_table):
                X[i, plan.state_inds[('cloth_{0}'.format(c), 'pose')]] = X[i, plan.state_inds[('cloth_target_end_{0}'.format(c), 'value')]]
                stationary_params.append('cloth_{0}'.format(c))

            discard_actions = (num_cloths - num_on_table) * 4
            X_0s.append((X[i], [i*4, i*4+3], stationary_params, plan_num))

    print "Found initial configuration.\n"
    return X_0s


def get_randomized_initial_state_left(plan):
    num_cloths = 0
    while 'cloth_{0}'.format(num_cloths) in plan.params:
        num_cloths += 1

    X = np.zeros((plan.dX))
    state_config = []

    success = False
    while not success:
        print 'Searching for initial configuration...'
        X_0s = []

        X[plan.state_inds[('robot_end_pose', 'lArmPose')]] = plan.params['robot_end_pose'].lArmPose.flatten()
        X[plan.state_inds[('robot_end_pose', 'lGripper')]] = plan.params['robot_end_pose'].lGripper
        X[plan.state_inds[('robot_end_pose', 'rArmPose')]] = np.zeros((7,))
        X[plan.state_inds[('robot_end_pose', 'rGripper')]] = 0

        X[plan.state_inds[('robot_init_pose', 'lArmPose')]] = plan.params['robot_init_pose'].lArmPose.flatten()
        X[plan.state_inds[('robot_init_pose', 'lGripper')]] = plan.params['robot_init_pose'].lGripper
        X[plan.state_inds[('robot_init_pose', 'rArmPose')]] = np.zeros((7,))
        X[plan.state_inds[('robot_init_pose', 'rGripper')]] = 0

        X[plan.state_inds[('baxter', 'lArmPose')]] = plan.params['robot_init_pose'].lArmPose.flatten()
        X[plan.state_inds[('baxter', 'lGripper')]] = plan.params['robot_init_pose'].lGripper

        basket = plan.params['basket']
        basket.pose[:,:] = [[np.random.uniform(BASKET_X_RANGE[0], BASKET_X_RANGE[1])],
                            [np.random.uniform(BASKET_Y_RANGE[0], BASKET_Y_RANGE[1])],
                            [BASKET_POSE[2]]]
        X[plan.state_inds[('basket', 'pose')]] = basket.pose[:,0]
        X[plan.state_inds[('init_target', 'value')]] = basket.pose[:,0]
        X[plan.state_inds[('init_target', 'rotation')]] = [0, 0, np.pi/2]

        possible_locs = np.random.choice(range(0, 35*35, STEP_DELTA), num_cloths).tolist()
        possible_basket_locs = np.random.choice(range(0, 144, BASKET_STEP_DELTA), num_cloths).tolist()

        success = True
        plan.params['table'].openrave_body.set_pose([10,10,10])
        plan.params['basket'].openrave_body.set_pose([-10,10,10])
        for c in range(num_cloths):
            plan.params['cloth_{0}'.format(c)].openrave_body.set_pose([10,-10,-10])
        for c in range(num_cloths-1, -1, -1):
            next_loc = possible_locs.pop()
            next_x = (next_loc / 35) / 100.0 + CLOTH_INIT_X_RANGE[0]
            next_y = (next_loc % 35) / 100.0 + CLOTH_INIT_Y_RANGE[0]
            X[plan.state_inds[('cloth_target_begin_{0}'.format(c), 'value')]] = [next_x, next_y, TABLE_TOP]

            arm_poses = plan.params['baxter'].openrave_body.get_ik_from_pose([next_x, next_y, TABLE_TOP + 0.075], [0, np.pi/2, 0], "left_arm")
            if not len(arm_poses): success = False

            next_x = (basket.pose[0,0] - 0.12) + (possible_basket_locs[c] / 24) / 100.0
            next_y = (basket.pose[1,0] - 0.12) + (possible_basket_locs[c] % 24) / 100.0
            X[plan.state_inds[('cloth_target_end_{0}'.format(c), 'value')]] = [next_x, next_y, TABLE_TOP+BASKET_HEIGHT_DELTA]

            arm_poses = plan.params['baxter'].openrave_body.get_ik_from_pose([next_x, next_y, TABLE_TOP + 0.075], [0, np.pi/2, 0], "left_arm")
            if not len(arm_poses): success = False

            X[plan.state_inds[('cloth_putdown_end_{0}'.format(c), 'lArmPose')]] = L_ARM_PUTDOWN_END
            X[plan.state_inds[('cloth_putdown_end_{0}'.format(c), 'lGripper')]] = 0.02
            X[plan.state_inds[('cloth_putdown_end_{0}'.format(c), 'rArmPose')]] = np.zeros((7,))
            X[plan.state_inds[('cloth_putdown_end_{0}'.format(c), 'rGripper')]] = 0.0

        num_on_table = np.random.randint(1, num_cloths + 1)

        for c in range(num_cloths-1, num_cloths-num_on_table-1, -1):
            X[plan.state_inds[('cloth_{0}'.format(c), 'pose')]] = X[plan.state_inds[('cloth_target_begin_{0}'.format(c), 'value')]]

        stationary_params = ['basket']
        for c in range(num_cloths-num_on_table):
            X[plan.state_inds[('cloth_target_begin_{0}'.format(c), 'value')]] = X[plan.state_inds[('cloth_target_end_{0}'.format(c), 'value')]]
            X[plan.state_inds[('cloth_{0}'.format(c), 'pose')]] = X[plan.state_inds[('cloth_target_end_{0}'.format(c), 'value')]]
            stationary_params.append('cloth_{0}'.format(c))

        plan_idx = num_cloths - num_on_table
        state_config = [X, [plan_idx*4, plan_idx*4+3], stationary_params]

    print "Found initial configuration.\n"
    return state_config