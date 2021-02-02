from core.util_classes.viewer import OpenRAVEViewer
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes import robot_predicates
from core.util_classes.items import Item
from core.util_classes.robots import Robot
import core.util_classes.transform_utils as T

from core.util_classes.openrave_body import USE_OPENRAVE
from functools import reduce
if USE_OPENRAVE:
    from openravepy import matrixFromAxisAngle, IkParameterization, IkParameterizationType, \
                           IkFilterOptions, Environment, Planner, RaveCreatePlanner, \
                           RaveCreateTrajectory, matrixFromAxisAngle, CollisionReport, \
                           RaveCreateCollisionChecker
import core.util_classes.baxter_constants as const
from collections import OrderedDict
from sco.expr import Expr
import math
import numpy as np
import pybullet as p
PI = np.pi
DEBUG = False

#These functions are helper functions that can be used by many robots
#@profile
def get_random_dir():
    """
        This helper function generates a random 2d unit vectors
    """
    rand_dir = np.random.rand(2) - 0.5
    rand_dir = rand_dir/np.linalg.norm(rand_dir)
    return rand_dir

#@profile
def get_random_theta():
    """
        This helper function generates a random angle between -PI to PI
    """
    theta =  2*PI*np.random.rand(1) - PI
    return theta[0]

#@profile
def smaller_ang(x):
    """
        This helper function takes in an angle in radius, and returns smaller angle
        Ex. 5pi/2 -> PI/2
            8pi/3 -> 2pi/3
    """
    return (x + PI)%(2*PI) - PI

#@profile
def closer_ang(x,a,dir=0):
    """
        find angle y (==x mod 2*PI) that is close to a
        dir == 0: minimize absolute value of difference
        dir == 1: y > x
        dir == 2: y < x
    """
    if dir == 0:
        return a + smaller_ang(x-a)
    elif dir == 1:
        return a + (x-a)%(2*PI)
    elif dir == -1:
        return a + (x-a)%(2*PI) - 2*PI

#@profile
def closest_arm_pose(arm_poses, cur_arm_pose):
    """
        Given a list of possible arm poses, select the one with the least displacement from current arm pose
    """
    min_change = np.inf
    chosen_arm_pose = arm_poses[0] if len(arm_poses) else None
    for arm_pose in arm_poses:
        change = sum((arm_pose - cur_arm_pose)**2)
        if change < min_change:
            chosen_arm_pose = arm_pose
            min_change = change
    return chosen_arm_pose

#@profile
def closest_base_poses(base_poses, robot_base):
    """
        Given a list of possible base poses, select the one with the least displacement from current base pose
    """
    val, chosen = np.inf, robot_base
    if len(base_poses) <= 0:
        return chosen
    for base_pose in base_poses:
        diff = base_pose - robot_base
        distance = reduce(lambda x, y: x**2 + y, diff, 0)
        if distance < val:
            chosen = base_pose
            val = distance
    return chosen

#@profile
def lin_interp_traj(start, end, time_steps):
    """
    This helper function returns a linear trajectory from start pose to end pose
    """
    assert start.shape == end.shape
    if time_steps == 0:
        assert np.allclose(start, end)
        return start.copy()
    rows = start.shape[0]
    traj = np.zeros((rows, time_steps+1))

    for i in range(rows):
        traj_row = np.linspace(start[i], end[i], num=time_steps+1)
        traj[i, :] = traj_row
    return traj

#@profile
def get_expr_mult(coeff, expr):
    """
        Multiply expresions with coefficients
    """
    new_f = lambda x: coeff*expr.eval(x)
    new_grad = lambda x: coeff*expr.grad(x)
    return Expr(new_f, new_grad)

# Sample base values to face the target
#@profile
def sample_base(target_pose, base_pose):
    vec = target_pose[:2] - np.zeros((2,))
    vec = vec / np.linalg.norm(vec)
    theta = math.atan2(vec[1], vec[0])
    return theta

#@profile
def process_traj(raw_traj, timesteps):
    """
        Process raw_trajectory so that it's length is desired timesteps
        when len(raw_traj) > timesteps
            sample Trajectory by space to reduce trajectory size
        when len(raw_traj) < timesteps
            append last timestep pose util the size fits

        Note: result_traj includes init_dof and end_dof
    """
    result_traj = []
    if len(raw_traj) == timesteps:
        result_traj = raw_traj.copy()
    else:
        traj_arr = [0]
        result_traj.append(raw_traj[0])
        #calculate accumulative distance
        for i in range(len(raw_traj)-1):
            traj_arr.append(traj_arr[-1] + np.linalg.norm(raw_traj[i+1] - raw_traj[i]))
        step_dist = traj_arr[-1]/(timesteps - 1)
        process_dist, i = 0, 1
        while i < len(traj_arr)-1:
            if traj_arr[i] == process_dist + step_dist:
                result_traj.append(raw_traj[i])
                process_dist += step_dist
            elif traj_arr[i] < process_dist+step_dist < traj_arr[i+1]:
                dist = process_dist+step_dist - traj_arr[i]
                displacement = (raw_traj[i+1] - raw_traj[i])/(traj_arr[i+1]-traj_arr[i])*dist
                result_traj.append(raw_traj[i]+displacement)
                process_dist += step_dist
            else:
                i += 1
    result_traj.append(raw_traj[-1])
    return np.array(result_traj).T


#@profile
def resample_pred(pred, negated, t, plan):
    res, attr_inds = [], OrderedDict()
    # Determine which action failed first
    rs_action, ref_index = None, None
    for i in range(len(plan.actions)):
        active = plan.actions[i].active_timesteps
        if active[0] <= t <= active[1]:
            rs_action, ref_index = plan.actions[i], i
            break

    if rs_action.name == 'moveto' or rs_action.name == 'movetoholding':
        return resample_move(plan, t, pred, rs_action, ref_index)
    elif rs_action.name == 'grasp' or rs_action.name == 'putdown':
        return resample_pick_place(plan, t, pred, rs_action, ref_index)
    else:
        raise NotImplemented

#@profile
def resample_move(plan, t, pred, rs_action, ref_index):
    res, attr_inds = [], OrderedDict()
    robot = rs_action.params[0]
    act_range = rs_action.active_timesteps
    body = robot.openrave_body.env_body
    manip_name = "right_arm"
    active_dof = body.GetManipulator(manip_name).GetArmIndices()
    active_dof = np.hstack([[0], active_dof])
    robot.openrave_body.set_dof({'rGripper': 0.02})

    # In pick place domain, action flow is natually:
    # moveto -> grasp -> movetoholding -> putdown
    sampling_trace = None
    #rs_param is pdp_target0
    rs_param = rs_action.params[2]
    if ref_index + 1 < len(plan.actions):
        # take next action's ee_pose and find it's ik value.
        # ref_param is ee_target0
        ref_action = plan.actions[ref_index + 1]
        ref_range = ref_action.active_timesteps
        ref_param = ref_action.params[4]
        event_timestep = (ref_range[1] - ref_range[0])/2
        pose = robot.pose[:, event_timestep]

        arm_pose = get_ik_from_pose(ref_param.value, ref_param.rotation, body, manip_name)
        # In the case ee_pose wasn't even feasible, resample other preds
        if arm_pose is None:
            return None, None

        sampling_trace = {'data': {rs_param.name: {'type': rs_param.get_type(), 'rArmPose': arm_pose, 'value': pose}}, 'timestep': t, 'pred': pred, 'action': rs_action.name}
        add_to_attr_inds_and_res(t, attr_inds, res, rs_param, [('rArmPose', arm_pose), ('value', pose)])

    else:
        arm_pose = rs_action.params[2].rArmPose[:,0].flatten()
        pose = rs_action.params[2].value[:,0]

    """Resample Trajectory by BiRRT"""
    init_arm_dof = rs_action.params[1].rArmPose[:,0].flatten()
    init_pose_dof = rs_action.params[1].value[:,0]
    init_dof = np.hstack([init_pose_dof, init_arm_dof])
    end_dof = np.hstack([pose, arm_pose])

    raw_traj = get_rrt_traj(plan.env, body, active_dof, init_dof, end_dof)
    if raw_traj == None and sampling_trace != None:
        # In the case resampled poses resulted infeasible rrt trajectory
        sampling_trace['reward'] = -1
        plan.sampling_trace.append(sampling_trace)
        return np.array(res), attr_inds
    elif raw_traj == None and sampling_trace == None:
        # In the case resample is just not possible, resample other preds
        return None, None
    # Restore dof0
    body.SetActiveDOFValues(np.hstack([[0], body.GetActiveDOFValues()[1:]]))
    # initailize feasible trajectory
    result_traj = process_traj(raw_traj, act_range[1] - act_range[0] + 2).T[1:-1]
    ts = 1
    for traj in result_traj:
        add_to_attr_inds_and_res(act_range[0] + ts, attr_inds, res, robot, [('rArmPose', traj[1:]), ('pose', traj[:1])])
        ts += 1
    return np.array(res), attr_inds

#@profile
def resample_pick_place(plan, t, pred, rs_action, ref_index):
    res, attr_inds = [], OrderedDict()
    robot = rs_action.params[0]
    act_range = rs_action.active_timesteps
    body = robot.openrave_body.env_body
    manip_name = "right_arm"
    active_dof = body.GetManipulator(manip_name).GetArmIndices()
    active_dof = np.hstack([[0], active_dof])
    # In pick place domain, action flow is natually:
    # moveto -> grasp -> movetoholding -> putdown
    #rs_param is ee_poses, ref_param is target
    rs_param = rs_action.params[4]
    ref_param = rs_action.params[2]
    ee_poses = get_ee_from_target(ref_param.value, ref_param.rotation)
    for samp_ee in ee_poses:
        arm_pose = get_ik_from_pose(samp_ee[0].flatten(), samp_ee[1].flatten(), body, manip_name)
        if arm_pose is not None:
            break

    if arm_pose is None:
        return None, None
    sampling_trace = {'data': {rs_param.name: {'type': rs_param.get_type(), 'value': samp_ee[0], 'rotation': samp_ee[1]}}, 'timestep': t, 'pred': pred, 'action': rs_action.name}
    add_to_attr_inds_and_res(t, attr_inds, res, rs_param, [('value', samp_ee[0].flatten()), ('rotation', samp_ee[1].flatten())])

    """Resample Trajectory by BiRRT"""
    if t < (act_range[1] - act_range[0])/2 and ref_index >= 1:
        # if resample time occured before grasp or putdown.
        # resample initial poses as well
        # ref_action is move
        ref_action = plan.actions[ref_index - 1]
        ref_range = ref_action.active_timesteps

        start_pose = ref_action.params[1]
        init_dof = start_pose.rArmPose[:,0].flatten()
        init_dof = np.hstack([start_pose.value[:,0], init_dof])
        end_dof = np.hstack([robot.pose[:,t], arm_pose])
        timesteps = act_range[1] - ref_range[0] + 2

        init_timestep = ref_range[0]

    else:
        start_pose = rs_action.params[3]

        init_dof = start_pose.rArmPose[:,0].flatten()
        init_dof = np.hstack([start_pose.value[:,0], init_dof])
        end_dof = np.hstack([robot.pose[:,t], arm_pose])
        timesteps = act_range[1] - act_range[0] + 2

        init_timestep = act_range[0]

    raw_traj = get_rrt_traj(plan.env, body, active_dof, init_dof, end_dof)
    if raw_traj == None:
        # In the case resampled poses resulted infeasible rrt trajectory
        plan.sampling_trace.append(sampling_trace)
        plan.sampling_trace[-1]['reward'] = -1
        return np.array(res), attr_inds

    # Restore dof0
    body.SetActiveDOFValues(np.hstack([[0], body.GetActiveDOFValues()[1:]]))
    # initailize feasible trajectory
    result_traj = process_traj(raw_traj, timesteps).T[1:-1]

    ts = 1
    for traj in result_traj:
        add_to_attr_inds_and_res(init_timestep + ts, attr_inds, res, robot, [('rArmPose', traj[1:]), ('pose', traj[:1])])
        ts += 1

    if init_timestep != act_range[0]:
        sampling_trace['data'][start_pose.name] = {'type': start_pose.get_type(), 'rArmPose': robot.rArmPose[:, act_range[0]], 'value': robot.pose[:, act_range[0]]}
        add_to_attr_inds_and_res(init_timestep + ts, attr_inds, res, start_pose, [('rArmPose', sampling_trace['data'][start_pose.name]['rArmPose']), ('value', sampling_trace['data'][start_pose.name]['value'])])

    return np.array(res), attr_inds

#@profile
def resample_eereachable_ver(pred, negated, t, plan, inv=False, arms=[]):
    attr_inds, res = OrderedDict(), OrderedDict()

    # Preparing the variables
    robot, rave_body = pred.robot, pred.robot.openrave_body

    act_inds, action = [(i, act) for i, act in enumerate(plan.actions) if act.active_timesteps[0] <= t and  t <= act.active_timesteps[1]][0]
    # print "resampling at {} action".format(action.name)
    ee_pose = pred.params[2]
    if not len(arms): arms = pred.arms
    if not len(arms): arms = robot.geom.arms

    ee_pos, ee_rot = ee_pose.value[:, 0], ee_pose.rotation[:, 0]

    robot_trans, arm_inds = pred.get_robot_info(rave_body, arm)
    if action.name.find('basket_grasp') >= 0 or action.name.find('basket_putdown') >= 0:
        basket, offset = plan.params['basket'], np.array([0, const.BASKET_OFFSET, 0])
        if arm == 'left':
            basket_pos = ee_pos - offset
        else:
            basket_pos = ee_pos + offset
        facing_pose = basket_pos[:2].dot([0,1])/np.linalg.norm(basket_pos[:2])
    elif action.name.find('cloth_grasp') >= 0 or action.name.find('cloth_putdown') >= 0:
        cloth = action.params[1]
        facing_pose = ee_pos[:2].dot([0,1])/np.linalg.norm(ee_pos[:2])
    else:
        facing_pose = robot.pose[:, t]

    rave_body.set_from_param(robot, t)
    for param in list(plan.params.values()):
        if not param.is_symbol() and 'Robot' not in param.get_types():
            if param.openrave_body is None: continue
            param.openrave_body.set_pose(param.pose[:, t].flatten(), param.rotation[:, t].flatten())

    # Resample poses at grasping time
    grasp_arm_pose = get_ik_from_pose(ee_pos, ee_rot, body, arm)

    # When Ik infeasible
    if grasp_arm_pose is None:
        return None, None

    rave_body.set_dof({arm: grasp_arm_pose})
    add_to_attr_inds_and_res(t, attr_inds, res, robot, [(arm, grasp_arm_pose.copy()), ('pose', np.array([facing_pose]))])

    # Resample entire approaching and retreating traj
    step = const.EEREACHABLE_STEPS
    for i in range(step):
        app_pos = ee_pos + np.array([0,0,const.APPROACH_DIST]) * (step-i)
        approach_arm_pose = get_ik_from_pose(app_pos, ee_rot, body, arm)
        if approach_arm_pose is None:
            return None, None
        add_to_attr_inds_and_res(t-step+i, attr_inds, res, robot, [(arm, approach_arm_pose), ('pose', np.array([facing_pose]))])

        if DEBUG:
            rave_body.set_dof({arm_attr_name: grasp_arm_pose})

        ret_pos = ee_pos + np.array([0,0,const.APPROACH_DIST]) * (i+1)
        retreat_arm_pose = get_ik_from_pose(ret_pos, ee_rot, body, arm)
        if  retreat_arm_pose is None:
            return None, None
        add_to_attr_inds_and_res(t+1+i, attr_inds, res, robot, [(arm, retreat_arm_pose), ('pose', np.array([facing_pose]))])

        if DEBUG:
            rave_body.set_dof({arm: grasp_arm_pose})

    """
    Linear Interp Traj
    """
    act_start, act_end = action.active_timesteps
    timesteps = t-step - act_start
    robot_poses = getattr(robot, arm)

    pose_traj = lin_interp_traj(robot.pose[:, act_start], robot.pose[:,t-step], timesteps)
    arm_traj = lin_interp_traj(robot_poses[:, act_start], robot_poses[:, t-step], timesteps)
    for i in range(act_start+1, t-step):
        traj_ind = i - act_start
        add_to_attr_inds_and_res(i, attr_inds, res, robot, [(arm, arm_traj[:, traj_ind]), ('pose', pose_traj[:, traj_ind])])

    timesteps = act_end - (t+step)
    pose_traj = lin_interp_traj(robot.pose[:, t+step], robot.pose[:, act_end], timesteps)
    arm_traj = lin_interp_traj(robot_poses[:, t+step], robot_poses[:, act_end], timesteps)
    for i in range(t+step+1, act_end):
        traj_ind = i - (t+step)
        add_to_attr_inds_and_res(i, attr_inds, res, robot, [(arm, arm_traj[:, traj_ind]), ('pose', pose_traj[:, traj_ind])])

    """
        Resample other parameters
    """
    if action.name.find('basket_grasp') >= 0:
        add_to_attr_inds_and_res(t, attr_inds, res, basket, [('pose', basket.pose[:, t])])
        for i in range(step):
            add_to_attr_inds_and_res(t+1+i, attr_inds, res, basket, [('pose', basket.pose[:, t] + np.array([0,0,const.RETREAT_DIST]) * (i+1))])
    elif action.name.find('basket_putdown') >= 0:
        add_to_attr_inds_and_res(t, attr_inds, res, basket, [('pose', basket.pose[:, t])])
        for i in range(step):
            add_to_attr_inds_and_res(t-step+i, attr_inds, res, basket, [('pose', basket.pose[:, t] + np.array([0,0,const.RETREAT_DIST]) * (i+1))])
    elif action.name.find('cloth_grasp') >= 0:
        add_to_attr_inds_and_res(t, attr_inds, res, cloth, [('pose', cloth.pose[:, t])])
        for i in range(step):
            add_to_attr_inds_and_res(t+1+i, attr_inds, res, cloth, [('pose', cloth.pose[:, t] + np.array([0,0,const.RETREAT_DIST]) * (i+1))])
    elif action.name.find('cloth_putdown') >= 0:
        add_to_attr_inds_and_res(t, attr_inds, res, cloth, [('pose', cloth.pose[:, t])])
        for i in range(step):
            add_to_attr_inds_and_res(t-step+i, attr_inds, res, cloth, [('pose', cloth.pose[:, t] + np.array([0,0,const.RETREAT_DIST]) * (i+1))])

    if DEBUG: test_resample_order(attr_inds, res)
    if DEBUG: assert pred.test(t, negated = negated, tol = 1e-3)
    return res, attr_inds

def resample_gripper_down_rot(pred, negated, t, plan, arms=[]):
    attr_inds, res = OrderedDict(), OrderedDict()

    robot = pred.robot
    if not len(arms): arms = pred.arms
    geom = robot.geom
    axis = pred.axis
    pose = robot.openrave_body.param_fwd_kinematics(robot, arms, t)
    iks = {}
    for arm in arms:
        ee_name = geom.ee_link_names[arm]
        ee_link = geom.get_ee_link(arm)
        pos = pose[arm]['pos']
        quat = pred.quats[arm]
        iks[arm] = np.array(robot.openrave_body.get_ik_from_pose(pos, quat, arm))

    add_to_attr_inds_and_res(t, attr_inds, res, robot, [(arm, iks[arm]) for arm in arms])
    return res, attr_inds

def resample_in_gripper(pred, negated, t, plan, fix_obj=False):
    attr_inds, res = OrderedDict(), OrderedDict()

    robot, obj = pred.robot, pred.obj
    body, arm = robot.openrave_body, pred.arm
    geom = robot.geom

    act_inds, action = [(i, act) for i, act in enumerate(plan.actions) if act.active_timesteps[0] <= t and  t <= act.active_timesteps[1]][0]

    if not fix_obj or action.name.find("moveholding") >= 0:
        ts_range = action.active_timesteps
        for ts in range(ts_range[0], ts_range[1]+1):
            body.set_from_param(robot, ts)
            pos = body.fwd_kinematics(arm)['pos']
            add_to_attr_inds_and_res(ts, attr_inds, res, cloth, [('pose', pos)])
    elif fix_obj or action.name.find("grasp") >= 0 or action.name.find("putdown") >= 0:
        grasp_time = action.active_timesteps[0]+1 + const.EEREACHABLE_STEPS
        st = max(action.active_timesteps[0], grasp_time-ts_delta)
        et = min(action.active_timesteps[1], grasp-time+ts_delta)
        ts_range = range(st, et)
        for ts in ts_range:
            pos = obj.pose[:, ts]
            body.set_from_param(robot, ts)
            quat = body.fwd_kinematics(arm)['quat']
            ik = body.get_ik_from_pose(pos, quat, arm)
            add_to_attr_inds_and_res(ts, attr_inds, res, robot, [(arm, ik)])

    if DEBUG: assert pred.test(t, negated = negated, tol = 1e-3)
    return res, attr_inds

#@profile
def resample_basket_moveholding_all(pred, negated, t, plan):
    attr_inds, res = OrderedDict(), OrderedDict()

    robot, basket = pred.robot, pred.obj
    rave_body, body = robot.openrave_body, robot.openrave_body.env_body
    offset = np.array([0, const.BASKET_OFFSET, 0])
    left_robot_trans, left_arm_inds = pred.get_robot_info(rave_body, 'left')
    right_robot_trans, right_arm_inds = pred.get_robot_info(rave_body, 'right')

    act_inds, action = [(i, act) for i, act in enumerate(plan.actions) if act.active_timesteps[0] <= t and  t <= act.active_timesteps[1]][0]

    act_name = action.name
    ind1, ind2 = act_name.find(': ')+2, act_name.find(' (')
    act_name = act_name[ind1:ind2]
    # print act_name

    if act_name == "moveholding":
        start_ts, end_ts = action.active_timesteps
        basket_begin = basket.pose[:, start_ts]
        basket_end = basket.pose[:, end_ts]
        timesteps = end_ts - start_ts
        pose_traj = lin_interp_traj(basket_begin, basket_end, timesteps)
        ee_rot = np.array([0, np.pi/2,0])
        for i in range(start_ts+1, end_ts):
            traj_ind = i - start_ts
            basket_pos = pose_traj[:, traj_ind]
            add_to_attr_inds_and_res(i, attr_inds, res, basket, [('pose', basket_pos)])
            basket.openrave_body.set_pose(basket_pos, [np.pi/2, 0, np.pi/2])

            facing_pose = basket_pos[:2].dot([0,1])/np.linalg.norm(basket_pos[:2])
            # TODO for demo2 base can't move
            facing_pose = 0
            rave_body.set_pose([0,0,facing_pose])

            ee_left = basket_pos + offset
            left_arm_pose = get_ik_from_pose(ee_left, ee_rot, body,  "left_arm")

            ee_right = basket_pos - offset
            right_arm_pose = get_ik_from_pose(ee_right, ee_rot, body,  "right_arm")
            add_to_attr_inds_and_res(i, attr_inds, res, robot, [('lArmPose', left_arm_pose), ('rArmPose', right_arm_pose), ('pose', np.array([facing_pose]))])
            rave_body.set_dof({'lArmPose': left_arm_pose, 'rArmPose': right_arm_pose})
        return res, attr_inds
    else:
        return None, None


#@profile
def resample_basket_moveholding(pred, negated, t, plan):
    attr_inds, res = OrderedDict(), OrderedDict()

    robot, basket = pred.robot, pred.obj
    rave_body, body = robot.openrave_body, robot.openrave_body.env_body
    offset = np.array([0, const.BASKET_OFFSET, 0])

    act_inds, action = [(i, act) for i, act in enumerate(plan.actions) if act.active_timesteps[0] <= t and  t <= act.active_timesteps[1]][0]

    act_name = action.name
    ind1, ind2 = act_name.find(': ')+2, act_name.find(' (')
    act_name = act_name[ind1:ind2]

#@profile
def set_pose(x, trag):
    basket.openrave_body.set_pose(trag[:, x], [np.pi/2, 0, np.pi/2])

    if action.name.find("moveholding_basket") >= 0:
        start_ts, end_ts = action.active_timesteps
        basket_begin = basket.pose[:, start_ts]
        basket_end = basket.pose[:, end_ts]
        timesteps = end_ts - start_ts
        pose_traj = lin_interp_traj(basket_begin, basket_end, timesteps)
        ee_rot = np.array([0, np.pi/2,0])

        traj_ind = t - start_ts
        basket_pos = pose_traj[:, traj_ind]
        add_to_attr_inds_and_res(t, attr_inds, res, basket, [('pose', basket_pos), ('rotation', np.array([np.pi/2, 0, np.pi/2]))])

        basket.openrave_body.set_pose(basket_pos, [np.pi/2, 0, np.pi/2])

        facing_pose = basket_pos[:2].dot([0,1])/np.linalg.norm(basket_pos[:2])
        # TODO for demo2 base can't move
        facing_pose = 0
        rave_body.set_pose([0,0,facing_pose])

        ee_left = basket_pos + offset
        left_arm_pose = get_ik_from_pose(ee_left, ee_rot, body,  "left_arm")

        ee_right = basket_pos - offset
        right_arm_pose = get_ik_from_pose(ee_right, ee_rot, body,  "right_arm")
        add_to_attr_inds_and_res(t, attr_inds, res, robot, [('lArmPose', left_arm_pose), ('rArmPose', right_arm_pose), ('pose', np.array([facing_pose]))])
        rave_body.set_dof({'lArmPose': left_arm_pose, 'rArmPose': right_arm_pose})
        return res, attr_inds
    else:
        return None, None


#@profile
def resample_basket_in_gripper(pred, negated, t, plan):
    attr_inds, res = OrderedDict(), OrderedDict()

    robot, basket = pred.robot, pred.obj
    rave_body, body = robot.openrave_body, robot.openrave_body.env_body
    offset = np.array([0, const.BASKET_OFFSET, 0])
    left_robot_trans, left_arm_inds = pred.get_robot_info(rave_body, 'left')
    right_robot_trans, right_arm_inds = pred.get_robot_info(rave_body, 'right')
    # Take the left arm as the anchor point
    basket_pos = left_robot_trans[:3,3] - offset
    #TODO Assuming basket not rotating
    basket_rot = np.array([np.pi/2, 0, np.pi/2])

    add_to_attr_inds_and_res(t, attr_inds, res, basket, [('pose', basket_pos), ('rotation', basket_rot)])


    if not np.allclose(right_robot_trans[:3,3], basket_rot - offset):
        targ_pos, targ_rot = basket_rot - offset, [0, np.pi/2,0]
        grasp_right_arm_pose = get_ik_from_pose(targ_pos, targ_rot, body,  "right_arm")
        if grasp_right_arm_pose is None:
            return res, attr_inds
        add_to_attr_inds_and_res(t, attr_inds, res, robot, [('rArmPose', grasp_right_arm_pose.copy())])

    return res, attr_inds

#@profile
def resample_cloth_in_gripper(pred, negated, t, plan):
    attr_inds, res = OrderedDict(), OrderedDict()

    robot, cloth = pred.robot, pred.obj
    rave_body, arm = robot.openrave_body, pred.arm
    manip = rave_body.env_body.GetManipulator("{}_arm".format(pred.arm))

    act_inds, action = [(i, act) for i, act in enumerate(plan.actions) if act.active_timesteps[0] <= t and  t <= act.active_timesteps[1]][0]

    if action.name.find("moveholding_cloth") >= 0 or action.name.find("moveholding_basket") >= 0:
        ts_range = action.active_timesteps
        for ts in range(ts_range[0], ts_range[1]+1):
            dof_value = np.r_[robot.lArmPose[:, ts],
                              robot.lGripper[:, ts],
                              robot.rArmPose[:, ts],
                              robot.rGripper[:, ts],
                              robot.pose[:, ts]]
            pred.set_robot_poses(dof_value, rave_body)
            pose = manip.GetTransform()[:3,3]
            add_to_attr_inds_and_res(ts, attr_inds, res, cloth, [('pose', pose)])
        for ts in range(ts_range[0], ts_range[1]):
            assert pred.test(ts, negated = negated, tol = 1e-3)
    elif action.name.find("grasp") >= 0:
        grasp_time = action.active_timesteps[0] + const.EEREACHABLE_STEPS

    elif action.name.find("putdown") >= 0:
        putdown_time = action.active_timesteps[0] + const.EEREACHABLE_STEPS

    if DEBUG: assert pred.test(t, negated = negated, tol = 1e-3)
    return res, attr_inds

#@profile
def resample_washer_in_gripper(pred, negated, t, plan):
    return None, None
    attr_inds, res = OrderedDict(), OrderedDict()
    act_inds, action = [(i, act) for i, act in enumerate(plan.actions) if act.active_timesteps[0] <= t and  t <= act.active_timesteps[1]][0]
    robot, washer = pred.robot, pred.obj
    rave_body, washer_body = robot.openrave_body, washer.openrave_body
    arm = pred.arm
    body = rave_body.env_body
    manip = body.GetManipulator("{}_arm".format(pred.arm))
    # Make sure baxter is well positioned in the env
    dof_value = np.r_[robot.lArmPose[:, t],
                      robot.lGripper[:, t],
                      robot.rArmPose[:, t],
                      robot.rGripper[:, t],
                      robot.pose[:,t]]
    pred.set_robot_poses(dof_value, rave_body)
    dof_value = np.r_[washer.pose[:, t],
                      washer.rotation[:, t],
                      washer.door[:, t]]
    pred.set_washer_poses(dof_value, washer_body)

    tool_link,offset = washer_body.env_body.GetLink("washer_handle"), np.array([0, 0.06, 0])
    last_arm_pose = robot.lArmPose[:, action.active_timesteps[0] + 10]
    rave_body.set_dof({'lArmPose': [0,-np.pi/2,0,0,0,0,0], 'rArmPose': [0,-np.pi/2,0,0,0,0,0]})
    rave_body.set_dof({'{}ArmPose'.format(pred.arm[0]): last_arm_pose})
    is_mp_arm_poses = []

    door_range = np.linspace(0, -np.pi/2, 21)
    open_door_range = (action.active_timesteps[0]+10, action.active_timesteps[1]-10)
    for door in door_range:
        washer_body.set_dof({'door': door})
        washer_trans = tool_link.GetTransform()
        targ_pos, targ_rot = washer_trans.dot(np.r_[offset, 1])[:3],  [-np.pi/2, 0, -np.pi/2]
        ik_arm_poses = rave_body.get_ik_from_pose(targ_pos, targ_rot,  "{}_arm".format(arm))
        arm_pose = get_is_mp_arm_pose(rave_body, ik_arm_poses, last_arm_pose, arm)
        if arm_pose is None:
            arm_pose = closest_arm_pose(ik_arm_poses, last_arm_pose)
            if arm_pose is None:
                return None,None
        rave_body.set_dof({'{}ArmPose'.format(arm[0]): arm_pose})
        is_mp_arm_poses.append(arm_pose)
        last_arm_pose = arm_pose

    resample_attr_name = '{}ArmPose'.format(pred.arm[0])

    arm_poses = lin_interp_traj(is_mp_arm_poses[-1], getattr(robot, "{}ArmPose".format(pred.arm[0]))[:, open_door_range[1]], 5)
    for ts in range(open_door_range[0]+1, open_door_range[1]+1):
        index = ts - open_door_range[0]
        if index < 20:
            add_to_attr_inds_and_res(ts, attr_inds, res, robot, [(resample_attr_name, is_mp_arm_poses[index])])
            add_to_attr_inds_and_res(ts, attr_inds, res, washer, [('door', np.array([door_range[index]]))])
        else:
            add_to_attr_inds_and_res(ts, attr_inds, res, robot, [(resample_attr_name, is_mp_arm_poses[20])])
            add_to_attr_inds_and_res(ts, attr_inds, res, washer, [('door', np.array([door_range[20]]))])


    # step = const.EEREACHABLE_STEPS
    # for i in range(step):
    #         targ_app_pos = targ_pos + np.array([0,+const.RETREAT_DIST,0]) * (i+1)
    #         ik_arm_pose = rave_body.get_ik_from_pose(targ_app_pos, targ_rot, '{}_arm'.format(arm))
    #         if ik_arm_pose is None:
    #             return None, None
    #         approach_arm_pose = closest_arm_pose(ik_arm_pose, last_arm_pose)
    #         if approach_arm_pose is None:
    #             return None, None
    #         rave_body.set_dof({'lArmPose': approach_arm_pose})
    #         add_to_attr_inds_and_res(open_door_range[1]+1+i, attr_inds, res, robot, [(resample_attr_name, approach_arm_pose), ('pose', robot.pose[:,t])])

    return res, attr_inds

def resample_washer_in_gripper2(pred, negated, t, plan):
    attr_inds, res = OrderedDict(), OrderedDict()

    robot, washer = pred.robot, pred.obj
    rave_body, arm = robot.openrave_body, pred.arm
    body = rave_body.env_body
    manip = rave_body.env_body.GetManipulator("{}_arm".format(pred.arm))

    act_inds, action = [(i, act) for i, act in enumerate(plan.actions) if act.active_timesteps[0] <= t and  t <= act.active_timesteps[1]][0]
    ts_range = action.active_timesteps
    if not const.PRODUCTION:
        print("resample at {}".format(action.name))
    if action.name.find("open_door") >= 0:
        resample_start, resample_end = ts_range[0], ts_range[1]
        door_pose = lin_interp_traj(washer.door[:, ts_range[0]], washer.door[:, ts_range[1]], ts_range[1]-ts_range[0])
        for i in range(ts_range[0], ts_range[1]):
            ind = i - ts_range[0]
            add_to_attr_inds_and_res(i, attr_inds, res, washer, [('door', door_pose[:, ind])])

    elif action.name.find("handle_grasp") >= 0:
        resample_start, resample_end = 5+const.EEREACHABLE_STEPS, ts_range[1]
    elif action.name.find("handle_release") >= 0:
        resample_start, resample_end = ts_range[0], ts_range[1] - 5 -const.EEREACHABLE_STEPS
    else:
        raise NotImplementedError
    for ts in range(resample_start+1, resample_end):
        washer.openrave_body.set_dof({'door': washer.door[0, ts]})
        washer_trans, washer_inds = pred.get_washer_info(washer.openrave_body)

        rel_pt =  [-0.04, 0.07, -0.115]
        targ_pos, targ_rot = washer_trans.dot(np.r_[rel_pt, 1])[:3],  OpenRAVEBody.obj_pose_from_transform(washer_trans)[3:]

        grasp_arm_pose = closest_arm_pose(rave_body.get_ik_from_pose(targ_pos, targ_rot,  "{}_arm".format(pred.arm)), robot.lArmPose[:,ts-1])
        if grasp_arm_pose is None:
            return res, attr_inds
        rave_body.set_dof({'{}ArmPose'.format(pred.arm[0]): grasp_arm_pose})
        add_to_attr_inds_and_res(ts, attr_inds, res, robot, [('{}ArmPose'.format(pred.arm[0]), grasp_arm_pose)])
        assert pred.test(ts, negated = negated, tol = 1e-2)

    return res, attr_inds


def resample_gripper_at(pred, negated, t, plan, arm=None):
    attr_inds, res = OrderedDict(), OrderedDict()

    robot, pose = pred.robot, pred.pose
    if arm is None: arm = pred.arm
    rave_body = robot.openrave_body

    rave_body.set_from_param(robot, t)
    arm_poses = robot.openrave_body.get_ik_from_pose(pose.value[:,0], pose.rotation[:,0], arm)
    if not len(arm_poses):
        return attr_inds, res

    arm_pose = closest_arm_pose(arm_poses, getattr(robot, arm)[:,t])
    add_to_attr_inds_and_res(t, attr_inds, res, robot, [(arm, arm_pose)])

    if DEBUG: assert pred.test(t, negated = negated, tol = 1e-3)
    return res, attr_inds


#@profile
def get_is_mp_arm_pose(robot_body, arm_poses, last_pose, arm):
    robot = robot_body.env_body
    dof_map = robot_body._geom.dof_map
    dof_inds = dof_map["{}ArmPose".format(arm[0])]
    lb_limit, ub_limit = robot.GetDOFLimits()
    active_ub = ub_limit[dof_inds].reshape((len(dof_inds),1))
    active_lb = lb_limit[dof_inds].reshape((len(dof_inds),1))
    joint_move = np.round((active_ub-active_lb)/const.JOINT_MOVE_FACTOR, 3).flatten()
    is_mp_poses = []
    for pose in arm_poses:
        dof_difference = np.abs(np.round(pose - last_pose, 3))
        if np.all(dof_difference < joint_move):
            is_mp_poses.append(pose)
    # print "Total {} poses satisfied".format(len(is_mp_poses))
    if not is_mp_poses:
        return None
    return closest_arm_pose(is_mp_poses, last_pose)


#@profile
def resample_washer_ee_approach(pred, negated, t, plan, approach = True, rel_pt=None):
    attr_inds, res = OrderedDict(), OrderedDict()
    # Preparing the variables
    if not rel_pt:
        rel_pt = np.array([0,-const.APPROACH_DIST,0]) if approach else np.array([0,-const.RETREAT_DIST,0])
    robot, rave_body = pred.robot, pred.robot.openrave_body
    body = rave_body.env_body
    actions = plan.actions
    action_inds, action = [(i, act) for i, act in enumerate(plan.actions) if act.active_timesteps[0] <= t and  t <= act.active_timesteps[1]][0]
    ee_pose, arm = pred.ee_pose, pred.arm
    robot_base_pose = robot.pose[:, t]

    targ_pos, targ_rot = ee_pose.value[:, 0], ee_pose.rotation[:, 0]

    robot_trans, arm_inds = pred.get_robot_info(rave_body, arm)

    # Make sure baxter is well positioned in the env
    dof_value = np.r_[robot.lArmPose[:, t],
                      robot.lGripper[:, t],
                      robot.rArmPose[:, t],
                      robot.rGripper[:, t],
                      robot.pose[:,t]]
    pred.set_robot_poses(dof_value, rave_body)

    for param in list(plan.params.values()):
        if not param.is_symbol() and param != robot:
            if param.openrave_body is None: continue
            if isinstance(param, Robot):
                attrs = list(param.geom.dof_map.keys())
                dof_val_map = {}
                for attr in attrs:
                    dof_val_map[attr] = getattr(param, attr)[:, t]
                param.openrave_body.set_dof(dof_val_map)

            param.openrave_body.set_pose(param.pose[:, t].flatten(), param.rotation[:, t].flatten())

    # Resample poses at grasping time
    approach_arm_pose = get_ik_from_pose(targ_pos, targ_rot, body, "{}_arm".format(arm))

    # When Ik infeasible
    if approach_arm_pose is None:
        return None, None

    resample_attr_name = '{}ArmPose'.format(arm[0])
    rave_body.set_dof({resample_attr_name: approach_arm_pose})

    add_to_attr_inds_and_res(t, attr_inds, res, robot, [(resample_attr_name, approach_arm_pose.copy()), ('pose', robot_base_pose)])

    # Resample entire approaching and retreating traj
    resample_failure = False
    step = const.EEREACHABLE_STEPS

    for i in range(step):
        if approach:
            targ_app_pos = targ_pos + rel_pt * (step-i)
            approach_arm_pose = get_ik_from_pose(targ_app_pos, targ_rot, body, '{}_arm'.format(arm))
            if approach_arm_pose is None:
                return None, None
            add_to_attr_inds_and_res(t-step+i, attr_inds, res, robot, [(resample_attr_name, approach_arm_pose), ('pose', robot_base_pose)])
        else:
            targ_app_pos = targ_pos + rel_pt * (i+1)
            approach_arm_pose = get_ik_from_pose(targ_app_pos, targ_rot, body, '{}_arm'.format(arm))
            if approach_arm_pose is None:
                return None, None
            add_to_attr_inds_and_res(t+1+i, attr_inds, res, robot, [(resample_attr_name, approach_arm_pose), ('pose', robot_base_pose)])

        if DEBUG:
            rave_body.set_dof({resample_attr_name: approach_arm_pose})
            rave_body.set_pose([0,0, robot_base_pose])

        if approach_arm_pose is None:
            resample_failure = True

    # Ik infeasible
    if resample_failure:
        plan.sampling_trace[-1]['reward'] = -1
        return None, None

    """
    Linear Interp Traj
    """
    if action_inds > 0:
        last_action = actions[action_inds-1]
        act_start, act_end = last_action.active_timesteps
        if action.name.find("moveto") >=0  or action.name.find("moveholding_basket") >= 0 or action.name.find("moveholding_cloth") >= 0:
            timesteps = act_end - act_start

            pose_traj = lin_interp_traj(robot.pose[:, act_start], robot.pose[:,t-step], timesteps)
            left_arm_traj = lin_interp_traj(robot.lArmPose[:, act_start], robot.lArmPose[:, t-step], timesteps)
            right_arm_traj = lin_interp_traj(robot.rArmPose[:, act_start], robot.rArmPose[:, t-step], timesteps)
            for i in range(act_start+1, act_end):
                traj_ind = i - act_start
                add_to_attr_inds_and_res(i, attr_inds, res, robot, [('lArmPose', left_arm_traj[:, traj_ind]), ('rArmPose', right_arm_traj[:, traj_ind]), ('pose', pose_traj[:, traj_ind])])
                rave_body.set_dof({'lArmPose': left_arm_traj[:, traj_ind], 'rArmPose': right_arm_traj[:, traj_ind]})

    # """
    #     Resample other parameters
    # """
    # begin = pred.start_pose
    # add_to_attr_inds_and_res(0, attr_inds, res, begin, [('lArmPose', robot.lArmPose[:, t-step]), ('rArmPose', robot.rArmPose[:, t-step]), ('value', robot.pose[:,t-step])])
    return res, attr_inds

#@profile
def resample_ee_grasp_valid(pred, negated, t, plan):
    # TODO EEGraspValid is not working properly, go back and fix it
    attr_inds, res = OrderedDict(), OrderedDict()
    washer = pred.robot
    ee_pose = pred.ee_pose
    washer_body = washer.openrave_body
    tool_link = washer_body.env_body.GetLink("washer_handle")
    offset, door = np.array([0, 0.06,  0]), washer.door[0, t]
    washer_body.set_dof({'door': door})
    handle_pos = tool_link.GetTransform().dot(np.r_[offset, 1])[:3]

    handle_rot = np.array([-np.pi/2, 0, -np.pi/2])

    add_to_attr_inds_and_res(t, attr_inds, res, ee_pose, [('value', handle_pos), ('rotation', handle_rot)])

    return res, attr_inds

#@profile
def resample_obstructs(pred, negated, t, plan):
    # viewer = OpenRAVEViewer.create_viewer(plan.env)
    attr_inds, res = OrderedDict(), OrderedDict()
    act_inds, action = [(i, act) for i, act in enumerate(plan.actions) if act.active_timesteps[0] <= t and  t <= act.active_timesteps[1]][0]

    robot, obstacle = pred.robot, pred.obstacle
    rave_body, obs_body = robot.openrave_body, obstacle.openrave_body
    r_geom, obj_geom = rave_body._geom, obs_body._geom
    dof_map = {arm: getattr(robot, arm)[:,t] for arm in r_geom.arms}
    for gripper in r_geom.grippers: dof_map[gripper] = getattr(robot, gripper)[:,t]
    rave_body.set_pose(robot.pose[:,t])

    for param in list(plan.params.values()):
        if not param.is_symbol() and param != robot:
            if param.openrave_body is None: continue
            param.openrave_body.set_pose(param.pose[:, t].flatten(), param.rotation[:, t].flatten())
    collisions = p.getClosestPoints(rave_body.body_id, obs_body.body_id, 0.01)
    arm = None
    for col in collisions:
        r_link, obj_link = col[3], col[4]
        for a in r_geom.arms:
            if r_link in r_geom.get_arm_inds(a):
                arm = a
                break
        if arm is not None: break
    if arm is None:
        return None, None

    ee_link = r_geom.get_ee_link(arm)
    pos, orn = rave_body.get_link_pose(ee_link)
    
    attempt, step = 0, 1
    while attempt < 50 and len(collisions) > 0:
        attempt += 1
        target_ee = ee_pos + step * np.multiply(np.random.sample(3)+[-0.5,-0.5,0.25], const.RESAMPLE_FACTOR)
        #ik_arm_poses = rave_body.get_ik_from_pose(target_ee, [0, np.pi/2, 0], arm)
        ik_arm_poses = rave_body.get_ik_from_pose(target_ee, orn, arm, multiple=True)
        arm_pose = closest_arm_pose(ik_arm_poses, getattr(robot, arm)[:, action.active_timesteps[0]])
        if arm_pose is None:
            step += 1
            continue
        add_to_attr_inds_and_res(t, attr_inds, res, robot, [(arm, arm_pose)])
        rave_body.set_dof({arm: arm_pose})
        collisions = p.getClosestPoints(rave_body.body_id, obs_body.body_id, 0.01)
        collisions = list(filter(lambda col: col[3] in r_geom.get_arm_inds(arm), collisions))

    if not const.PRODUCTION:
        print("resampling at {} action".format(action.name))
    act_start, act_end = action.active_timesteps
    res_arm = arm
    if action.name.find("moveto") >=0  or action.name.find("moveholding") >= 0:
        timesteps_1 = t - act_start
        pose_traj_1 = lin_interp_traj(robot.pose[:, act_start], robot.pose[:, t], timesteps_1)
        add_to_attr_inds_and_res(i, attr_inds, res, robot, [('pose', pose_traj_1[:, traj_ind])])
        for arm in r_geom.arms:
            old_traj = getattr(robot, arm)
            arm_traj_1 = lin_interp_traj(old_traj[:, act_start], old_traj[:, t], timesteps_1)
            for i in range(act_start+1, t):
                traj_ind = i - act_start
                add_to_attr_inds_and_res(i, attr_inds, res, robot, [(arm, arm_traj_1[:, traj_ind])])

        timesteps_2 = act_end - t
        pose_traj_2 = lin_interp_traj(robot.pose[:, t], robot.pose[:, act_end], timesteps_2)
        arm_traj_2 = lin_interp_traj(old_traj[:, t], old_traj[:, act_end], timesteps_1)

        add_to_attr_inds_and_res(i, attr_inds, res, robot, [('pose', pose_traj_1[:, traj_ind])])
        for arm in r_geom.arms:
            for i in range(t+1, act_end):
                traj_ind = i - t
                add_to_attr_inds_and_res(i, attr_inds, res, robot, [(arm, arm_traj_2[:, traj_ind])])

    return res, attr_inds


#@profile
def resample_basket_obstructs_holding(pred, negated, t, plan):
    # viewer = OpenRAVEViewer.create_viewer(plan.env)
    attr_inds, res = OrderedDict(), OrderedDict()
    robot, obstacle = pred.robot, pred.obstacle
    rave_body, obs_body = robot.openrave_body, obstacle.openrave_body
    body, obs = rave_body.env_body, obs_body.env_body
    held, held_body = pred.obj, pred.obj.openrave_body
    held_env_body = held_body.env_body
    act_inds, action = [(i, act) for i, act in enumerate(plan.actions) if act.active_timesteps[0] <= t and  t <= act.active_timesteps[1]][0]
    dof_value = np.r_[robot.lArmPose[:, t],
                      robot.lGripper[:, t],
                      robot.rArmPose[:, t],
                      robot.rGripper[:, t],
                      robot.pose[:, t]]
    pred.set_robot_poses(dof_value, rave_body)
    for param in list(plan.params.values()):
        if not param.is_symbol() and param != robot:
            if param.openrave_body is None: continue
            param.openrave_body.set_pose(param.pose[:, t].flatten(), param.rotation[:, t].flatten())
    pred._cc.SetContactDistance(pred.dsafe)
    l_manip = body.GetManipulator("left_arm")
    r_manip = body.GetManipulator("right_arm")
    left_ee = OpenRAVEBody.obj_pose_from_transform(l_manip.GetTransform())[:3]
    right_ee = OpenRAVEBody.obj_pose_from_transform(r_manip.GetTransform())[:3]
    dist_left = np.linalg.norm(held.pose[:, t] - left_ee)
    dist_right = np.linalg.norm(held.pose[:, t] - right_ee)
    if dist_left < dist_right:
        arm = 'left'
        manip = l_manip
    else:
        arm = 'right'
        manip = r_manip

    pred._cc.SetContactDistance(pred.dsafe)
    collisions = pred._cc.BodyVsBody(body, obs)
    for col in collisions:
        linkA, linkB = col.GetLinkAName(), col.GetLinkBName()
        if linkA[0] == 'l' or linkB[0] == 'l':
            arm = 'left'
            manip = l_manip
            break
        elif linkA[0] == 'r' or linkB[0] == 'r':
            arm = 'right'
            manip = r_manip
            break
        else:
            continue

    pos_rot = OpenRAVEBody.obj_pose_from_transform(manip.GetTransform())
    ee_pos, ee_rot = pos_rot[:3], pos_rot[3:]
    attempt, step = 0, 1

    while attempt < 50 and (len(pred._cc.BodyVsBody(body, obs)) > 0 or len(pred._cc.BodyVsBody(held_env_body, obs)) > 0 ):
        attempt += 1
        target_ee = ee_pos + step * np.multiply(np.random.sample(3)+[-0.5,-0.5,0.5], [0.02, 0.02, 0.25])
        ik_arm_poses = rave_body.get_ik_from_pose(target_ee, [0, np.pi/2, 0], "{}_arm".format(arm))
        arm_pose = closest_arm_pose(ik_arm_poses, getattr(robot, "{}ArmPose".format(arm[0]))[:, action.active_timesteps[0]])
        if arm_pose is None:
            step += 1
            continue
        add_to_attr_inds_and_res(t, attr_inds, res, robot, [('{}ArmPose'.format(arm[0]), arm_pose)])
        add_to_attr_inds_and_res(t, attr_inds, res, held, [('pose', target_ee)])
        rave_body.set_dof({'{}ArmPose'.format(arm[0]): arm_pose})
        held_body.set_pose(target_ee)

    """
    Resample Trajectory
    """
    if not const.PRODUCTION:
        print("resampling at {} action".format(action.name))
    act_start, act_end = action.active_timesteps
    if action.name.find("moveto") >=0  or action.name.find("moveholding_basket") >= 0 or action.name.find("moveholding_cloth") >= 0:
        timesteps_1 = t - act_start
        pose_traj_1 = lin_interp_traj(robot.pose[:, act_start], robot.pose[:, t], timesteps_1)
        l_arm_traj_1 = lin_interp_traj(robot.lArmPose[:, act_start], robot.lArmPose[:, t], timesteps_1)
        r_arm_traj_1 = lin_interp_traj(robot.rArmPose[:, act_start], robot.rArmPose[:, t], timesteps_1)
        for i in range(max(t-5, act_start), t):
            traj_ind = i - act_start
            add_to_attr_inds_and_res(i, attr_inds, res, robot, [('lArmPose', l_arm_traj_1[:, traj_ind]), ('rArmPose', r_arm_traj_1[:, traj_ind]), ('pose', pose_traj_1[:, traj_ind])])
            rave_body.set_dof({'lArmPose': l_arm_traj_1[:, traj_ind], 'rArmPose': r_arm_traj_1[:, traj_ind]})
            rave_body.set_pose([0,0,pose_traj_1[:, traj_ind]])
            target_ee = OpenRAVEBody.obj_pose_from_transform(manip.GetTransform())[:3]
            add_to_attr_inds_and_res(i, attr_inds, res, held, [('pose', target_ee)])
            held_body.set_pose(target_ee)

        timesteps_2 = act_end - t
        pose_traj_2 = lin_interp_traj(robot.pose[:, t], robot.pose[:, act_end], timesteps_2)
        l_arm_traj_2 = lin_interp_traj(robot.lArmPose[:, t], robot.lArmPose[:, act_end], timesteps_2)
        r_arm_traj_2 = lin_interp_traj(robot.rArmPose[:, t], robot.rArmPose[:, act_end], timesteps_2)

        for i in range(t+1, min(t+5, act_end)):
            traj_ind = i - t
            add_to_attr_inds_and_res(i, attr_inds, res, robot, [('lArmPose', l_arm_traj_2[:, traj_ind]), ('rArmPose', r_arm_traj_2[:, traj_ind]), ('pose', pose_traj_2[:, traj_ind])])
            rave_body.set_dof({'lArmPose': l_arm_traj_2[:, traj_ind], 'rArmPose': r_arm_traj_2[:, traj_ind]})
            rave_body.set_pose([0,0,pose_traj_2[:, traj_ind]])
            target_ee = OpenRAVEBody.obj_pose_from_transform(manip.GetTransform())[:3]
            add_to_attr_inds_and_res(i, attr_inds, res, held, [('pose', target_ee)])
            held_body.set_pose(target_ee)

    return res, attr_inds

#@profile
def resample_washer_obstructs(pred, negated, t, plan):
    # viewer = OpenRAVEViewer.create_viewer(plan.env)
    attr_inds, res = OrderedDict(), OrderedDict()
    robot, obstacle = pred.robot, pred.obstacle
    rave_body, obs_body = robot.openrave_body, pred._param_to_body[obstacle][0] # obstacle.openrave_body
    body, obs = rave_body.env_body, obs_body.env_body
    obs_body.set_pose(obstacle.pose[:,t]-[.1,0,0])
    act_inds, action = [(i, act) for i, act in enumerate(plan.actions) if act.active_timesteps[0] <= t and  t < act.active_timesteps[1]][0]
    dof_value = np.r_[robot.lArmPose[:, t],
                      robot.lGripper[:, t],
                      robot.rArmPose[:, t],
                      robot.rGripper[:, t],
                      robot.pose[:, t]]
    pred.set_robot_poses(dof_value, rave_body)
    for param in list(plan.params.values()):
        if not param.is_symbol() and param != robot and param != obstacle:
            if param.openrave_body is None: continue
            param.openrave_body.set_pose(param.pose[:, t].flatten(), param.rotation[:, t].flatten())
    pred._cc.SetContactDistance(pred.dsafe)
    collisions = pred._cc.BodyVsBody(body, obs)
    arm = 'left'
    for col in collisions:
        linkA, linkB = col.GetLinkAName(), col.GetLinkBName()
        if linkA[0] == 'l' or linkB[0] == 'l':
            arm = 'left'
            break
        elif linkA[0] == 'r' or linkB[0] == 'r':
            arm = 'right'
            break
        else:
            continue
    if arm is None:
        return None, None

    # arm = "left"
    manip = body.GetManipulator("{}_arm".format(arm))
    robot_trans = manip.GetTransform()
    pos_rot = OpenRAVEBody.obj_pose_from_transform(robot_trans)
    ee_pos, ee_rot = pos_rot[:3], pos_rot[3:]
    attempt, step = 0, 1
    while attempt < 5 and len(pred._cc.BodyVsBody(body, obs)) > 0:
        attempt += 1
        target_ee = ee_pos + step * np.multiply(np.random.sample(3)+[-2, -1, 0.25], [0.05, 0.03, 0.02])
        ik_arm_poses = rave_body.get_ik_from_pose(target_ee, [0, np.pi/4, 0], "{}_arm".format(arm))
        arm_pose = closest_arm_pose(ik_arm_poses, getattr(robot, "{}ArmPose".format(arm[0]))[:, action.active_timesteps[0]])
        if arm_pose is None:
            step += 1
            continue
        add_to_attr_inds_and_res(t, attr_inds, res, robot, [('{}ArmPose'.format(arm[0]), arm_pose)])
        rave_body.set_dof({'{}ArmPose'.format(arm[0]): arm_pose})

    if not const.PRODUCTION:
        print("resampling at {} action".format(action.name))
    act_start, act_end = action.active_timesteps
    if action.name.find("moveto") >=0  or action.name.find("moveholding_basket") >= 0 or action.name.find("moveholding_cloth") >= 0:
        timesteps_1 = t - act_start
        pose_traj_1 = lin_interp_traj(robot.pose[:, act_start], robot.pose[:, t], timesteps_1)
        l_arm_traj_1 = lin_interp_traj(robot.lArmPose[:, act_start], robot.lArmPose[:, t], timesteps_1)
        r_arm_traj_1 = lin_interp_traj(robot.rArmPose[:, act_start], robot.rArmPose[:, t], timesteps_1)
        for i in range(act_start+1, t):
            traj_ind = i - act_start
            add_to_attr_inds_and_res(i, attr_inds, res, robot, [('lArmPose', l_arm_traj_1[:, traj_ind]), ('rArmPose', r_arm_traj_1[:, traj_ind]), ('pose', pose_traj_1[:, traj_ind])])
            rave_body.set_dof({'lArmPose': l_arm_traj_1[:, traj_ind], 'rArmPose': r_arm_traj_1[:, traj_ind]})
            rave_body.set_pose([0,0,pose_traj_1[:, traj_ind]])

        timesteps_2 = act_end - t
        pose_traj_2 = lin_interp_traj(robot.pose[:, t], robot.pose[:, act_end], timesteps_2)
        l_arm_traj_2 = lin_interp_traj(robot.lArmPose[:, t], robot.lArmPose[:, act_end], timesteps_2)
        r_arm_traj_2 = lin_interp_traj(robot.rArmPose[:, t], robot.rArmPose[:, act_end], timesteps_2)

        for i in range(t+1, act_end):
            traj_ind = i - t
            add_to_attr_inds_and_res(i, attr_inds, res, robot, [('lArmPose', l_arm_traj_2[:, traj_ind]), ('rArmPose', r_arm_traj_2[:, traj_ind]), ('pose', pose_traj_2[:, traj_ind])])
            rave_body.set_dof({'lArmPose': l_arm_traj_2[:, traj_ind], 'rArmPose': r_arm_traj_2[:, traj_ind]})
            rave_body.set_pose([0,0,pose_traj_2[:, traj_ind]])

    obs_body.set_pose([0,0,0])
    return res, attr_inds


#@profile
def resample_washer_rcollides(pred, negated, t, plan):
    # viewer = OpenRAVEViewer.create_viewer(plan.env)
    attr_inds, res = OrderedDict(), OrderedDict()
    robot, obstacle = pred.robot, pred.obstacle
    rave_body, obs_body = robot.openrave_body, obstacle.openrave_body
    body, obs = rave_body.env_body, obs_body.env_body
    act_inds, action = [(i, act) for i, act in enumerate(plan.actions) if act.active_timesteps[0] <= t and  t < act.active_timesteps[1]][0]
    dof_value = np.r_[robot.lArmPose[:, t],
                      robot.lGripper[:, t],
                      robot.rArmPose[:, t],
                      robot.rGripper[:, t],
                      robot.pose[:, t]]
    pred.set_robot_poses(dof_value, rave_body)
    for param in list(plan.params.values()):
        if not param.is_symbol() and param != robot:
            if param.openrave_body is None: continue
            param.openrave_body.set_pose(param.pose[:, t].flatten(), param.rotation[:, t].flatten())
    pred._cc.SetContactDistance(pred.dsafe)
    collisions = pred._cc.BodyVsBody(body, obs)
    arm = 'left'
    for col in collisions:
        linkA, linkB = col.GetLinkAName(), col.GetLinkBName()
        if linkA[0] == 'l' or linkB[0] == 'l':
            arm = 'left'
            break
        elif linkA[0] == 'r' or linkB[0] == 'r':
            arm = 'right'
            break
        else:
            continue
    if arm is None:
        return None, None

    # arm = "left"
    manip = body.GetManipulator("{}_arm".format(arm))
    robot_trans = manip.GetTransform()
    pos_rot = OpenRAVEBody.obj_pose_from_transform(robot_trans)
    ee_pos, ee_rot = pos_rot[:3], pos_rot[3:]
    attempt, step = 0, 1
    while attempt < 5 and len(pred._cc.BodyVsBody(body, obs)) > 0:
        attempt += 1
        target_ee = ee_pos + step * np.multiply(np.random.sample(3)+[-0.5, -0.5, -0.5], [0.025, 0.025, 0.025])
        ik_arm_poses = rave_body.get_ik_from_pose(target_ee, [0, np.pi/4, 0], "{}_arm".format(arm))
        arm_pose = closest_arm_pose(ik_arm_poses, getattr(robot, "{}ArmPose".format(arm[0]))[:, action.active_timesteps[0]])
        if arm_pose is None:
            step += 1
            continue
        add_to_attr_inds_and_res(t, attr_inds, res, robot, [('{}ArmPose'.format(arm[0]), arm_pose)])
        rave_body.set_dof({'{}ArmPose'.format(arm[0]): arm_pose})

    if not const.PRODUCTION:
        print("resampling at {} action".format(action.name))
    act_start, act_end = action.active_timesteps
    if action.name.find("moveto") >=0  or action.name.find("moveholding_basket") >= 0 or action.name.find("moveholding_cloth") >= 0:
        timesteps_1 = t - act_start
        pose_traj_1 = lin_interp_traj(robot.pose[:, act_start], robot.pose[:, t], timesteps_1)
        l_arm_traj_1 = lin_interp_traj(robot.lArmPose[:, act_start], robot.lArmPose[:, t], timesteps_1)
        r_arm_traj_1 = lin_interp_traj(robot.rArmPose[:, act_start], robot.rArmPose[:, t], timesteps_1)
        for i in range(act_start+1, t):
            traj_ind = i - act_start
            add_to_attr_inds_and_res(i, attr_inds, res, robot, [('lArmPose', l_arm_traj_1[:, traj_ind]), ('rArmPose', r_arm_traj_1[:, traj_ind]), ('pose', pose_traj_1[:, traj_ind])])
            rave_body.set_dof({'lArmPose': l_arm_traj_1[:, traj_ind], 'rArmPose': r_arm_traj_1[:, traj_ind]})
            rave_body.set_pose([0,0,pose_traj_1[:, traj_ind]])

        timesteps_2 = act_end - t
        pose_traj_2 = lin_interp_traj(robot.pose[:, t], robot.pose[:, act_end], timesteps_2)
        l_arm_traj_2 = lin_interp_traj(robot.lArmPose[:, t], robot.lArmPose[:, act_end], timesteps_2)
        r_arm_traj_2 = lin_interp_traj(robot.rArmPose[:, t], robot.rArmPose[:, act_end], timesteps_2)

        for i in range(t+1, act_end):
            traj_ind = i - t
            add_to_attr_inds_and_res(i, attr_inds, res, robot, [('lArmPose', l_arm_traj_2[:, traj_ind]), ('rArmPose', r_arm_traj_2[:, traj_ind]), ('pose', pose_traj_2[:, traj_ind])])
            rave_body.set_dof({'lArmPose': l_arm_traj_2[:, traj_ind], 'rArmPose': r_arm_traj_2[:, traj_ind]})
            rave_body.set_pose([0,0,pose_traj_2[:, traj_ind]])

    return res, attr_inds

#@profile
def resample_rcollides(pred, negated, t, plan):
    # Variable that needs to added to BoundExpr and latter pass to the planner
    JOINT_STEP = 20
    STEP_DECREASE_FACTOR = 1.5
    ATTEMPT_SIZE = 7
    LIN_SAMP_RANGE = 5

    attr_inds = OrderedDict()
    res = OrderedDict()
    robot, rave_body = pred.robot, pred._param_to_body[pred.robot]
    body = rave_body.env_body
    manip = body.GetManipulator("right_arm")
    arm_inds = manip.GetArmIndices()
    lb_limit, ub_limit = body.GetDOFLimits()
    step_factor = JOINT_STEP
    joint_step = (ub_limit[arm_inds] - lb_limit[arm_inds])/ step_factor
    original_pose, arm_pose = robot.rArmPose[:, t].copy(), robot.rArmPose[:, t].copy()
    rave_body.set_pose([0,0,robot.pose[:, t]])
    rave_body.set_dof({"lArmPose": robot.lArmPose[:, t].flatten(),
                       "lGripper": robot.lGripper[:, t].flatten(),
                       "rArmPose": robot.rArmPose[:, t].flatten(),
                       "rGripper": robot.rGripper[:, t].flatten()})

    ## Determine the range we should resample
    pred_list = [act_pred['active_timesteps'] for act_pred in plan.actions[0].preds if act_pred['pred'].spacial_anchor == True]
    start, end = 0, plan.horizon-1
    for action in plan.actions:
        if action.active_timesteps[0] <= t and action.active_timesteps[1] > t:
            for act_pred in plan.actions[0].preds:
                if act_pred['pred'].spacial_anchor == True:
                    if act_pred['active_timesteps'][0] + act_pred['pred'].active_range[0] > t:
                        end = min(end, act_pred['active_timesteps'][0] + act_pred['pred'].active_range[0])
                    if act_pred['active_timesteps'][1] + act_pred['pred'].active_range[1] < t:
                        start = max(start, act_pred['active_timesteps'][1] + act_pred['pred'].active_range[1])

    desired_end_pose = robot.rArmPose[:, end]
    current_end_pose = robot.rArmPose[:, t]
    col_report = CollisionReport()
    collisionChecker = RaveCreateCollisionChecker(plan.env,'pqp')
    count = 1
    while (body.CheckSelfCollision() or
           collisionChecker.CheckCollision(body, report=col_report) or
           col_report.minDistance <= pred.dsafe):
        step_sign = np.ones(len(arm_inds))
        step_sign[np.random.choice(len(arm_inds), len(arm_inds)/2, replace=False)] = -1
        # Ask in collision pose to randomly move a step, hopefully out of collision
        arm_pose = original_pose + np.multiply(step_sign, joint_step)
        rave_body.set_dof({"rArmPose": arm_pose})
        # arm_pose = body.GetActiveDOFValues()[arm_inds]
        if not count % ATTEMPT_SIZE:
            step_factor = step_factor/STEP_DECREASE_FACTOR
            joint_step = (ub_limit[arm_inds] - lb_limit[arm_inds])/ step_factor
        count += 1

        # For Debug
        rave_body.set_pose([0,0,robot.pose[:, t]])
    add_to_attr_inds_and_res(t, attr_inds, res, robot,[('rArmPose', arm_pose)])
    robot._free_attrs['rArmPose'][:, t] = 0


    start, end = max(start, t-LIN_SAMP_RANGE), min(t+LIN_SAMP_RANGE, end)
    rcollides_traj = np.hstack([lin_interp_traj(robot.rArmPose[:, start], arm_pose, t-start), lin_interp_traj(arm_pose, robot.rArmPose[:, end], end - t)[:, 1:]]).T
    i = start + 1
    for traj in rcollides_traj[1:-1]:
        add_to_attr_inds_and_res(i, attr_inds, res, robot, [('rArmPose', traj)])
        i +=1


    return np.array(res), attr_inds

# Alternative approaches, frequently failed, Not used
#@profile
def get_col_free_armPose_ik(pred, negated, t, plan):
    ee_pose = OpenRAVEBody.obj_pose_from_transform(body.env_body.GetManipulator('right_arm').GetTransform())
    pos, rot = ee_pose[:3], ee_pose[3:]
    while arm_pose is None and iteration < const.MAX_ITERATION_STEP:
    # for i in range(const.NUM_RESAMPLES):
        pos_bias = np.random.random_sample((3,))*const.BIAS_RADIUS*2 - const.BIAS_RADIUS
        rot_bias = np.random.random_sample((3,))*const.ROT_BIAS*2 - const.ROT_BIAS
        # print pos_bias, rot_bias, iteration
        print(pos_bias, rot_bias)
        iteration += 1
        arm_pose = get_ik_from_pose(pos + pos_bias, rot + rot_bias, body.env_body, 'right_arm')
        if arm_pose is not None:
            print(iteration)
            body.set_dof({'rArmPose': arm_pose})

#@profile
def sample_arm_pose(robot_body, old_arm_pose=None):
    dof_inds = robot_body.GetManipulator("right_arm").GetArmIndices()
    lb_limit, ub_limit = robot_body.GetDOFLimits()
    active_ub = ub_limit[dof_inds].flatten()
    active_lb = lb_limit[dof_inds].flatten()
    if old_arm_pose is not None:
        arm_pose = np.random.random_sample((len(dof_inds),)) - 0.5
        arm_pose = np.multiply(arm_pose, (active_ub - active_lb)/5) + old_arm_pose
    else:
        arm_pose = np.random.random_sample((len(dof_inds),))
        arm_pose = np.multiply(arm_pose, active_ub - active_lb) + active_lb
    return arm_pose

#@profile
def add_to_attr_inds_and_res(t, attr_inds, res, param, attr_name_val_tuples):
    # param_attr_inds = []
    if param.is_symbol():
        t = 0
    for attr_name, val in attr_name_val_tuples:
        inds = np.where(param._free_attrs[attr_name][:, t])[0]
        getattr(param, attr_name)[inds, t] = val[inds]
        if param in attr_inds:
            res[param].extend(val[inds].flatten().tolist())
            attr_inds[param].append((attr_name, inds, t))
        else:
            res[param] = val[inds].flatten().tolist()
            attr_inds[param] = [(attr_name, inds, t)]

#@profile
def test_resample_order(attr_inds, res):
    for p in attr_inds:
        i = 0
        for attr, inds, t in attr_inds[p]:
            if not np.allclose(getattr(p, attr)[inds, t], res[p][i:i+len(inds)]):
                print(getattr(p, attr)[inds, t])
                print("v.s.")
                print(res[p][i:i+len(inds)])
            i += len(inds)


#@profile
def resample_eereachable(pred, negated, t, plan, inv=False):
    attr_inds, res = OrderedDict(), OrderedDict()
    robot, robot_body = pred.robot, pred._param_to_body[pred.robot]
    if hasattr(pred, 'obj'):
        targ_pos = pred.obj.pose[:,t]
    elif hasattr(pred, 'targ'):
        targ_pos = pred.targ.value[:,0]

    acts = [a for a in plan.actions if a.active_timesteps[0] < t and a.active_timesteps[1] >= t]
    if not len(acts): return None, None
    act = acts[0]
    a_st, a_et = act.active_timesteps
    arm = pred.arm
    robot_body.set_dof({arm: getattr(robot, arm)[:,t]})
    info = robot_body.fwd_kinematics(arm)
    pos, quat = info['pos'], info['quat']
    mat = T.quat2mat(quat)
    st, et = pred.active_range
    for ts in range(max(a_st, t+st), min(a_et-1, t+et)):
        dist = pred.approach_dist if ts <= t else pred.retreat_dist
        vec = dist * np.abs(t-ts) * pred.axis
        ik = robot_body.get_ik_from_pose(targ_pos+vec, quat, arm)
        add_to_attr_inds_and_res(ts, attr_inds, res, robot, [(arm, np.array(ik).flatten())])

    return np.array(res), attr_inds
