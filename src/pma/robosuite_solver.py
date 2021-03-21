import numpy as np
import pybullet as p

from sco.expr import BoundExpr, QuadExpr, AffExpr
from core.internal_repr.parameter import Object
from core.util_classes.matrix import Vector
import core.util_classes.common_constants as const
from core.util_classes.openrave_body import OpenRAVEBody
import core.util_classes.robot_sampling as robot_sampling
import core.util_classes.transform_utils as T
from pma import backtrack_ll_solver


REF_JNTS = np.array([0, -np.pi/4, 0, np.pi/4, 0, np.pi/2, 0])
REF_JNTS = np.array([0, -np.pi/3, 0, np.pi/3, 0, np.pi/2, 0])
REF_JNTS = np.array([0, -np.pi/3, 0, np.pi/6, 0, 2*np.pi/3, 0])

class RobotSolver(backtrack_ll_solver.BacktrackLLSolver):
    def get_resample_param(self, a):
        if a.name.find('move') < 0 and a.name.find('grasp') >= 0:
            return [a.params[0], a.params[1]]

        if a.name.find('move') >= 0 and a.name.find('put') >= 0:
            return [a.params[0], a.params[2]]

        return a.params[0]


    def freeze_rs_param(self, act):
        return True


    def vertical_gripper(self, robot, arm, obj, gripper_open=True, ts=(0,20), rand=False, null_zero=True):
        start_ts, end_ts = ts
        robot_body = robot.openrave_body
        robot_body.set_from_param(robot, start_ts)

        old_arm_pose = getattr(robot, arm)[:, start_ts].copy()
        offset = obj.geom.grasp_point if hasattr(obj.geom, 'grasp_point') else np.zeros(3)
        if obj.is_symbol():
            target_loc = obj.value[:, 0] + np.array([0, 0, const.GRASP_DIST]) + offset
        else:
            target_loc = obj.pose[:, start_ts] + np.array([0, 0, const.GRASP_DIST]) + offset

        gripper_axis = robot.geom.get_gripper_axis(arm)
        target_axis = [0, 0, -1]
        quat = OpenRAVEBody.quat_from_v1_to_v2(gripper_axis, target_axis)

        euler = obj.rotation[:,ts[0]] if not obj.is_symbol() else obj.rotation[:,0]
        obj_quat = T.euler_to_quaternion(euler, 'xyzw')
        robot_mat = T.quat2mat(quat)
        obj_mat = T.quat2mat(obj_quat)
        quat = T.mat2quat(obj_mat.dot(robot_mat))

        iks = []
        attempt = 0
        robot_body.set_pose(robot.pose[:,ts[0]])
        #robot_body.set_dof({arm: np.zeros(len(robot.geom.jnt_names[arm]))})
        #if not null_zero: #ts[1]-ts[0] > 5:
        #    robot_body.set_dof({arm: getattr(robot, arm)[:, ts[0]]})
        #robot_body.set_dof({arm: getattr(robot, arm)[:, ts[0]]})
        robot_body.set_dof({arm: REF_JNTS})

        while not len(iks) and attempt < 20:
            if rand:
                target_loc += np.clip(np.random.normal(0, 0.015, 3), -0.03, 0.03)

            iks = robot_body.get_ik_from_pose(target_loc, quat, arm)
            rand = not len(iks)
            attempt += 1
        if not len(iks): return None
        arm_pose = np.array(iks).reshape((-1,1))
        pose = {arm: arm_pose}
        gripper = robot.geom.get_gripper(arm)
        if gripper is not None:
            pose[gripper] = robot.geom.get_gripper_open_val() if gripper_open else robot.geom.get_gripper_closed_val()
            pose[gripper] = np.array(pose[gripper]).reshape((-1,1))
        for aux_arm in robot.geom.arms:
            if aux_arm == arm: continue
            old_pose = getattr(robot, aux_arm)[:, start_ts].reshape((-1,1))
            pose[aux_arm] = old_pose
            aux_gripper = robot.geom.get_gripper(aux_arm)
            if aux_gripper is not None:
                pose[aux_gripper] = getattr(robot, aux_gripper)[:, start_ts].reshape((-1,1))

        robot_body.set_pose(robot.pose[:,ts[0]])
        for arm in robot.geom.arms:
            robot_body.set_dof({arm: pose[arm].flatten().tolist()})
            info = robot_body.fwd_kinematics(arm)
            pose['{}_ee_pos'.format(arm)] = np.array(info['pos']).reshape((-1,1))
            pose['{}_ee_rot'.format(arm)] = np.array(T.quaternion_to_euler(info['quat'], 'xyzw')).reshape((-1,1))
        return pose

    def obj_in_gripper(self, ee_pos, targ_rot, obj):
        pose = {}
        pose['pose'] = ee_pos.flatten() - np.array(obj.geom.grasp_point).flatten()
        pose['pose'] = pose['pose'].reshape((-1,1))
        pose['rotation'] = targ_rot.reshape((-1,1))
        return pose

    def obj_pose_suggester(self, plan, anum, resample_size=20, st=0):
        robot_pose = []
        assert anum + 1 <= len(plan.actions)

        if anum + 1 < len(plan.actions):
            act, next_act = plan.actions[anum], plan.actions[anum+1]
        else:
            act, next_act = plan.actions[anum], None

        robot = act.params[0]
        robot_body = robot.openrave_body
        act_st, et = act.active_timesteps
        st = max(act_st, st)
        zero_null = True
        if hasattr(plan, 'freeze_ts') and plan.freeze_ts > 0:
            st = max(st, plan.freeze_ts)
            zero_null = False

        robot_body.set_pose(robot.pose[:,st])
        #for arm in robot.geom.arms:
        #    robot_body.set_dof({arm: getattr(robot, arm)[:,st].flatten().tolist()})

        #for arm in robot.geom.arms:
        #    attr = '{}_ee_pos'.format(arm)
        #    #rot_attr = '{}_ee_rot'.format(arm)
        #    if hasattr(robot, attr):
        #        info = robot_body.fwd_kinematics(arm)
        #        getattr(robot, attr)[:, st] = info['pos']
        #        #getattr(robot, rot_attr)[:, st] = T.quaternion_to_euler(info['quat'], 'xyzw')

        obj = act.params[1]
        targ = act.params[2]
        st, et = act.active_timesteps
        for param in plan.params.values():
            if hasattr(param, 'openrave_body') and param.openrave_body is not None:
                if param.is_symbol():
                    if hasattr(param, 'rotation'):
                        param.openrave_body.set_pose(param.value[:,0], param.rotation[:,0])
                    else:
                        param.openrave_body.set_pose(param.value[:,0])
                else:
                    if hasattr(param, 'rotation'):
                        param.openrave_body.set_pose(param.pose[:,st], param.rotation[:,st])
                    else:
                        param.openrave_body.set_pose(param.pose[:,st])

        a_name = act.name.lower()
        arm = robot.geom.arms[0]
        if a_name.find('left') >= 0: arm = 'left'
        if a_name.find('right') >= 0: arm = 'right'

        gripper_open = False
        if a_name.find('move_to_grasp') >= 0 or (a_name.find('putdown') >= 0 and a_name.find('move_to') < 0):
            gripper_open = True

        rand = False # a_name.find('move') >= 0

        if next_act is not None:
            next_obj = next_act.params[1]
            next_a_name = next_act.name.lower()
            next_arm = robot.geom.arms[0]
            next_gripper_open = not gripper_open
            next_st, next_et = next_act.active_timesteps
            if next_a_name.find('left') >= 0: arm = 'left'
            if next_a_name.find('right') >= 0: arm = 'right'

        ### Sample poses
        for i in range(resample_size):
            ### Cases for when behavior can be inferred from current action
            if a_name.find('grasp') >= 0:
                pose = self.vertical_gripper(robot, arm, obj, gripper_open, (st, et), rand=(rand or (i>0)), null_zero=zero_null)
            elif a_name.find('putdown') >= 0:
                pose = self.vertical_gripper(robot, arm, obj, gripper_open, (st, et), rand=(rand or (i>0)), null_zero=zero_null)

            ### Cases for when behavior cannot be inferred from current action
            elif next_act is None:
                pose = None

            elif next_a_name.find('grasp') >= 0 or next_a_name.find('putdown') >= 0:
                pose = self.vertical_gripper(robot, next_arm, next_obj, next_gripper_open, (next_st, next_et), rand=(rand or (i>0)))

            if a_name.find('grasp') >= 0 and a_name.find('move') < 0:
                obj = act.params[1]
                targ = act.params[2]
                pose = {robot: pose, obj: self.obj_in_gripper(pose['{}_ee_pos'.format(arm)], targ.rotation[:,0], obj)}

            if a_name.find('put') >= 0 and a_name.find('move') >= 0:
                obj = act.params[2]
                targ = act.params[1]
                pose = {robot: pose, obj: self.obj_in_gripper(pose['{}_ee_pos'.format(arm)], targ.rotation[:,0], obj)}
                obj = act.params[1]
                targ = act.params[2]

            if pose is None: break
            robot_pose.append(pose)

        return robot_pose


    def _cleanup_plan(self, plan, active_ts):
        for param in plan.params.values():
            if 'Robot' not in param.get_type(True): continue
            for arm in param.geom.arms:
                attr = '{}_ee_pos'.format(arm)
                rot_attr = '{}_ee_rot'.format(arm)
                if not hasattr(param, attr): continue
                for t in range(active_ts[0], active_ts[1]):
                    if np.any(np.isnan(getattr(param, arm)[:,t])): continue
                    param.openrave_body.set_dof({arm: getattr(param, arm)[:,t]})
                    info = param.openrave_body.fwd_kinematics(arm)
                    getattr(param, attr)[:,t] = info['pos']



    def _get_trajopt_obj(self, plan, active_ts=None):
        if active_ts == None:
            active_ts = (0, plan.horizon-1)
        start, end = active_ts
        traj_objs = []
        for param in list(plan.params.values()):
            if param not in self._param_to_ll:
                continue
            if isinstance(param, Object):
                for attr_name in param.__dict__.keys():
                    attr_type = param.get_attr_type(attr_name)
                    if issubclass(attr_type, Vector):
                        T = end - start + 1
                        K = attr_type.dim
                        attr_val = getattr(param, attr_name)
                        KT = K*T
                        v = -1 * np.ones((KT - K, 1))
                        d = np.vstack((np.ones((KT - K, 1)), np.zeros((K, 1))))
                        # [:,0] allows numpy to see v and d as one-dimensional so
                        # that numpy will create a diagonal matrix with v and d as a diagonal
                        P = np.diag(v[:, 0], K) + np.diag(d[:, 0])
                        Q = np.dot(np.transpose(P), P)
                        Q *= self.trajopt_coeff/float(plan.horizon)

                        quad_expr = None
                        coeff = 1.
                        if attr_name.find('ee_pos') >= 0:
                            coeff = 7e-3
                        elif attr_name.find('ee_rot') >= 0:
                            coeff = 2e-3
                        elif attr_name.find('right') >= 0 or attr_name.find('left') >= 0:
                            coeff = 1e1
                        else:
                            coeff = 1e-2

                        quad_expr = QuadExpr(coeff*Q, np.zeros((1,KT)), np.zeros((1,1)))
                        param_ll = self._param_to_ll[param]
                        ll_attr_val = getattr(param_ll, attr_name)
                        param_ll_grb_vars = ll_attr_val.reshape((KT, 1), order='F')
                        attr_val = getattr(param, attr_name)
                        init_val = attr_val[:, start:end+1].reshape((KT, 1), order='F')
                        sco_var = self.create_variable(param_ll_grb_vars, init_val)
                        bexpr = BoundExpr(quad_expr, sco_var)
                        traj_objs.append(bexpr)

            #traj_objs.extend(self._get_null_obj(plan, active_ts))
        return traj_objs


    def _get_null_obj(self, plan, active_ts=None):
        if active_ts == None:
            active_ts = (0, plan.horizon-1)

        act = [a for a in plan.actions if a.active_timesteps[0] >= active_ts[0] and a.active_timesteps[1] <= active_ts[1]][0]
        if act.name.find('move') < 0: return []
        start, end = active_ts
        traj_objs = []
        param = plan.params['sawyer']
        T = end - start + 1
        attr_name = 'right'
        attr_type = param.get_attr_type('right')
        K = attr_type.dim
        KT = K*T
        coeff = 1e-3
        ref_jnts = np.tile(REF_JNTS, (T,))
        c = np.sum(ref_jnts**2)
        quad_expr = QuadExpr(coeff*np.eye(KT), -2*coeff*ref_jnts.reshape((1, KT)), c*coeff*np.ones((1,1)))
        param_ll = self._param_to_ll[param]
        ll_attr_val = getattr(param_ll, attr_name)
        param_ll_grb_vars = ll_attr_val.reshape((KT, 1), order='F')
        attr_val = getattr(param, attr_name)
        init_val = attr_val[:, start:end+1].reshape((KT, 1), order='F')
        sco_var = self.create_variable(param_ll_grb_vars, init_val)
        bexpr = BoundExpr(quad_expr, sco_var)
        traj_objs.append(bexpr)
        return traj_objs

