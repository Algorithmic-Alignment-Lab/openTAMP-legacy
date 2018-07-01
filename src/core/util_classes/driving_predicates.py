from collections import OrderedDict
import numpy as np

from sco.expr import Expr, AffExpr, EqExpr, LEqExpr

from core.util_classes.common_predicates import ExprPredicate

from internal_state.dynamics import *

MOVE_FACTOR = 2
END_DIST = 4
COL_DIST = 0.5

DYNAMICS_COEFF = 1
COLLISION_COEFF = 0.1
LOC_COEFF = 1
FOLLOW_COEFF = 0.01
STOP_COEFF = 1
DOWN_ROAD_COEFF = 0.1
DIST_COEFF = 1

def add_to_attr_inds_and_res(t, attr_inds, res, param, attr_name_val_tuples):
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

class DrivingPredicate(ExprPredicate):
    '''
    Used to introduce a layer of interface to the simulator but wans't necessary.
    May become revleant in future.
    '''
    def __init__(self, name, e, attr_inds, params, expected_param_types, active_range, env, priority):
        super(DrivingPredicate, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=active_range, env=env, priority=priority)

class HLPred(DrivingPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        attr_inds = OrderedDict([(self.obj, [("xy", np.array([0,1], dtype=np.int))])])
        A = np.zeros((2,2))
        b = np.zeros((2,1))
        val = np.zeros((2, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(HLPred, self).__init__(name, e, attr_inds, params, expected_param_types, env=env, priority=-2)    

class HLNoCollisions(HLPred):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 1
        self.obj, = params

        super(HLNoCollisions, self).__init__(name, params, expected_param_types, env=env, priority=-2)
        self.spacial_anchor = False

    def check_if_true(self, env):
        return np.any([env.check_all_collisions(v) for v in env.user_vehicles]) or \
               np.any([env.check_all_collisions(v) for v in env.external_vehicles])

class HLCrateInTrunk(HLPred):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 2
        self.obj, self.crate = params

        super(HLCrateInTrunk, self).__init__(name, params, expected_param_types, env=env, priority=-2)
        self.spacial_anchor = False

    def check_if_true(self, env):
        return self.obj.geom.in_trunk(self.crate.geom)

class DynamicPredicate(DrivingPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 1
        self.obj, = params
        attr_inds = OrderedDict([(self.obj, [("xy", np.array([0,1], dtype=np.int)),
                                             ("theta", np.array([0], dtype=np.int)),
                                             ("vel", np.array([0], dtype=np.int)),
                                             ("phi", np.array([0], dtype=np.int)),
                                             ("u1", np.array([0], dtype=np.int)),
                                             ("u2", np.array([0], dtype=np.int))])])

        val = np.zeros((1, 14))
        dynamics_expr = Expr(self.f, self.grad)
        e = EqExpr(dynamics_expr, val)

        super(DynamicPredicate, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), env=env, priority=1)
        self.spacial_anchor = False

class XValid(DynamicPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        def f(x):
            return DYNAMICS_COEFF * (np.array([x[7] - next_px_f(x[0], x[2], x[3], x[5], x[6], self.sess)]))

        def grad(x):
            grad = np.zeros((1, 14))
            grad[0,7] = 1
            grad[0,5], grad[0,6] = next_px_grad(x[0], x[2], x[3], x[5], x[6], self.sess)
            return DYNAMICS_COEFF * grad

        self.f = f
        self.grad = grad
        super(XValid).__init__(name, params, expected_param_types, env)
        
    def resample(self, negated, t, plan):
        if negated:
            return None, None
        else:
            attr_inds, res = OrderedDict(), OrderedDict()
            act_inds, action = [(i, act) for i, act in enumerate(plan.actions) if act.active_timesteps[0] < t and  t <= act.active_timesteps[1]][0]
            act_start, act_end = action.active_timesteps

            old_pose1 = self.obj1.geom.update_xy_theta(x[0], x[1], x[2], 0)
            old_pose2 = self.obj2.geom.update_xy_theta(x[3], x[4], x[5], 0)
            obj1_pts = self.obj1.geom.get_points(0, COL_DIST)
            obj2_pts = self.obj2.geom.get_points(0, COL_DIST)
            self.obj1.geom.update_xy_theta(0, old_pose1[0], old_pose1[1], old_pose1[2])
            self.obj2.geom.update_xy_theta(0, old_pose2[0], old_pose2[1], old_pose2[2])
            col_vec = collision_vector(obj1_pts, obj2_pts)

            random_dir = np.random.randint(-1, 2) * np.array([-1./col_vec[0], 1./col_vec[1]])
            target_xy = np.random.uniform(1, 3) * col_vec - self.obj1.xy[t] + np.random.uniform(1, 3) * random_dir / np.linalg.norm(random_dir)

            start_t = max(t - 5, act_start)
            start_xy = self.obj1.xy[:, start_t]
            start_theta = self.obj1.theta[0, start_t]
            end_theta = np.arccos((target_xy - start_xy).dot([1,0]) / np.linalg.norm(target_xy - start_xy))

            for i in range(start_t+1, t+1):
                t_ratio = float((i - start_t)) / (t - start_t)
                add_to_attr_inds_and_res(i, attr_inds, res, self.obj1, [('xy', t_ratio * target_xy + (1 - t_ratio) * start_xy), ('theta', t_ratio * end_theta + (1 - t_ratio) * start_theta)])

            return attr_inds, res

class YValid(DynamicPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        def f(x):
            return DYNAMICS_COEFF * (np.array([x[8] - next_py_f(x[1], x[2], x[3], x[5], x[6], self.sess)]))

        def grad(x):
            grad = np.zeros((1, 14))
            grad[0,8] = 1
            grad[0,5], grad[0,6] = next_py_grad(x[1], x[2], x[3], x[5], x[6], self.sess)
            return DYNAMICS_COEFF * grad

        self.f = f
        self.grad = grad
        super(YValid).__init__(name, params, expected_param_types, env)

class ThetaValid(DynamicPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        def f(x):
            return DYNAMICS_COEFF * (np.array([x[9] - next_theta_f(x[2], x[3], x[4], x[5], x[6], self.sess)]))

        def grad(x):
            grad = np.zeros((1, 14))
            grad[0,9] = 1
            grad[0,5], grad[0,6] = next_theta_grad(x[2], x[3], x[4], x[5], x[6], self.sess)
            return DYNAMICS_COEFF * grad

        self.f = f
        self.grad = grad
        super(ThetaValid).__init__(name, params, expected_param_types, env)

class VelocityValid(DynamicPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        def f(x):
            return DYNAMICS_COEFF * (np.array([x[10] - next_v_f(x[3], x[5], x[6], self.sess)]))

        def grad(x):
            grad = np.zeros((1, 14))
            grad[0,10] = 1
            grad[0,5], grad[0,6] = next_v_grad(x[3], x[5], x[6], self.sess)
            return DYNAMICS_COEFF * grad

        self.f = f
        self.grad = grad
        super(VelocityValid).__init__(name, params, expected_param_types, env)
        
class PhiValid(DynamicPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        def f(x):
            return DYNAMICS_COEFF * (np.array([x[11] - next_phi_f(x[4], x[5], x[6], self.sess)]))

        def grad(x):
            grad = np.zeros((1, 14))
            grad[0,11] = 1
            grad[0,5], grad[0,6] = next_phi_grad(x[4], x[5], x[6], self.sess)
            return DYNAMICS_COEFF * grad

        self.f = f
        self.grad = grad
        super(PhiValid).__init__(name, params, expected_param_types, env)
        
class At(DrivingPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 2
        self.obj, self.target = params
        attr_inds = OrderedDict([(self.obj,    [("xy", np.array([0,1], dtype=np.int)),
                                                ("theta", np.array([0], dtype=np.int))]),
                                 (self.target, [("xy", np.array([0,1], dtype=np.int)),
                                                ("theta", np.array([0], dtype=np.int))])])

        A = np.c_[np.eye(3), -np.eye(3)]
        b, val = np.zeros((3, 1)), np.zeros((3, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)

        super(At, self).__init__(name, e, attr_inds, params, expected_param_types, env=env, priority=-2)
        self.spacial_anchor = True
    
class VehicleAt(At):
    pass

class CrateAt(At):
    pass

class ObstacleAt(At):
    pass

class Near(DrivingPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 3
        self.obj, self.target, self.dist = params
        attr_inds = OrderedDict([(self.obj,    [("xy", np.array([0,1], dtype=np.int))]),
                                 (self.target, [("xy", np.array([0,1], dtype=np.int))]),,
                                 (self.dist,   [("value", np.array([0], dtype=np.int))])])

        A = np.c_[np.r_[np.eye(2), -np.eye(2)], np.r_[-np.eye(2), np.eye(2)], -np.ones((4,1))]
        b, val = np.zeros((4, 1)), np.zeros((4, 1))
        aff_e = AffExpr(A, b)
        e = LEqExpr(aff_e, val)

        super(At, self).__init__(name, e, attr_inds, params, expected_param_types, env=env, priority=-2)
        self.spacial_anchor = True

class VehicleAtSign(Near):
    pass

class VelAt(DrivingPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 2
        self.obj, self.target = params
        attr_inds = OrderedDict([(self.obj,    [("vel", np.array([0], dtype=np.int))]),
                                 (self.target, [("value", np.array([0], dtype=np.int))])])

        A = np.c_[1, -1]
        b, val = np.zeros((1, 1)), np.zeros((1, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)

        super(VelAt, self).__init__(name, e, attr_inds, params, expected_param_types, env=env, priority=-2)
        self.spacial_anchor = True

class VehicleVelAt(Velt):
    pass

class ExternalVehicleVelAt(VelAt):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 2
        self.obj, self.target = params

        if self.obj.geom.is_user:
            attr_inds = OrderedDict([(self.obj,    [("vel", np.array([0], dtype=np.int))]),
                                     (self.target, [("value", np.array([0], dtype=np.int))])])

            A = np.c_[0, 0]
            b, val = np.zeros((1, 1)), np.zeros((1, 1))
            aff_e = AffExpr(A, b)
            e = EqExpr(aff_e, val)

            super(VelAt, self).__init__(name, e, attr_inds, params, expected_param_types, env=env, priority=-2)
            self.spacial_anchor = True

        else:
            super(ExternalVehicleVelAt, self).__init__(name, params, expected_param_types)

class ExternalVehiclePastRoadEnd(DrivingPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 1

        self.obj, = params
        attr_inds = OrderedDict([(self.obj, [("xy", np.array([0,1], dtype=np.int))])])
        if not self.obj.geom.road:
            A = np.zeros((2,2))
            b = np.zeros((2,1))
            val = np.zeros((2, 1))
            aff_e = AffExpr(A, b)
            e = EqExpr(aff_e, val)

        else:
            direction = self.obj.geom.road
            rot_mat = np.array([[np.cos(direction), -np.sin(direction)],
                                [np.sin(direction), np.cos(direction)]])

            road_len = self.obj.geom.road.length
            self.road_end = np.array([self.obj.geom.road.x, self.obj.geom.road.y]) + rot_mat.dot([road_len + END_DIST, 0])

            A = np.eye(2)
            b = -self.road_end
            val = np.zeros((2, 1))
            aff_e = AffExpr(A, b)
            e = EqExpr(aff_e, val)

        super(ExternalVehiclePastRoadEnd, self).__init__(name, e, attr_inds, params, expected_param_types, env=env, priority=-2)
        self.spacial_anchor = True

class Stationary(DrivingPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 1
        self.obj, = params
        attr_inds = OrderedDict([(self.obj, [("xy", np.array([0,1], dtype=np.int)),
                                             ("theta", np.array([0], dtype=np.int))])])

        A = np.c_[np.eye(3), -np.eye(3)]
        b, val = np.zeros((3, 1)), np.zeros((3, 1))
        e = EqExpr(AffExpr(A, b), val)
        super(Stationary, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), env=env, priority=-2)
        self.spacial_anchor = False

class VehicleStationary(Stationary):
    pass

class CrateStationary(Stationary):
    pass

class ObstacleStationary(Stationary):
    pass

class StationaryLimit(DrivingPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 1
        self.limit, = params
        attr_inds = OrderedDict([(self.limit, [("value", np.array([0], dtype=np.int))])])

        A = np.c_[1, -1]
        b, val = np.zeros((1, 1)), np.zeros((1, 1))
        e = EqExpr(AffExpr(A, b), val)
        super(StationaryLimit, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), env=env, priority=-2)
        self.spacial_anchor = False

class IsMP(DrivingPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 1
        self.obj, = params
        attr_inds = OrderedDict([(self.obj, [("xy", np.array([0,1], dtype=np.int)),
                                             ("theta", np.array([0], dtype=np.int))])])

        A = np.c_[np.r_[np.eye(3), -np.eye(3)], np.r_[-np.eye(3), np.eye(3)]]
        b, val = np.zeros((6, 1)), MOVE_FACTOR * np.ones((6, 1))
        e = LEqExpr(AffExpr(A, b), val)
        super(IsMP, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), env=env, priority=-2)
        self.spacial_anchor = False

class OnSurface(DrivingPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 2
        self.obj, self.surface = params
        attr_inds = OrderedDict([(self.obj, [("xy", np.array([0,1], dtype=np.int))])])

        f = lambda x: LOC_COEFF * self.surface.geom.to(x[0], x[1])
        grad = lambda x: LOC_COEFF * np.eye(2)

        val = np.zeros((2, 1))
        expr = Expr(f, grad)
        e = EqExpr(expr, val)

        super(OnSurface, self).__init__(name, e, attr_inds, params, expected_param_types, env=env, priority=2)
        self.spacial_anchor = False

class OnRoad(OnSurface):
    pass

class OnLot(DrivingPredicate):
    pass

class InLane(DrivingPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 3
        self.obj, self.road, self.lane_num = params
        attr_inds = OrderedDict([(self.obj, [("xy",    np.array([0,1], dtype=np.int)),
                                             ("theta", np.array([0], dtype=np.int))])])

        f = lambda x: LOC_COEFF * self.road.geom.to_lane(x[0], x[1], x[2], self.lane_num.value[0, 0])[0]
        def grad(x):
            grad = np.zeros((3, 2))
            grad[:2,:] = np.eye(2)
            return LOC_COEFF * grad

        val = np.zeros((2, 1))
        expr = Expr(f, grad)
        e = EqExpr(expr, val)

        super(InLane, self).__init__(name, e, attr_inds, params, expected_param_types, env=env, priority=2)
        self.spacial_anchor = False

class ExternalInLane(DrivingPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 3
        self.obj, self.road, self.lane_num = params
        attr_inds = OrderedDict([(self.obj, [("xy",    np.array([0,1], dtype=np.int)),
                                             ("theta", np.array([0], dtype=np.int))])])

        f = lambda x: LOC_COEFF * self.road.geom.to_lane(x[0], x[1], x[2], self.lane_num.value[0, 0])[0] if not self.road.geom.is_user else np.zeros((2,))
        def grad(x):
            grad = np.zeros((3, 2))
            grad[:2,:] = np.eye(2)
            return LOC_COEFF * grad

        val = np.zeros((2, 1))
        expr = Expr(f, grad)
        e = EqExpr(expr, val)

        super(ExternalInLane, self).__init__(name, e, attr_inds, params, expected_param_types, env=env, priority=2)
        self.spacial_anchor = False

class LeftOfLane(DrivingPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 3
        self.obj, self.road, self.lane_num = params
        attr_inds = OrderedDict([(self.obj, [("xy",    np.array([0,1], dtype=np.int)),
                                             ("theta", np.array([0], dtype=np.int))])])

        lane_n = self.lane_num.value[0,0]
        f = lambda x: LOC_COEFF * self.road.geom.to_lane(x[0], x[1], x[2], lane_n - 1)[0] if lane_n > 0 else LOC_COEFF * self.road.geom.to_lane(x[0], x[1], x[2], lane_n)[0]
        def grad(x):
            grad = np.zeros((3, 2))
            grad[:2,:] = np.eye(2)
            return LOC_COEFF * grad

        val = np.zeros((2, 1))
        expr = Expr(f, grad)
        e = EqExpr(expr, val)

        super(LeftOfLane, self).__init__(name, e, attr_inds, params, expected_param_types, env=env, priority=2)
        self.spacial_anchor = False

class RightOfLane(DrivingPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 3
        self.obj, self.road, self.lane_num = params
        attr_inds = OrderedDict([(self.obj, [("xy",    np.array([0,1], dtype=np.int)),
                                             ("theta", np.array([0], dtype=np.int))])])

        f = lambda x: LOC_COEFF * self.road.geom.to_lane(x[0], x[1], x[2], lane_n + 1)[0] if lane_n < self.road.geom.num_lanes - 1 else LOC_COEFF * self.road.geom.to_lane(x[0], x[1], x[2], lane_n)[0]
        def grad(x):
            grad = np.zeros((3, 2))
            grad[:2,:] = np.eye(2)
            return LOC_COEFF * grad

        val = np.zeros((2, 1))
        expr = Expr(f, grad)
        e = EqExpr(expr, val)

        super(LeftOfLane, self).__init__(name, e, attr_inds, params, expected_param_types, env=env, priority=2)
        self.spacial_anchor = False

class PoseInLane(InLane):
    def __init__(self, name, params, expected_param_types, env=None):
        pass

class PoseLeftOfLane(LeftOfLane):
    def __init__(self, name, params, expected_param_types, env=None):
        pass

class PoseRightOfLane(RightOfLane):
    def __init__(self, name, params, expected_param_types, env=None):
        pass

class XY_Limit(DrivingPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 3
        self.obj, self.xlimit, self.ylimit = params
        attr_inds = OrderedDict([(self.obj, [("xy", np.array([0, 1], dtype=np.int))]),
                                 (self.xlimit, [("value", np.array([0], dtype=np.int))]),
                                 (self.ylimit, [("value", np.array([0], dtype=np.int))])])

        A = np.zeros((4,4))
        A[:2,:2] = -np.eye(2)
        A[:2,2:4] = np.eye(2)
        A[2:4,:2] = -np.eye(2) 
        b, val = np.zeros((4, 1)), np.zeros((4, 1))
        e = LEqExpr(AffExpr(A, b), val)
        super(Stationary, self).__init__(name, e, attr_inds, params, expected_param_types, env=env, priority=-2)
        self.spacial_anchor = False

class Limit(DrivingPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 2
        self.obj, self.limit = params
        attr_inds = OrderedDict([(self.obj, [("v", np.array([0], dtype=np.int)),
                                             ("u1", np.array([0], dtype=np.int))]),
                                 (self.limit, [("value", np.array([0], dtype=np.int))])])

        b, val = np.zeros((1, 1)), np.zeros((1, 1))
        e = LEqExpr(AffExpr(self.A, b), val)
        super(Stationary, self).__init__(name, e, attr_inds, params, expected_param_types, env=env, priority=-2)
        self.spacial_anchor = False

class VelLowerLimit(Limit):
    def __init__(self, name, params, expected_param_types, env=None):
        self.A = np.zeros((1,3))
        self.A[0,0] = -1
        self.A[0,2] = 1
        super(VelLowerLimit, self).__init__(name, params, expected_param_types, env)
        self.spacial_anchor = False

class VelUpperLimit(Limit):
    def __init__(self, name, params, expected_param_types, env=None):
        self.A = np.zeros((1,3))
        self.A[0,0] = 1
        self.A[0,2] = -1
        super(VelUpperLimit, self).__init__(name, params, expected_param_types, env)
        self.spacial_anchor = False

class AccLowerLimit(Limit):
    def __init__(self, name, params, expected_param_types, env=None):
        self.A = np.zeros((1,3))
        self.A[0,1] = -1
        self.A[0,2] = 1
        super(AccLowerLimit, self).__init__(name, params, expected_param_types, env)
        self.spacial_anchor = False

class AccUpperLimit(Limit):
    def __init__(self, name, params, expected_param_types, env=None):
        self.A = np.zeros((1,3))
        self.A[0,1] = 1
        self.A[0,2] = -1
        super(AccUpperLimit, self).__init__(name, params, expected_param_types, env)
        self.spacial_anchor = False

class CollisionPredicate(DrivingPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 2
        self.obj1, self.obj2 = params
        attr_inds = OrderedDict([(self.obj1, [("xy", np.array([0,1], dtype=np.int)),
                                              ("theta", np.array([0], dtype=np.int))]),
                                 (self.obj2, [("xy", np.array([0,1], dtype=np.int)),
                                              ("theta", np.array([0], dtype=np.int))])])

        def f(x):
            old_pose1 = self.obj1.geom.update_xy_theta(x[0], x[1], x[2], 0)
            old_pose2 = self.obj2.geom.update_xy_theta(x[3], x[4], x[5], 0)
            obj1_pts = self.obj1.geom.get_points(0, COL_DIST)
            obj2_pts = self.obj2.geom.get_points(0, COL_DIST)
            self.obj1.geom.update_xy_theta(0, old_pose1[0], old_pose1[1], old_pose1[2])
            self.obj2.geom.update_xy_theta(0, old_pose2[0], old_pose2[1], old_pose2[2])
            return COLLISION_COEFF * collision_vector(obj1_pts, obj2_pts)
        
        def grad(obj1_body, obj2_body):
            grad = np.zeros((2,6))
            grad[:, :2] = -np.eye(2)
            grad[:, 3:5] = np.eye(2)
            return COLLISION_COEFF * grad

        val = np.zeros((2, 1))
        col_expr = Expr(f, grad)
        e = EqExpr(col_expr, val)

        super(CollisionPredicate, self).__init__(name, e, attr_inds, params, expected_param_types, env=env, priority=3)
        self.spacial_anchor = False

    def resample(self, negated, t, plan):
        if negated:
            return None, None
        else:
            attr_inds, res = OrderedDict(), OrderedDict()
            act_inds, action = [(i, act) for i, act in enumerate(plan.actions) if act.active_timesteps[0] < t and  t <= act.active_timesteps[1]][0]
            act_start, act_end = action.active_timesteps

            old_pose1 = self.obj1.geom.update_xy_theta(x[0], x[1], x[2], 0)
            old_pose2 = self.obj2.geom.update_xy_theta(x[3], x[4], x[5], 0)
            obj1_pts = self.obj1.geom.get_points(0, COL_DIST)
            obj2_pts = self.obj2.geom.get_points(0, COL_DIST)
            self.obj1.geom.update_xy_theta(0, old_pose1[0], old_pose1[1], old_pose1[2])
            self.obj2.geom.update_xy_theta(0, old_pose2[0], old_pose2[1], old_pose2[2])
            col_vec = collision_vector(obj1_pts, obj2_pts)

            random_dir = np.random.randint(-1, 2) * np.array([-1./col_vec[0], 1./col_vec[1]])
            target_xy = np.random.uniform(1, 3) * col_vec - self.obj1.xy[t] + np.random.uniform(1, 3) * random_dir / np.linalg.norm(random_dir)

            start_t = max(t - 5, act_start)
            start_xy = self.obj1.xy[:, start_t]
            start_theta = self.obj1.theta[0, start_t]
            end_theta = np.arccos((target_xy - start_xy).dot([1,0]) / np.linalg.norm(target_xy - start_xy))

            for i in range(start_t+1, t+1):
                t_ratio = float((i - start_t)) / (t - start_t)
                add_to_attr_inds_and_res(i, attr_inds, res, self.obj1, [('xy', t_ratio * target_xy + (1 - t_ratio) * start_xy), ('theta', t_ratio * end_theta + (1 - t_ratio) * start_theta)])

            return attr_inds, res

class VehicleVehicleCollision(CollisionPredicate):
    pass

class VehicleObstacleCollision(CollisionPredicate):
    pass

class VehicleCrateCollision(CollisionPredicate):
    pass

class CrateObstacleCollision(CollisionPredicate):
    pass

class PathCollisionPredicate(DrivingPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 2
        self.obj1, self.obj2 = params
        attr_inds = OrderedDict([(self.obj1, [("xy", np.array([0,1], dtype=np.int)),
                                              ("theta", np.array([0], dtype=np.int))]),
                                 (self.obj2, [("xy", np.array([0,1], dtype=np.int)),
                                              ("theta", np.array([0], dtype=np.int))])])

        def f(x):
            old_0_pose1 = self.obj1.geom.update_xy_theta(x[0], x[1], x[2], 0)
            old_0_pose2 = self.obj2.geom.update_xy_theta(x[3], x[4], x[5], 0)
            old_1_pose1 = self.obj1.geom.update_xy_theta(x[6], x[7], x[8], 1)
            old_1_pose2 = self.obj2.geom.update_xy_theta(x[9], x[10], x[11], 1)
            obj1_pts = self.obj1.geom.get_points(0, COL_DIST) + self.obj1.geom.get_points(1, COL_DIST)
            obj2_pts = self.obj2.geom.get_points(0, COL_DIST) + self.obj2.geom.get_points(1, COL_DIST)
            self.obj1.geom.update_xy_theta(0, old_0_pose1[0], old_0_pose1[1], old_0_pose1[2])
            self.obj2.geom.update_xy_theta(0, old_0_pose2[0], old_0_pose2[1], old_0_pose2[2])
            self.obj1.geom.update_xy_theta(1, old_1_pose1[0], old_1_pose1[1], old_1_pose1[2])
            self.obj2.geom.update_xy_theta(1, old_1_pose2[0], old_1_pose2[1], old_1_pose2[2])
            return COLLISION_COEFF * collision_vector(obj1_pts, obj2_pts)
        
        def grad(obj1_body, obj2_body):
            grad = np.zeros((2,12))
            grad[:, :2] = -np.eye(2)
            grad[:, 3:5] = np.eye(2)
            grad[:, 5:8] = -np.eye(2)
            grad[:, 8:11] = np.eye(2)
            return COLLISION_COEFF * grad

        val = np.zeros((2, 1))
        col_expr = Expr(f, grad)
        e = EqExpr(col_expr, val)

        super(CollisionPredicate, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), env=env, priority=3)
        self.spacial_anchor = False

    def resample(self, negated, t, plan):
        if negated:
            return None, None
        else:
            attr_inds, res = OrderedDict(), OrderedDict()
            act_inds, action = [(i, act) for i, act in enumerate(plan.actions) if act.active_timesteps[0] < t and  t <= act.active_timesteps[1]][-1]
            act_start, act_end = action.active_timesteps

            old_0_pose1 = self.obj1.geom.update_xy_theta(x[0], x[1], x[2], 0)
            old_0_pose2 = self.obj2.geom.update_xy_theta(x[3], x[4], x[5], 0)
            old_1_pose1 = self.obj1.geom.update_xy_theta(x[6], x[7], x[8], 1)
            old_1_pose2 = self.obj2.geom.update_xy_theta(x[9], x[10], x[11], 1)
            obj1_pts = self.obj1.geom.get_points(0, COL_DIST) + self.obj1.geom.get_points(1, COL_DIST)
            obj2_pts = self.obj2.geom.get_points(0, COL_DIST) + self.obj2.geom.get_points(1, COL_DIST)
            self.obj1.geom.update_xy_theta(0, old_0_pose1[0], old_0_pose1[1], old_0_pose1[2])
            self.obj2.geom.update_xy_theta(0, old_0_pose2[0], old_0_pose2[1], old_0_pose2[2])
            self.obj1.geom.update_xy_theta(1, old_1_pose1[0], old_1_pose1[1], old_1_pose1[2])
            self.obj2.geom.update_xy_theta(1, old_1_pose2[0], old_1_pose2[1], old_1_pose2[2])
            col_vec = collision_vector(obj1_pts, obj2_pts)

            random_dir = np.random.randint(-1, 2) * np.array([-1./col_vec[0], 1./col_vec[1]])
            target_xy = np.random.uniform(1, 3) * col_vec - self.obj1.xy[t] + np.random.uniform(1, 3) * random_dir / np.linalg.norm(random_dir)

            start_t = max(t - 5, act_start)
            start_xy = self.obj1.xy[:, start_t]
            start_theta = self.obj1.theta[0, start_t]
            target_theta = np.arccos((target_xy - start_xy).dot([1,0]) / np.linalg.norm(target_xy - start_xy))

            for i in range(start_t+1, t+1):
                t_ratio = float((i - start_t)) / (t - start_t)
                add_to_attr_inds_and_res(i, attr_inds, res, self.obj1, [('xy', t_ratio * target_xy + (1 - t_ratio) * start_xy), ('theta', t_ratio * end_theta + (1 - t_ratio) * start_theta)])

            add_to_attr_inds_and_res(t+1, attr_inds, res, self.obj1, [('xy', target_xy), ('theta', target_theta)])

            return attr_inds, res


class VehicleVehiclePathCollision(PathCollisionPredicate):
    pass

class VehicleObstaclePathCollision(PathCollisionPredicate):
    pass

class VehicleCratePathCollision(PathCollisionPredicate):
    pass

class CrateObstaclePathCollision(CollisionPredicate):
    pass

class Follow(DrivingPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 3
        self.v1, self.v2, self.dist = params
        attr_inds = OrderedDict([(self.v1, [("xy", np.array([0,1], dtype=np.int)),
                                            ("theta", np.array([0], dtype=np.int))]),
                                 (self.v2, [("xy", np.array([0,1], dtype=np.int)),
                                            ("theta", np.array([0], dtype=np.int))]),
                                 (self.dist, [("value", np.array([0], dtype=np.int))])])
        
        def f(x):
            old_v1_pose = self.v1.geom.update_xy_theta(x[0], x[1], x[5], 0)
            front_x, front_y = self.v1.geom.vehicle_front()

            target_x = x[3] - np.cos(x[5]) * x[6]
            target_y =x[4] - np.sin(x[5]) * x[6]

            x_delta = target_x - x[0]
            y_delta = target_y - x[1]

            theta_delta = x[5] - x[2]
            while theta_delta > np.pi:
                theta_delta -= 2 * np.pi

            while theta_delta < np.pi:
                theta_delta += 2 * np.pi

            return FOLLOW_COEFF * np.r_[x_delta, y_delta, theta_delta].reshape((3,1))

        def grad(x):
            return FOLLOW_COEFF * np.c_[np.eye(3), np.zeros((3,3))]

        val = np.zeros((3, 1))
        e = EqExpr(Expr(f, grad), val)
        super(Stationary, self).__init__(name, e, attr_inds, params, expected_param_types, env=env, priority=3)
        self.spacial_anchor = False

class StopAtStopSign(DrivingPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 2
        self.obj, self.sign = params
        attr_inds = OrderedDict([(self.obj, [("xy", np.array([0,1], dtype=np.int))])])

        def f(x):
            if not self.sign.geom.road.is_on(x[0], x[1]):
                return np.zeros((2,))

            direction = self.sign.geom.road.direction
            rot_mat = np.array([[np.cos(direction), -np.sin(direction)], 
                                [np.sin(direction), np.cos(direction)]])
            dist_vec = self.sign.geom.loc - x[:2]
            rot_dist_vec = rot_mat.dot(dist_vec)

            if np.abs(rot_dist_vec[0]) < self.sign.geom.length / 2. and np.abs(rot_dist_vec[1]) < self.sign.geom.width / 2.:
                return STOP_COEFF * (x[2:] - x[:2])

            return np.zeros((2,))

        def grad(x):
            return STOP_COEFF * np.c_[np.eye(2), -np.eye(2)]

        val = np.zeros((2, 1))
        e = EqExpr(Expr(f, grad), val)
        super(StopAtStopSign, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), env=env, priority=2)
        self.spacial_anchor = False

class ExternalDriveDownRoad(DrivingPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 1
        self.obj, = params
        attr_inds = OrderedDict([(self.obj, [("xy", np.array([0,1], dtype=np.int))])])

        if self.obj.geom.road:
            direction = self.obj.geom.road.direction
            rot_mat = np.array([[np.cos(direction), -np.sin(direction)], 
                                [np.sin(direction), np.cos(direction)]])
            self.dir_vec = rot_mat.dot([1,0])
        else:
            self.dir_vec = np.zeros((2,))

        def f(x):
            if not self.obj.geom.road:
                return np.zeros((2,))

            dist_vec = x[2:4] - x[:2]
            return DOWN_ROAD_COEFF * ((dist_vec / np.linalg.norm(dist_vec)) - self.dir_vec)

        def grad(x):
            return DOWN_ROAD_COEFF * np.c_[np.eye(2), -np.eye(2)]

        val = np.zeros((2, 1))
        e = EqExpr(Expr(f, grad), val)
        super(ExternalDriveDownRoad, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), env=env, priority=2)
        self.spacial_anchor = False

class WithinDistance(DrivingPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        assert len(params) == 3
        self.target1, self.target2, self.dist = params
        attr_inds = OrderedDict([(self.target1, [("xy", np.array([0,1], dtype=np.int))]),
                                 (self.target2, [("xy", np.array([0,1], dtype=np.int))]),
                                 (self.dist,    [("value", np.array([0], dtype=np.int))])])

        def f(x):
            scaled_vec = np.abs((x[2:4] - x[:2]) / np.linalg.norm(x[2:4] - x[:2]) * x[4])
            if np.all(scaled_vec < x[2:4] - x[:2]):
                return DIST_COEFF * (-x[2:4] + x[:2] + scaled_vec)
            elif np.all(-scaled_vec > x[2:4] - x[:2]):
                return DIST_COEFF * (-scaled_vec - x[2:4] + x[:2])
            else:
                return np.zeros((2,))

        def grad(x):
            return DIST_COEFF * np.c_[-np.eye(2), np.eye(2), np.zeros((1,2))]

        val = np.zeros((2, 1))
        dynamics_expr = Expr(self.f, self.grad)
        e = LEqExpr(dynamics_expr, val)

        super(WithinDistance, self).__init__(name, e, attr_inds, params, expected_param_types, env=env, priority=1)
        self.spacial_anchor = False

class PosesWithDistance(WithinDistance):
    pass
