from core.util_classes import robot_predicates
from errors_exceptions import PredicateException
from core.util_classes.openrave_body import OpenRAVEBody
from sco.expr import Expr, AffExpr, EqExpr, LEqExpr
from collections import OrderedDict
from openravepy import DOFAffine
import numpy as np

BASE_DIM = 1
JOINT_DIM = 16
ROBOT_ATTR_DIM = 17

BASE_MOVE = 1
JOINT_MOVE_FACTOR = 10
TWOARMDIM = 16
# EEReachable Constants
APPROACH_DIST = 0.05
RETREAT_DIST = 0.075
EEREACHABLE_STEPS = 3
# Collision Constants
DIST_SAFE = 1e-2
COLLISION_TOL = 1e-3
#Plan Coefficient
IN_GRIPPER_COEFF = 1.
EEREACHABLE_COEFF = 1e0
EEREACHABLE_OPT_COEFF = 1e3
EEREACHABLE_ROT_OPT_COEFF = 3e2
INGRIPPER_OPT_COEFF = 3e2
RCOLLIDES_OPT_COEFF = 1e2
OBSTRUCTS_OPT_COEFF = 1e2
GRASP_VALID_COEFF = 1e1
GRIPPER_OPEN_VALUE = 0.2
GRIPPER_CLOSE_VALUE = 0.

# Attribute map used in baxter domain. (Tuple to avoid changes to the attr_inds)
ATTRMAP = {"Robot": (("lArmPose", np.array(range(7), dtype=np.int)),
                     ("lGripper", np.array([0], dtype=np.int)),
                     ("rArmPose", np.array(range(7), dtype=np.int)),
                     ("rGripper", np.array([0], dtype=np.int)),
                     ("pose", np.array([0], dtype=np.int))),
           "RobotPose": (("lArmPose", np.array(range(7), dtype=np.int)),
                         ("lGripper", np.array([0], dtype=np.int)),
                         ("rArmPose", np.array(range(7), dtype=np.int)),
                         ("rGripper", np.array([0], dtype=np.int)),
                         ("value", np.array([0], dtype=np.int))),
           "Can": (("pose", np.array([0,1,2], dtype=np.int)),
                   ("rotation", np.array([0,1,2], dtype=np.int))),
           "EEPose": (("value", np.array([0,1,2], dtype=np.int)),
                      ("rotation", np.array([0,1,2], dtype=np.int))),
           "Target": (("value", np.array([0,1,2], dtype=np.int)),
                      ("rotation", np.array([0,1,2], dtype=np.int))),
           "Table": (("pose", np.array([0,1,2], dtype=np.int)),
                     ("rotation", np.array([0,1,2], dtype=np.int))),
           "Obstacle": (("pose", np.array([0,1,2], dtype=np.int)),
                        ("rotation", np.array([0,1,2], dtype=np.int)))
          }

class BaxterAt(robot_predicates.At):
    pass

class BaxterRobotAt(robot_predicates.RobotAt):

    # RobotAt, Robot, RobotPose

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_dim = 17
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type])),
                                 (params[1], list(ATTRMAP[params[1]._type]))])
        super(BaxterRobotAt, self).__init__(name, params, expected_param_types, env)

class BaxterIsMP(robot_predicates.IsMP):

    # IsMP Robot (Just the Robot Base)

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type]))])
        super(BaxterIsMP, self).__init__(name, params, expected_param_types, env, debug)

    def setup_mov_limit_check(self):
        # Get upper joint limit and lower joint limit
        robot_body = self._param_to_body[self.robot]
        robot = robot_body.env_body
        robot_body._set_active_dof_inds(list(range(2,2+JOINT_DIM)))
        dof_inds = robot.GetActiveDOFIndices()
        lb_limit, ub_limit = robot.GetDOFLimits()
        active_ub = ub_limit[dof_inds].reshape((JOINT_DIM,1))
        active_lb = lb_limit[dof_inds].reshape((JOINT_DIM,1))
        joint_move = (active_ub-active_lb)/JOINT_MOVE_FACTOR
        # Setup the Equation so that: Ax+b < val represents
        # |base_pose_next - base_pose| <= BASE_MOVE
        # |joint_next - joint| <= joint_movement_range/JOINT_MOVE_FACTOR
        val = np.vstack((joint_move, BASE_MOVE*np.ones((BASE_DIM, 1)), joint_move, BASE_MOVE*np.ones((BASE_DIM, 1))))
        A = np.eye(2*ROBOT_ATTR_DIM) - np.eye(2*ROBOT_ATTR_DIM, k=ROBOT_ATTR_DIM) - np.eye(2*ROBOT_ATTR_DIM, k=-ROBOT_ATTR_DIM)
        b = np.zeros((2*ROBOT_ATTR_DIM,1))
        robot_body._set_active_dof_inds(range(18))

        # Setting attributes for testing
        self.base_step = BASE_MOVE*np.ones((BASE_DIM, 1))
        self.joint_step = joint_move
        self.lower_limit = active_lb
        return A, b, val

class BaxterWithinJointLimit(robot_predicates.WithinJointLimit):

    # WithinJointLimit Robot

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type][:-1]))])
        super(BaxterWithinJointLimit, self).__init__(name, params, expected_param_types, env, debug)

    def setup_mov_limit_check(self):
        # Get upper joint limit and lower joint limit
        robot_body = self._param_to_body[self.robot]
        robot = robot_body.env_body
        robot_body._set_active_dof_inds(list(range(2,2+JOINT_DIM)))
        dof_inds = robot.GetActiveDOFIndices()
        lb_limit, ub_limit = robot.GetDOFLimits()
        active_ub = ub_limit[dof_inds].reshape((JOINT_DIM,1))
        active_lb = lb_limit[dof_inds].reshape((JOINT_DIM,1))
        # Setup the Equation so that: Ax+b < val represents
        # lb_limit <= pose <= ub_limit
        val = np.vstack((-active_lb, active_ub))
        A_lb_limit = -np.eye(JOINT_DIM)
        A_up_limit = np.eye(JOINT_DIM)
        A = np.vstack((A_lb_limit, A_up_limit))
        b = np.zeros((2*JOINT_DIM,1))
        robot_body._set_active_dof_inds(range(18))

        joint_move = (active_ub-active_lb)/JOINT_MOVE_FACTOR
        self.base_step = BASE_MOVE*np.ones((BASE_DIM,1))
        self.joint_step = joint_move
        self.lower_limit = active_lb
        return A, b, val

class BaxterStationary(robot_predicates.Stationary):
    pass

class BaxterStationaryBase(robot_predicates.StationaryBase):

    # StationaryBase, Robot (Only Robot Base)

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][-1]])])
        self.attr_dim = BASEDIM
        super(BaxterStationaryBase, self).__init__(self, name, params, expected_param_types, env)

class BaxterStationaryArms(robot_predicates.StationaryArms):

    # StationaryArms, Robot (Only Robot Arms)

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type][:-1]))])
        self.attr_dim = TWOARMDIM
        super(BaxterStationaryArms, self).__init__(self, name, params, expected_param_types, env)

class BaxterStationaryW(robot_predicates.StationaryW):
    pass

class BaxterStationaryNEq(robot_predicates.StationaryNEq):
    pass

class BaxterGraspValid(robot_predicates.GraspValid):
    pass

class BaxterGraspValidPos(BaxterGraspValid):

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][0]]),(params[1], [ATTRMAP[params[1]._type][0]])])
        self.attr_dim = 3
        super(BaxterGraspValidPos, self).__init__(name, params, expected_param_types, env, debug)

class BaxterGraspValidRot(BaxterGraspValid):

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][1]]),(params[1], [ATTRMAP[params[1]._type][1]])])
        self.attr_dim = 3
        super(BaxterGraspValidRot, self).__init__(name, params, expected_param_types, env, debug)

class BaxterInContact(robot_predicates.InContact):

    # InContact robot EEPose target

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        # Define constants
        self.GRIPPER_CLOSE = GRIPPER_CLOSE_VALUE
        self.GRIPPER_OPEN = GRIPPER_OPEN_VALUE
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][3]])])
        super(BaxterInContact, self).__init__(name, params, expected_param_types, env, debug)

class BaxterInGripper(robot_predicates.InGripper):

    # InGripper, Robot, Can

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type])),
                                 (params[1], list(ATTRMAP[params[1]._type]))])
        super(BaxterInGripper, self).__init__(name, params, expected_param_types, env, debug)

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        l_arm_pose, l_gripper = x[0:7], x[7]
        r_arm_pose, r_gripper = x[8:15], x[15]
        base_pose = x[16]
        body = self._param_to_body[self.robot].env_body
        dof = body.GetActiveDOFValues()
        dof[0] = base_pose
        dof[2:9], dof[9] = l_arm_pose.reshape((7,)), l_gripper
        dof[10:17], dof[17] = r_arm_pose.reshape((7,)), r_gripper
        body.SetActiveDOFValues(dof)

    def get_robot_info(self, robot_body):
        # Provide functionality of Obtaining Robot information
        tool_link = robot_body.env_body.GetLink("right_gripper")
        manip_trans = tool_link.GetTransform()
        # This manip_trans is off by 90 degree
        pose = OpenRAVEBody.obj_pose_from_transform(manip_trans)
        robot_trans = OpenRAVEBody.get_ik_transform(pose[:3], pose[3:])
        arm_inds = list(range(10,17))
        return robot_trans, arm_inds

class BaxterInGripperPos(BaxterInGripper):

    # InGripper, Robot, Can

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        # Sets up constants
        self.IN_GRIPPER_COEFF = IN_GRIPPER_COEFF
        self.INGRIPPER_OPT_COEFF = INGRIPPER_OPT_COEFF
        self.eval_f = lambda x: self.pos_check(x)[0]
        self.eval_grad = lambda x: self.pos_check(x)[1]
        super(BaxterInGripperPos, self).__init__(name, params, expected_param_types, env, debug)


class BaxterInGripperRot(BaxterInGripper):

    # InGripper, Robot, Can

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        # Sets up constants
        self.IN_GRIPPER_COEFF = IN_GRIPPER_COEFF
        self.INGRIPPER_OPT_COEFF = INGRIPPER_OPT_COEFF
        self.eval_f = lambda x: self.rot_check(x)[0]
        self.eval_grad = lambda x: self.rot_check(x)[1]
        super(BaxterInGripperRot, self).__init__(name, params, expected_param_types, env, debug)


class BaxterEEReachable(robot_predicates.EEReachable):

    # EEreachable Robot, StartPose, EEPose

    def __init__(self, name, params, expected_param_types, env=None, debug=False, steps=EEREACHABLE_STEPS):
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type])),
                                 (params[2], list(ATTRMAP[params[2]._type]))])
        super(BaxterEEReachable, self).__init__(name, params, expected_param_types, env, debug, steps)

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        l_arm_pose, l_gripper = x[0:7], x[7]
        r_arm_pose, r_gripper = x[8:15], x[15]
        base_pose = x[16]
        body = self._param_to_body[self.robot].env_body
        dof = body.GetActiveDOFValues()
        dof[0] = base_pose
        dof[2:9], dof[9] = l_arm_pose.reshape((7,)), l_gripper
        dof[10:17], dof[17] = r_arm_pose.reshape((7,)), r_gripper
        body.SetActiveDOFValues(dof)

    def get_robot_info(self, robot_body):
        # Provide functionality of Obtaining Robot information
        tool_link = robot_body.env_body.GetLink("right_gripper")
        manip_trans = tool_link.GetTransform()
        # This manip_trans is off by 90 degree
        pose = OpenRAVEBody.obj_pose_from_transform(manip_trans)
        robot_trans = OpenRAVEBody.get_ik_transform(pose[:3], pose[3:])
        arm_inds = list(range(10,17))
        return robot_trans, arm_inds

    def get_rel_pt(self, rel_step):
        if rel_step <= 0:
            return rel_step*np.array([APPROACH_DIST, 0, 0])
        else:
            return rel_step*np.array([0, 0, RETREAT_DIST])

    def stacked_f(self, x):
        i = 0
        f_res = []
        start, end = self.active_range
        for s in range(start, end+1):
            rel_pt = self.get_rel_pt(s)
            f_res.append(self.ee_pose_check_rel_obj(x[i:i+self.attr_dim], rel_pt)[0])
            i += self.attr_dim
        return np.vstack(tuple(f_res))


    def stacked_grad(self, x):
        f_grad = []
        start, end = self.active_range
        t = (2*self._steps+1)
        k = 3

        grad = np.zeros((k*t, self._dim*t))
        i = 0
        j = 0
        for s in range(start, end+1):
            rel_pt = self.get_rel_pt(s)
            grad[j:j+k, i:i+self._dim] = self.ee_pose_check_rel_obj(x[i:i+self._dim], rel_pt)[1]
            i += self._dim
            j += k
        return grad

class BaxterEEReachablePos(BaxterEEReachable):

    # EEUnreachable Robot, StartPose, EEPose

    def __init__(self, name, params, expected_param_types, env=None, debug=False, steps=EEREACHABLE_STEPS):
        self.EEREACHABLE_COEFF = 1
        self.EEREACHABLE_OPT_COEFF = 1
        self.eval_f = self.stacked_f
        self.eval_grad = self.stacked_grad
        self.attr_dim = 26
        super(BaxterEEReachablePos, self).__init__(name, params, expected_param_types, env, debug, steps)

class BaxterEEReachableRot(BaxterEEReachable):

    # EEUnreachable Robot, StartPose, EEPose

    def __init__(self, name, params, expected_param_types, env=None, debug=False, steps=EEREACHABLE_STEPS):
        self.EEREACHABLE_COEFF = EEREACHABLE_COEFF
        self.EEREACHABLE_OPT_COEFF = EEREACHABLE_ROT_OPT_COEFF
        self.check_f = lambda x: self.ee_rot_check[0]
        self.check_grad = lambda x: self.ee_rot_check[1]
        super(BaxterEEReachableRot, self).__init__(name, params, expected_param_types, env, debug, steps)

class BaxterObstructs(robot_predicates.Obstructs):

    # Obstructs, Robot, RobotPose, RobotPose, Can

    def __init__(self, name, params, expected_param_types, env=None, debug=False, tol=COLLISION_TOL):
        self.attr_dim = 17
        self.dof_cache = None
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type])),
                                 (params[3], list(ATTRMAP[params[3]._type]))])
        super(BaxterObstructs, self).__init__(name, params, expected_param_types, env, debug, tol)
    
    def set_active_dof_inds(self, robot_body, reset = False):
        robot = robot_body.env_body
        if reset == True and self.dof_cache != None:
            robot.SetActiveDOFs(self.dof_cache)
            self.dof_cache = None
        elif reset == False and self.dof_cache == None:
            self.dof_cache = robot.GetActiveDOFIndices()
            robot.SetActiveDOFs(list(range(2,18))+ [0])
        else:
            raise PredicateException("Incorrect Active DOF Setting")

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        l_arm_pose, l_gripper = x[0:7], x[7]
        r_arm_pose, r_gripper = x[8:15], x[15]
        base_pose = x[16]
        body = self._param_to_body[self.robot].env_body
        dof = body.GetActiveDOFValues()
        dof[0] = base_pose
        dof[2:9], dof[9] = l_arm_pose.reshape((7,)), l_gripper
        dof[10:17], dof[17] = r_arm_pose.reshape((7,)), r_gripper
        body.SetActiveDOFValues(dof)

class BaxterObstructsHolding(robot_predicates.ObstructsHolding):

    # ObstructsHolding, Robot, RobotPose, RobotPose, Can, Can

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_dim = 17
        self.dof_cache = None
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type])),
                                 (params[3], list(ATTRMAP[params[3]._type])),
                                 (params[4], list(ATTRMAP[params[4]._type]))])
        self.OBSTRUCTS_OPT_COEFF = OBSTRUCTS_OPT_COEFF
        super(BaxterObstructsHolding, self).__init__(name, params, expected_param_types, env, debug)

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        l_arm_pose, l_gripper = x[0:7], x[7]
        r_arm_pose, r_gripper = x[8:15], x[15]
        base_pose = x[16]
        body = self._param_to_body[self.robot].env_body
        dof = body.GetActiveDOFValues()
        dof[0] = base_pose
        dof[2:9], dof[9] = l_arm_pose.reshape((7,)), l_gripper
        dof[10:17], dof[17] = r_arm_pose.reshape((7,)), r_gripper
        body.SetActiveDOFValues(dof)
    
    def set_active_dof_inds(self, robot_body, reset = False):
        robot = robot_body.env_body
        if reset == True and self.dof_cache != None:
            robot.SetActiveDOFs(self.dof_cache)
            self.dof_cache = None
        elif reset == False and self.dof_cache == None:
            self.dof_cache = robot.GetActiveDOFIndices()
            robot.SetActiveDOFs(list(range(2,18))+ [0])
        else:
            raise PredicateException("Incorrect Active DOF Setting")

class BaxterCollides(robot_predicates.Collides):
    pass

class BaxterRCollides(robot_predicates.RCollides):

    # RCollides Robot Obstacle

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_dim = 17
        self.dof_cache = None
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type])),
                                 (params[1], list(ATTRMAP[params[1]._type]))])
        self.RCOLLIDES_OPT_COEFF = RCOLLIDES_OPT_COEFF
        super(BaxterRCollides, self).__init__(name, params, expected_param_types, env, debug)

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        l_arm_pose, l_gripper = x[0:7], x[7]
        r_arm_pose, r_gripper = x[8:15], x[15]
        base_pose = x[16]
        body = self._param_to_body[self.robot].env_body
        dof = body.GetActiveDOFValues()
        dof[0] = base_pose
        dof[2:9], dof[9] = l_arm_pose.reshape((7,)), l_gripper
        dof[10:17], dof[17] = r_arm_pose.reshape((7,)), r_gripper
        body.SetActiveDOFValues(dof)

    def set_active_dof_inds(self, robot_body, reset = False):
        robot = robot_body.env_body
        if reset == True and self.dof_cache != None:
            robot.SetActiveDOFs(self.dof_cache)
            self.dof_cache = None
        elif reset == False and self.dof_cache == None:
            self.dof_cache = robot.GetActiveDOFIndices()
            robot.SetActiveDOFs(list(range(2,18))+ [0])
        else:
            raise PredicateException("Incorrect Active DOF Setting")

