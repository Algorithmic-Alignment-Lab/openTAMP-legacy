from IPython import embed as shell
import itertools
import numpy as np
import random

# SEED = 1234
NUM_PROBS = 1
filename = "laundry_probs/moveholding_basket_isolation.prob"
GOAL = "(BaxterRobotAt baxter robot_end_pose), (BaxterBasketInGripper baxter basket)"


# init Baxter pose
BAXTER_INIT_POSE = [0]
R_ARM_INIT = [-0.1       , -1.19867189,  0.46799631,  1.25141751, -0.16478213, 1.55519343, -0.44274586]
L_ARM_INIT = [-0.9       , -1.37465729, -0.37976225,  1.47880368,  0.07260239, 1.48054184, -0.49429886]
INT_GRIPPER = [0.02]

BAXTER_END_POSE = [0]
R_ARM_END = [ 0.9       , -1.31104393,  0.48269435,  1.40766811, -0.11977619, 1.5032262 ,  0.59152858]
L_ARM_END = [-0.2       , -1.16795162,  0.065334  ,  1.16072076, -0.02559873, 1.57879428,  0.64531233]

END_GRIPPER = [0.015]
# init basket pose
BASKET_INIT_POS = [0.65 , -0.283,  1]
# BASKET_INIT_POS = [0.65 , 0.323,  1]
BASKET_INIT_ROT = [np.pi/2, 0, np.pi/2]

# end basket pose
BASKET_END_POS = [0.65, 0.323, 0.81]
BASKET_END_ROT = [np.pi/2, 0, np.pi/2]


ROBOT_DIST_FROM_TABLE = 0.05
TABLE_GEOM = [0.3, 0.6, 0.018]
TABLE_POS = [0.75, 0.02, 0.522]
TABLE_ROT = [0,0,0]

# WASHER_POS = [2,2,2]
WASHER_POS = [0.08, 0.781, 0.28]
WASHER_ROT = [np.pi, 0, np.pi/2]
WASHER_DOOR = [0.0]
WASHER_END_DOOR = [-np.pi/2]
WASHER_CONFIG = [True, True]

CLOTH_INIT_POS = [2,2,2]
CLOTH_INIT_ROT = [0,0,0]

CLOTH_INIT_POS_1 = [0.65, 0.401, 0.557]
CLOTH_INIT_ROT_1 = [0,0,0]

CLOTH_END_POS_1 = [0.65, -0.283,0.626]
CLOTH_END_ROT_1 = [0,0,0]


"""
Intermediate Poses adding it simplifies the plan
"""
GRASP_POSE = [-np.pi/8]
GRASP_LARMPOSE = [-0.2       , -1.61656414, -0.61176606,  1.93732774, -0.02776806, 1.24185857, -0.40960045]
GRASP_RARMPOSE = [ 0.7       , -0.96198717,  0.03612888,  0.99775438, -0.02067175, 1.5353429 , -0.44772444]
GRASP_GRIPPER = [0.02]

PUTDOWN_POSE = [np.pi/8]
PUTDOWN_LARMPOSE = [-0.8       , -0.87594019,  0.2587353 ,  0.92223949,  2.97696004, -1.54149409, -2.5580562 ]
PUTDOWN_RARMPOSE = [-0.2       , -1.38881187,  1.25178981,  1.81230334, -0.18056559, 1.27622517,  0.70704811]
PUTDOWN_GRIPPER = [0.02]

WASHER_BEGIN_POSE = [np.pi/3]
WASHER_BEGIN_LARMPOSE = [-0.8       , -0.93703369, -0.27464748,  1.09904023, -2.97863535, -1.4287909 ,  2.35686368]
WASHER_BEGIN_RARMPOSE = [-0.2       , -1.38881187,  1.25178981,  1.81230334, -0.18056559, 1.27622517,  0.70704811]

CLOTH_PUTDOWN_BEGIN_1_POSE = [0]
CLOTH_PUTDOWN_BEGIN_1_LARMPOSE = [-1.2, 0.30161054, -2.28704166, 0.95204077, 2.26996069, 1.91600073, -1.12607844]
CLOTH_PUTDOWN_BEGIN_1_RARMPOSE = [0, -0.785, 0, 0, 0, 0, 0]

WASHER_EE_POS = [0.29 ,  0.781,  0.785]
WASHER_EE_ROT = [0, np.pi/2, -np.pi/2]

WASHER_END_EE_POS = [-0.305,  0.781,  1.17 ]
WASHER_END_EE_ROT = [0,  0,  -np.pi/2]

def get_baxter_str(name, LArm = L_ARM_INIT, RArm = R_ARM_INIT, G = INT_GRIPPER, Pos = BAXTER_INIT_POSE):
    s = ""
    s += "(geom {})".format(name)
    s += "(lArmPose {} {}), ".format(name, LArm)
    s += "(lGripper {} {}), ".format(name, G)
    s += "(rArmPose {} {}), ".format(name, RArm)
    s += "(rGripper {} {}), ".format(name, G)
    s += "(pose {} {}), ".format(name, Pos)
    return s

def get_robot_pose_str(name, LArm = L_ARM_INIT, RArm = R_ARM_INIT, G = INT_GRIPPER, Pos = BAXTER_INIT_POSE):
    s = ""
    s += "(lArmPose {} {}), ".format(name, LArm)
    s += "(lGripper {} {}), ".format(name, G)
    s += "(rArmPose {} {}), ".format(name, RArm)
    s += "(rGripper {} {}), ".format(name, G)
    s += "(value {} {}), ".format(name, Pos)
    return s

def get_undefined_robot_pose_str(name):
    s = ""
    s += "(lArmPose {} undefined), ".format(name)
    s += "(lGripper {} undefined), ".format(name)
    s += "(rArmPose {} undefined), ".format(name)
    s += "(rGripper {} undefined), ".format(name)
    s += "(value {} undefined), ".format(name)
    return s

def get_underfine_symbol(name):
    s = ""
    s += "(value {} undefined), ".format(name)
    s += "(rotation {} undefined), ".format(name)
    return s

def get_underfine_washer_pose(name):
    s = ""
    s += "(value {} undefined), ".format(name)
    s += "(rotation {} undefined), ".format(name)
    s += "(door {} undefined), ".format(name)
    return s

def main():
    for iteration in range(NUM_PROBS):
        s = "# AUTOGENERATED. DO NOT EDIT.\n# Configuration file for CAN problem instance. Blank lines and lines beginning with # are filtered out.\n\n"

        s += "# The values after each attribute name are the values that get passed into the __init__ method for that attribute's class defined in the domain configuration.\n"
        s += "Objects: "
        s += "Basket (name {}); ".format("basket")
        s += "BasketTarget (name {}); ".format("init_target")
        s += "BasketTarget (name {}); ".format("end_target")

        s += "Robot (name {}); ".format("baxter")
        s += "EEPose (name {}); ".format("cg_ee_1")
        s += "EEPose (name {}); ".format("cp_ee_1")
        s += "EEPose (name {}); ".format("bg_ee_left")
        s += "EEPose (name {}); ".format("bg_ee_right")
        s += "EEPose (name {}); ".format("bp_ee_left")
        s += "EEPose (name {}); ".format("bp_ee_right")
        s += "EEPose (name {}); ".format("open_door_ee")
        s += "EEPose (name {}); ".format("cg_ee_2")
        s += "EEPose (name {}); ".format("cp_ee_2")
        s += "EEPose (name {}); ".format("close_door_ee")
        s += "RobotPose (name {}); ".format("robot_init_pose")
        s += "RobotPose (name {}); ".format("cloth_grasp_begin_1")
        s += "RobotPose (name {}); ".format("cloth_grasp_end_1")
        s += "RobotPose (name {}); ".format("cloth_putdown_begin_1")
        s += "RobotPose (name {}); ".format("cloth_putdown_end_1")
        s += "RobotPose (name {}); ".format("basket_grasp_begin")
        s += "RobotPose (name {}); ".format("basket_grasp_end")
        s += "RobotPose (name {}); ".format("basket_putdown_begin")
        s += "RobotPose (name {}); ".format("basket_putdown_end")
        s += "RobotPose (name {}); ".format("open_door_begin")
        s += "RobotPose (name {}); ".format("open_door_end")
        s += "RobotPose (name {}); ".format("cloth_grasp_begin_2")
        s += "RobotPose (name {}); ".format("cloth_grasp_end_2")
        s += "RobotPose (name {}); ".format("cloth_putdown_begin_2")
        s += "RobotPose (name {}); ".format("cloth_putdown_end_2")
        s += "RobotPose (name {}); ".format("close_door_begin")
        s += "RobotPose (name {}); ".format("close_door_end")
        s += "RobotPose (name {}); ".format("robot_end_pose")
        s += "Washer (name {}); ".format("washer")
        s += "WasherPose (name {}); ".format("washer_init_pose")
        s += "WasherPose (name {}); ".format("washer_end_pose")
        s += "Obstacle (name {}); ".format("table")
        s += "Cloth (name {}); ".format("cloth")
        s += "ClothTarget (name {}); ".format("cloth_target_begin_1")
        s += "ClothTarget (name {}); ".format("cloth_target_end_1")
        s += "ClothTarget (name {}); ".format("cloth_target_begin_2")
        s += "ClothTarget (name {}) \n\n".format("cloth_target_end_2")

        s += "Init: "
        s += "(geom basket), "
        s += "(pose basket {}), ".format(BASKET_INIT_POS)
        s += "(rotation basket {}), ".format(BASKET_INIT_ROT)

        s += "(geom init_target)"
        s += "(value init_target {}), ".format(BASKET_INIT_POS)
        s += "(rotation init_target {}), ".format(BASKET_INIT_ROT)

        s += "(geom end_target), "
        s += "(value end_target {}), ".format(BASKET_END_POS)
        s += "(rotation end_target {}), ".format(BASKET_END_ROT)

        s += "(geom cloth), "
        s += "(pose cloth {}), ".format(CLOTH_INIT_POS)
        s += "(rotation cloth {}), ".format(CLOTH_INIT_ROT)

        s += "(value cloth_target_begin_1 {}), ".format(CLOTH_INIT_POS_1)
        s += "(rotation cloth_target_begin_1 {}), ".format(CLOTH_INIT_ROT_1)

        s += "(value cloth_target_end_1 {}), ".format(CLOTH_END_POS_1)
        s += "(rotation cloth_target_end_1 {}), ".format(CLOTH_END_ROT_1)

        s += get_underfine_symbol("cloth_target_begin_2")
        s += get_underfine_symbol("cloth_target_end_2")

        s += get_underfine_symbol("cg_ee_1")
        s += get_underfine_symbol("cp_ee_1")
        s += get_underfine_symbol("bg_ee_left")
        s += get_underfine_symbol("bg_ee_right")
        s += get_underfine_symbol("bp_ee_left")
        s += get_underfine_symbol("bp_ee_right")
        # s += get_underfine_symbol("open_door_ee")
        s += '(value open_door_ee {}), '.format(WASHER_EE_POS)
        s += '(rotation open_door_ee {}), '.format(WASHER_EE_ROT)
        s += get_underfine_symbol("cg_ee_2")
        s += get_underfine_symbol("cp_ee_2")
        # s += get_underfine_symbol("close_door_ee")
        s += '(value close_door_ee {}), '.format(WASHER_END_EE_POS)
        s += '(rotation close_door_ee {}), '.format(WASHER_END_EE_ROT)

        s += get_baxter_str('baxter', L_ARM_INIT, R_ARM_INIT, INT_GRIPPER, BAXTER_INIT_POSE)

        s += get_robot_pose_str('robot_init_pose', L_ARM_INIT, R_ARM_INIT, INT_GRIPPER, BAXTER_INIT_POSE)
        s += get_undefined_robot_pose_str("cloth_grasp_begin_1")
        s += get_undefined_robot_pose_str("cloth_grasp_end_1")
        s += get_undefined_robot_pose_str("cloth_putdown_begin_1")
        # s += get_robot_pose_str('cloth_putdown_begin_1', CLOTH_PUTDOWN_BEGIN_1_LARMPOSE, CLOTH_PUTDOWN_BEGIN_1_RARMPOSE, CLOSE_GRIPPER, CLOTH_PUTDOWN_BEGIN_1_POSE)
        s += get_undefined_robot_pose_str("cloth_putdown_end_1")
        s += get_undefined_robot_pose_str("basket_grasp_begin")
        s += get_undefined_robot_pose_str("basket_grasp_end")
        s += get_undefined_robot_pose_str("basket_putdown_begin")
        s += get_undefined_robot_pose_str("basket_putdown_end")
        # s += get_undefined_robot_pose_str("open_door_begin")
        s += get_robot_pose_str('open_door_begin', WASHER_BEGIN_LARMPOSE, WASHER_BEGIN_RARMPOSE, INT_GRIPPER, WASHER_BEGIN_POSE)
        s += get_undefined_robot_pose_str("open_door_end")
        s += get_undefined_robot_pose_str("cloth_grasp_begin_2")
        s += get_undefined_robot_pose_str("cloth_grasp_end_2")
        s += get_undefined_robot_pose_str("cloth_putdown_begin_2")
        s += get_undefined_robot_pose_str("cloth_putdown_end_2")
        s += get_undefined_robot_pose_str("close_door_begin")
        s += get_undefined_robot_pose_str("close_door_end")
        s += get_robot_pose_str('robot_end_pose', L_ARM_END, R_ARM_END, END_GRIPPER, BAXTER_END_POSE)

        s += "(geom washer {}), ".format(WASHER_CONFIG)
        s += "(pose washer {}), ".format(WASHER_POS)
        s += "(rotation washer {}), ".format(WASHER_ROT)
        s += "(door washer {}), ".format(WASHER_DOOR)

        s += "(geom washer_init_pose {}), ".format(WASHER_CONFIG)
        s += "(value washer_init_pose {}), ".format(WASHER_POS)
        s += "(rotation washer_init_pose {}), ".format(WASHER_ROT)
        s += "(door washer_init_pose {}), ".format(WASHER_DOOR)

        s += "(geom washer_end_pose {}), ".format(WASHER_CONFIG)
        s += "(value washer_end_pose {}), ".format(WASHER_POS)
        s += "(rotation washer_end_pose {}), ".format(WASHER_ROT)
        s += "(door washer_end_pose {}), ".format(WASHER_END_DOOR)

        s += "(geom table {}), ".format(TABLE_GEOM)
        s += "(pose table {}), ".format(TABLE_POS)
        s += "(rotation table {}); ".format(TABLE_ROT)


        s += "(BaxterAt basket init_target), "
        s += "(BaxterBasketLevel basket), "
        s += "(BaxterRobotAt baxter robot_init_pose), "
        s += "(BaxterWasherAt washer washer_init_pose), "
        s += "(BaxterEEReachableLeftVer baxter basket_grasp_begin bg_ee_left), "
        s += "(BaxterEEReachableRightVer baxter basket_grasp_begin bg_ee_right), "

        s += "(BaxterBasketGraspLeftPos bg_ee_left init_target), "
        s += "(BaxterBasketGraspLeftRot bg_ee_left init_target), "
        s += "(BaxterBasketGraspRightPos bg_ee_right init_target), "
        s += "(BaxterBasketGraspRightRot bg_ee_right init_target), "

        s += "(BaxterBasketGraspLeftPos bp_ee_left end_target), "
        s += "(BaxterBasketGraspLeftRot bp_ee_left end_target), "
        s += "(BaxterBasketGraspRightPos bp_ee_right end_target), "
        s += "(BaxterBasketGraspRightRot bp_ee_right end_target), "
        s += "(BaxterStationaryBase baxter), "
        s += "(BaxterIsMP baxter), "
        s += "(BaxterWithinJointLimit baxter), "
        s += "(BaxterStationaryW table) \n\n"

        s += "Goal: {}".format(GOAL)

        with open(filename, "w") as f:
            f.write(s)

if __name__ == "__main__":
    main()
