from IPython import embed as shell
import itertools
import numpy as np
import random

import ros_interface.utils as utils


NUM_CLOTH = 4
NUM_SYMBOLS = 5

# SEED = 1234
NUM_PROBS = 1
filename = "laundry_probs/baxter_policy_{0}.prob".format(NUM_CLOTH)
GOAL = "(BaxterRobotAt baxter robot_end_pose)"


# init Baxter pose
BAXTER_INIT_POSE = [0]
BAXTER_END_POSE = [0]
R_ARM_INIT = [0, 0, 0, 0, 0, 0, 0] # [0, -0.8436, -0.09, 0.91, 0.043, 1.5, -0.05] # [ 0.1, -1.36681967, -0.23718529, 1.45825713, 0.04779009, 1.48501637, -0.92194262]
L_ARM_INIT = [0, 0, 0, 0, 0, 0, 0] # [-0.6, -1.2513685 , -0.63979997, 1.41307933, -2.9520384, -1.4709618, 2.69274026]
OPEN_GRIPPER = [0.02]
CLOSE_GRIPPER = [0.015]

MONITOR_LEFT = [np.pi/4, -np.pi/4, 0, 0, 0, 0, 0]
MONITOR_RIGHT = [-np.pi/4, -np.pi/4, 0, 0, 0, 0, 0]

# init basket pose
BASKET_INIT_POS = [0.65, 0.1, 0.875]
BASKET_END_POS = [0.65, 0.1, 0.875]
BASKET_INIT_ROT = [np.pi/2, 0, np.pi/2]
BASKET_END_ROT = [np.pi/2, 0, np.pi/2]

CLOTH_ROT = [0, 0, 0]

TABLE_GEOM = [1.23/2, 2.45/2, 0.97/2]
TABLE_POS = [1.23/2-0.1, 0, 0.97/2-0.375]
TABLE_ROT = [0,0,0]

ROBOT_DIST_FROM_TABLE = 0.05

WASHER_CONFIG = [True, True]
# WASHER_INIT_POS = [0.97, 1.0, 0.97-0.375+0.65/2]
# WASHER_INIT_ROT = [np.pi/2,0,0]
# WASHER_INIT_POS = [0.85, 1.25, 0.97-0.375+0.65/2]
WASHER_INIT_POS = [1.35, 1.75, 0.97-0.375+0.65/2]
WASHER_INIT_ROT = [np.pi/4,0,0]

WASHER_OPEN_DOOR = [-np.pi/2]
WASHER_CLOSE_DOOR = [0.0]
WASHER_PUSH_DOOR = [-np.pi/6]

REGION1 = [np.pi/4]
REGION2 = [0]
REGION3 = [-np.pi/4]
REGION4 = [-np.pi/2]

cloth_init_poses = np.ones((NUM_CLOTH, 3)) * 0.615
cloth_init_poses = cloth_init_poses.tolist()

def get_baxter_str(name, LArm = L_ARM_INIT, RArm = R_ARM_INIT, G = OPEN_GRIPPER, Pos = BAXTER_INIT_POSE):
    s = ""
    s += "(geom {})".format(name)
    s += "(lArmPose {} {}), ".format(name, LArm)
    s += "(lGripper {} {}), ".format(name, G)
    s += "(rArmPose {} {}), ".format(name, RArm)
    s += "(rGripper {} {}), ".format(name, G)
    s += "(pose {} {}), ".format(name, Pos)
    return s

def get_robot_pose_str(name, LArm = L_ARM_INIT, RArm = R_ARM_INIT, G = OPEN_GRIPPER, Pos = BAXTER_INIT_POSE):
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

def get_undefined_symbol(name):
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
        # s += "Basket (name {}); ".format("basket")

        s += "Robot (name baxter); "
        for i in range(NUM_CLOTH):
            s += "Cloth (name {}); ".format("cloth_{0}".format(i))
            s += "ClothTarget (name {}); ".format("cloth_target_{0}".format(i))
            s += "ClothTarget (name {}); ".format("cloth{0}_init_target".format(i))

        s += "EEPose (name {}); ".format("cg_ee".format(i))
        s += "EEPose (name {}); ".format("cp_ee".format(i))
        s += "RobotPose (name {}); ".format("cloth_grasp_begin".format(i))
        s += "RobotPose (name {}); ".format("cloth_grasp_end".format(i))
        s += "RobotPose (name {}); ".format("cloth_putdown_begin".format(i))
        s += "RobotPose (name {}); ".format("cloth_putdown_end".format(i))
        s += "ClothTarget (name {}); ".format("middle_target_1")
        s += "ClothTarget (name {}); ".format("middle_target_2")

        s += "RobotPose (name {}); ".format("robot_init_pose")
        s += "RobotPose (name {}); ".format("robot_end_pose")
        s += "RobotPose (name {}); ".format("basket_grasp_begin")
        s += "RobotPose (name {}); ".format("basket_grasp_end")
        s += "RobotPose (name {}); ".format("basket_putdown_begin")
        s += "RobotPose (name {}); ".format("basket_putdown_end")
        s += "RobotPose (name {}); ".format("basket_rotate_begin")
        s += "RobotPose (name {}); ".format("rotate_end_pose")
        s += "EEPose (name {}); ".format("bg_ee_left")
        s += "EEPose (name {}); ".format("bp_ee_left")
        s += "EEPose (name {}); ".format("bg_ee_right")
        s += "EEPose (name {}); ".format("bp_ee_right")
        s += "EEPose (name {}); ".format("open_door_ee_approach")
        s += "EEPose (name {}); ".format("open_door_ee_approach")
        s += "EEPose (name {}); ".format("close_door_ee_approach")
        s += "EEPose (name {}); ".format("close_door_ee_retreat")
        s += "RobotPose (name {}); ".format("close_door_begin")
        s += "RobotPose (name {}); ".format("open_door_begin")
        s += "RobotPose (name {}); ".format("close_door_end")
        s += "RobotPose (name {}); ".format("open_door_end")
        # s += "WasherPose (name {}); ".format("washer_open_pose")
        # s += "WasherPose (name {}); ".format("washer_close_pose")
        # s += "Washer (name {}); ".format("washer")
        s += "Obstacle (name {}); ".format("table")
        s += "BasketTarget (name {}); ".format("basket_init_target")
        s += "BasketTarget (name {}) \n\n".format("basket_end_target")

        s += "Init: "
        # s += "(geom basket), "
        # s += "(pose basket {}), ".format(BASKET_INIT_POS)
        # s += "(rotation basket {}), ".format(BASKET_INIT_ROT)

        for i in range(NUM_CLOTH):
            s += "(geom cloth_{0}), ".format(i)
            s += "(pose cloth_{0} {1}), ".format(i, cloth_init_poses[i])
            s += "(rotation cloth_{0} {1}), ".format(i, CLOTH_ROT)
            s += "(value cloth{0}_init_target [0, 0, 0], ".format(i)
            s += "(rotation cloth{0}_init_target [0, 0, 0], ".format(i)
            s += "(value cloth_target_{0} [0, 0, 0], ".format(i)
            s += "(rotation cloth_target_{0} [0, 0, 0], ".format(i)

        s += "(value middle_target_1 [0, 0, 0], "
        s += "(rotation middle_target_1 [0, 0, 0], "
        s += "(value middle_target_2 [0, 0, 0], "
        s += "(rotation middle_target_2 [0, 0, 0], "

        s += get_undefined_symbol('cloth_target_end')
        s += get_undefined_symbol('cloth_target_begin')

        s += get_undefined_symbol("cg_ee")
        s += get_undefined_symbol("cp_ee")

        s += get_undefined_robot_pose_str("cloth_grasp_begin".format(i))
        s += get_undefined_robot_pose_str("cloth_grasp_end".format(i))
        s += get_undefined_robot_pose_str("cloth_putdown_begin".format(i))
        s += get_undefined_robot_pose_str("cloth_putdown_end".format(i))

        s += get_baxter_str('baxter', L_ARM_INIT, R_ARM_INIT, OPEN_GRIPPER, BAXTER_INIT_POSE)
        s += get_robot_pose_str('robot_init_pose', L_ARM_INIT, R_ARM_INIT, OPEN_GRIPPER, BAXTER_INIT_POSE)
        s += get_robot_pose_str('robot_end_pose', L_ARM_INIT, R_ARM_INIT, OPEN_GRIPPER, BAXTER_END_POSE)
        s += get_undefined_robot_pose_str("basket_grasp_begin")
        s += get_undefined_robot_pose_str("basket_grasp_end")
        s += get_undefined_robot_pose_str("basket_putdown_begin")
        s += get_undefined_robot_pose_str("basket_putdown_end")
        s += get_undefined_robot_pose_str("basket_rotate_begin")
        s += get_undefined_robot_pose_str("rotate_end_pose")
        s += get_undefined_symbol("bg_ee_left")
        s += get_undefined_symbol("bp_ee_left")
        s += get_undefined_symbol("bg_ee_right")
        s += get_undefined_symbol("bp_ee_right")
        s += get_undefined_symbol("open_door_ee_approach")
        s += get_undefined_symbol("open_door_ee_approach")
        s += get_undefined_symbol("close_door_ee_approach")
        s += get_undefined_symbol("close_door_ee_retreat")
        s += get_undefined_robot_pose_str("close_door_begin")
        s += get_undefined_robot_pose_str("open_door_begin")
        s += get_undefined_robot_pose_str("close_door_end")
        s += get_undefined_robot_pose_str("open_door_end")

        # s += "(geom washer_open_pose {0}), ".format(WASHER_CONFIG)
        # s += "(value washer_open_pose {0}), ".format(WASHER_INIT_POS)
        # s += "(rotation washer_open_pose {0}), ".format(WASHER_INIT_ROT)
        # s += "(door washer_open_pose {0}), ".format(WASHER_OPEN_DOOR)

        # s += "(geom washer_close_pose {0}), ".format(WASHER_CONFIG)
        # s += "(value washer_close_pose {0}), ".format(WASHER_INIT_POS)
        # s += "(rotation washer_close_pose {0}), ".format(WASHER_INIT_ROT)
        # s += "(door washer_close_pose {0}), ".format(WASHER_CLOSE_DOOR)

        # s += "(geom washer {}), ".format(WASHER_CONFIG)
        # s += "(pose washer {}), ".format(WASHER_INIT_POS)
        # s += "(rotation washer {}), ".format(WASHER_INIT_ROT)
        # s += "(door washer {}), ".format(WASHER_CLOSE_DOOR)

        s += "(geom table {}), ".format(TABLE_GEOM)
        s += "(pose table {}), ".format(TABLE_POS)
        s += "(rotation table {}), ".format(TABLE_ROT)

        s += "(geom basket_init_target), "
        s += "(value basket_init_target {}), ".format(BASKET_INIT_POS)
        s += "(rotation basket_init_target {}), ".format(BASKET_INIT_ROT)

        s += "(geom basket_end_target), "
        s += "(value basket_end_target {}), ".format(BASKET_END_POS)
        s += "(rotation basket_end_target {}); ".format(BASKET_END_ROT)


        # s += "(BaxterAt basket basket_init_target), "
        # s += "(BaxterBasketLevel basket), "
        # s += "(BaxterRobotAt baxter robot_init_pose) \n\n"
        # s += "(BaxterWasherAt washer washer_init_pose), "
        # s += "(BaxterEEReachableLeftVer baxter basket_grasp_begin bg_ee_left), "
        # s += "(BaxterEEReachableRightVer baxter basket_grasp_begin bg_ee_right), "
        # s += "(BaxterBasketGraspValidPos bg_ee_left bg_ee_right basket_init_target), "
        # s += "(BaxterBasketGraspValidRot bg_ee_left bg_ee_right basket_init_target), "
        # s += "(BaxterBasketGraspValidPos bp_ee_left bp_ee_right end_target), "
        # s += "(BaxterBasketGraspValidRot bp_ee_left bp_ee_right end_target), "
        # s += "(BaxterStationaryBase baxter), "
        s += "(BaxterIsMP baxter), "
        s += "(BaxterWithinJointLimit baxter), "
        s += "(BaxterStationaryW table) \n\n"

        s += "Goal: {}".format(GOAL)

        with open(filename, "w") as f:
            f.write(s)

if __name__ == "__main__":
    main()
