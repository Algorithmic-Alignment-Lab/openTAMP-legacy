from IPython import embed as shell
import itertools
import numpy as np
import random


# SEED = 1234
NUM_PROBS = 1
filename = "probs/temp_pickplace_prob.prob"
GOAL = "(RobotAt sawyer robot_end_pose)"


SAWYER_INIT_POSE = [-0.5, -0.1, 0.912]
SAWYER_END_POSE = [0, 0, 0]
R_ARM_INIT = [-0.3962099, -0.97739413, 0.04612799, 1.742205 , -0.03562013, 0.8089644, 0.45207395]
OPEN_GRIPPER = [0.02, -0.01]
CLOSE_GRIPPER = [0.01, -0.02]
EE_POS = [0.11338, -0.16325, 1.03655]
EE_ROT = [3.139, 0.00, -2.182]

TABLE_GEOM = [1.23/2, 2.45/2, 0.97/2]
TABLE_POS = [1.23/2-0.1, 0, -3.]
TABLE_ROT = [0,0,0]

def get_sawyer_pose_str(name, RArm = R_ARM_INIT, G = OPEN_GRIPPER, Pos = SAWYER_INIT_POSE):
    s = ""
    s += "(right {} {}), ".format(name, RArm)
    s += "(right_ee_pos {} {}), ".format(name, EE_POS)
    s += "(right_ee_rot {} {}), ".format(name, EE_ROT)
    s += "(right_gripper {} {}), ".format(name, G)
    s += "(value {} {}), ".format(name, Pos)
    s += "(rotation {} {}), ".format(name, [0.,0.,0.])
    return s

def get_sawyer_str(name, RArm = R_ARM_INIT, G = OPEN_GRIPPER, Pos = SAWYER_INIT_POSE):
    s = ""
    s += "(geom {})".format(name)
    s += "(right {} {}), ".format(name, RArm)
    s += "(right_ee_pos {} {}), ".format(name, EE_POS)
    s += "(right_ee_rot {} {}), ".format(name, EE_ROT)
    s += "(right_gripper {} {}), ".format(name, G)
    s += "(pose {} {}), ".format(name, Pos)
    s += "(rotation {} {}), ".format(name, [0.,0.,0.])
    return s

def get_undefined_robot_pose_str(name):
    s = ""
    s += "(right {} undefined), ".format(name)
    s += "(right_ee_pos {} undefined), ".format(name)
    s += "(right_ee_rot {} undefined), ".format(name)
    s += "(right_gripper {} undefined), ".format(name)
    s += "(value {} undefined), ".format(name)
    s += "(rotation {} undefined), ".format(name)
    return s

def get_undefined_symbol(name):
    s = ""
    s += "(value {} undefined), ".format(name)
    s += "(rotation {} undefined), ".format(name)
    return s

def main():
    s = "# AUTOGENERATED. DO NOT EDIT.\n# Configuration file for CAN problem instance. Blank lines and lines beginning with # are filtered out.\n\n"

    s += "# The values after each attribute name are the values that get passed into the __init__ method for that attribute's class defined in the domain configuration.\n"
    s += "Objects: "
    s += "Sawyer (name sawyer); "

    for item in ['milk', 'bread', 'cereal', 'can']:
        s += "SawyerPose (name {}); ".format("{0}_grasp_begin".format(item))
        s += "SawyerPose (name {}); ".format("{0}_grasp_end".format(item))
        s += "SawyerPose (name {}); ".format("{0}_putdown_begin".format(item))
        s += "SawyerPose (name {}); ".format("{0}_putdown_end".format(item))
        if item is 'can':
            item_type = 'Can'
        else:
            item_type = 'Box'
        s += "{} (name {}); ".format(item_type, item)
        s += "{}Target (name {}_init_target); ".format(item_type, item)
        s += "{}Target (name {}_end_target); ".format(item_type, item)

    s += "SawyerPose (name {}); ".format("robot_init_pose")
    s += "SawyerPose (name {}); ".format("robot_end_pose")
    s += "Obstacle (name {}) \n\n".format("table")

    s += "Init: "
    dims = [[0.0225, 0.0225, 0.06], [0.02, 0.02, 0.02], [0.05, 0.02, 0.065], [0.02, 0.04]]
    s += "(geom milk {0}), ".format(dims[0])
    s += "(geom bread {0}), ".format(dims[1])
    s += "(geom cereal {0}), ".format(dims[2])
    s += "(geom can {0} {1}), ".format(dims[3][0], dims[3][1])
    s += "(geom milk_end_target {0}), ".format(dims[0])
    s += "(geom bread_end_target {0}), ".format(dims[1])
    s += "(geom cereal_end_target {0}), ".format(dims[2])
    s += "(geom can_end_target {0} {1}), ".format(dims[3][0], dims[3][1])
    s += "(geom milk_init_target {0}), ".format(dims[0])
    s += "(geom bread_init_target {0}), ".format(dims[1])
    s += "(geom cereal_init_target {0}), ".format(dims[2])
    s += "(geom can_init_target {0} {1}), ".format(dims[3][0], dims[3][1])
    end_targets = [[0.0025, 0.1575, 0.885], [0.1975, 0.1575, 0.845], [0.0025, 0.4025, 0.9], [0.1975, 0.4025, 0.86]]
    for ind, item in enumerate(['milk', 'bread', 'cereal', 'can']):
        s += "(pose {0} {1}), ".format(item, [0, 0, 0])
        s += "(rotation {0} {1}), ".format(item, [0, 0, 0])
        s += "(value {0}_init_target [0, 0, 0]), ".format(item)
        s += "(rotation {0}_init_target [0, 0, 0]), ".format(item)
        s += "(value {}_end_target {}), ".format(item, end_targets[ind])
        s += "(rotation {}_end_target [0, 0, 0]), ".format(item)

        s += get_undefined_robot_pose_str("{0}_grasp_begin".format(item))
        s += get_undefined_robot_pose_str("{0}_grasp_end".format(item))
        s += get_undefined_robot_pose_str("{0}_putdown_begin".format(item))
        s += get_undefined_robot_pose_str("{0}_putdown_end".format(item))
    s += get_sawyer_str('sawyer', R_ARM_INIT, OPEN_GRIPPER, SAWYER_INIT_POSE)
    s += get_sawyer_pose_str('robot_init_pose', R_ARM_INIT, OPEN_GRIPPER, SAWYER_INIT_POSE)
    s += get_undefined_robot_pose_str('robot_end_pose')

    s += "(geom table {}), ".format(TABLE_GEOM)
    s += "(pose table {}), ".format(TABLE_POS)
    s += "(rotation table {}); ".format(TABLE_ROT)

    for item in ['milk', 'bread', 'can', 'cereal']:
        s += "(At {0} {0}_init_target), ".format(item)
        s += "(AtInit {0} {0}_init_target), ".format(item)
        s += "(Near {0} {0}_init_target), ".format(item)
    s += "(RobotAt sawyer robot_init_pose),"
    s += "(StationaryBase sawyer), "
    s += "(HeightBlock bread cereal), "
    s += "(HeightBlock can cereal), "
    s += "(HeightBlock milk cereal), "
    s += "(HeightBlock bread milk), "
    s += "(HeightBlock can milk), "
    s += "(HeightBlock bread can), "
    s += "(IsMP sawyer), "
    s += "(WithinJointLimit sawyer), "
    s += "(StationaryW table) \n\n"

    s += "Goal: {}\n\n".format(GOAL)

    s += "Invariants: "
    s += "(StationaryBase sawyer), "
    #s += "(EEIsMP sawyer), "
    #s += "(RightGripperDownRot sawyer), "
    #s += "(RightEEValid sawyer), "
    s += "\n\n"

    with open(filename, "w") as f:
        f.write(s)

if __name__ == "__main__":
    main()
