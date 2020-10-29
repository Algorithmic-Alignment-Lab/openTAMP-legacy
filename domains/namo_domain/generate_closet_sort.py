import itertools
import random

from core.util_classes.namo_predicates import dsafe

# NUM_CANS = 4


GOAL = "(RobotAt pr2 robot_end_pose)"
HEIGHT = 5
WIDTH = 5
NUM_END = 20


def main():
    for NUM_CANS in range(1, 10):
        for N_END in range(1, 20):
            for N_AUX in range(0, 5):
                for N_GRASP in range(1, 5):
                    filename = "namo_probs/sort_closet_prob_{0}_{1}end_{2}_{3}aux.prob".format(NUM_CANS, N_END, N_AUX, N_GRASP)
                    s = "# AUTOGENERATED. DO NOT EDIT.\n# Configuration file for NAMO problem instance. Blank lines and lines beginning with # are filtered out.\n\n"
                    coords = list(itertools.product(list(range(-HEIGHT, HEIGHT)), list(range(-WIDTH, WIDTH))))
                    random.shuffle(coords)
                    coord_ind = 0
                    s += "# The values after each attribute name are the values that get passed into the __init__ method for that attribute's class defined in the domain configuration.\n"
                    s += "Objects: "
                    for n in range(NUM_CANS):
                        s += "Target (name can%d_init_target); "%(n)
                        s += "Can (name can%d); "%(n)
                        s += "Target (name can%d_end_target); "%(n)
                    for i in range(N_END):
                        s += "Target (name end_target_%d); "%(i)

                    s += "Robot (name %s); "%"pr2"
                    s += "Grasp (name {}); ".format("grasp0")
                    if N_GRASP > 1: s += "Grasp (name {}); ".format("grasp1")
                    if N_GRASP > 2: s += "Grasp (name {}); ".format("grasp2")
                    if N_GRASP > 3: s += "Grasp (name {}); ".format("grasp3")
                    s += "RobotPose (name %s); "%"robot_init_pose"
                    s += "RobotPose (name %s); "%"robot_end_pose"
                    for i in range(NUM_CANS):
                        s += "RobotPose (name %s); "%"grasp_pose_{0}".format(i)

                    for i in range(N_AUX):
                        s += "Target (name %s); "%"aux_target_{0}".format(i)
                    s += "Obstacle (name %s) \n\n"%"obs0"

                    s += "Init: "
                    for i in range(NUM_CANS):
                        s += "(geom can%d_init_target 0.3), (value can%d_init_target %s), "%(i, i, list(coords[i]))
                        s += "(geom can%d 0.3), (pose can%d %s), "%(i, i, list(coords[i]))
                        s += "(geom can%d_end_target 0.3), (value can%d_end_target %s), "%(i, i, list(coords[i]))
                    for i in range(N_END):
                        s += "(geom end_target_%d 0.3), (value end_target_%d %s), "%(i, i, list(coords[i]))
                    # s += "(value grasp0 undefined), "
                    s += "(value grasp0 [0, {0}]), ".format(-0.6-dsafe)
                    if N_GRASP > 1: s += "(value grasp1 [0, {0}]), ".format(0.6+dsafe)
                    if N_GRASP > 2: s += "(value grasp2 [{0}, 0]), ".format(-0.6-dsafe)
                    if N_GRASP > 3: s += "(value grasp3 [{0}, 0]), ".format(0.6+dsafe)
                    s += "(geom %s 0.3), (pose %s %s), "%("pr2", "pr2", [0, 0])
                    s += "(gripper pr2 [-1.]), "
                    s += "(value %s %s), "%("robot_init_pose", [0., 0.])
                    s += "(value %s %s), "%("robot_end_pose", "undefined")
                    s += "(geom %s %s), "%("robot_init_pose", 0.3)
                    s += "(geom %s %s), "%("robot_end_pose", 0.3)
                    s += "(gripper %s [-1.]), "%("robot_init_pose")
                    s += "(gripper %s undefined), "%("robot_end_pose")
                    for i in range(NUM_CANS):
                        s += "(gripper %s undefined), "%("grasp_pose_{0}".format(i))
                        s += "(value %s undefined), "%("grasp_pose_{0}".format(i))
                        s += "(geom %s 0.3), "%("grasp_pose_{0}".format(i))

                    for i in range(N_AUX):
                        s += "(value %s [0., 0.]), "%("aux_target_{0}".format(i))
                        s += "(geom aux_target_{0} 0.3), ".format(i)
                    s += "(pose %s [-3.5, 0]), "%"obs0"
                    s += "(geom %s %s); "%("obs0", "closet")

                    for i in range(NUM_CANS):
                        s += "(AtInit can{} can{}_init_target), ".format(i, i)
                        # s += "(Near can{} can{}_init_target), ".format(i, i)
                    s += "(RobotAt pr2 robot_init_pose), "
                    s += "(IsMP pr2), "
                    s += "(StationaryW obs0) \n\n"

                    s += "Goal: %s"%GOAL

                    with open(filename, "w") as f:
                        f.write(s)

if __name__ == "__main__":
    main()

