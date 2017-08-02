import numpy as np
"""
This file contains constants specifically in baxter domain.
This file is refferenced in:
    baxter_predicates
    baxter_sampling
    test_baxter_predicates
"""

"""
Following Constants are used in baxter_predicates
"""
# Baxter dimension constant
BASE_DIM = 1
JOINT_DIM = 16
ROBOT_ATTR_DIM = 17
TWOARMDIM = 16
JOINT_FAST_VELOCITY = 0.1
JOINT_SLOW_VELOCITY = 0.01

# Baxter Movement Constraints
BASE_MOVE = 1
JOINT_MOVE_FACTOR = 15

# EEReachable Constants
APPROACH_DIST = 0.015
RETREAT_DIST = 0.015
EEREACHABLE_STEPS = 5

# Collision Constants
DIST_SAFE = 1e-4
RCOLLIDES_DSAFE = 1e-4
COLLIDES_DSAFE = 1e-4

# Plan Coefficient

EEREACHABLE_COEFF = 5e1
EEREACHABLE_ROT_COEFF = 3e1
IN_GRIPPER_COEFF = 7e1
IN_GRIPPER_ROT_COEFF = 2e1
WASHER_IN_GRIPPER_ROT_COEFF = 1e-2

OBSTRUCTS_COEFF = 1e0
COLLIDE_COEFF = 1e0
RCOLLIDE_COEFF = 1e1
MAX_CONTACT_DISTANCE = .1

GRASP_VALID_COEFF = 1.3e2
EEGRASP_VALID_COEFF = 2e2
# Gripper Value
GRIPPER_OPEN_VALUE = 0.02
GRIPPER_CLOSE_VALUE = 0.015
BASKET_OFFSET = 0.317

WASHER_DEPTH_OFFSET = 0.05

"""
Following constants are used in baxter_sampling
"""

EE_ANGLE_SAMPLE_SIZE = 8
NUM_RESAMPLES = 10
MAX_ITERATION_STEP = 200
BIAS_RADIUS = 0.1
ROT_BIAS = np.pi/8
RESAMPLE_FACTOR = [0.005,0.005,0.25]
RESAMPLE_DIR = [1,1,0.5]
RESAMPLE_ROT = [np.pi/2, 0, 0]
RESAMPLE_FACTOR_LR = [0.1, 0.1, 0.05]
"""
Following constants are for testing purposes
"""

TOL = 1e-4
TEST_GRAD = True
