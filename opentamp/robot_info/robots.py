import opentamp
import numpy as np
from openravepy import IkParameterizationType, databases


class Robot(object):
    """
    Base class of every robot parameter
    """
    def __init__(self, shape):
        self.shape = shape

class Baxter(Robot):
    """
    Defines geometry used in the Baxter domain.
    """
    def __init__(self):
        self._type = "baxter"
        baxter_shape = baxter_gym.__path__[0]+"/robot_info/baxter/baxter.zae"
        # self.col_links = set(["torso", "pedestal", "head", "sonar_ring", "screen", "collision_head_link_1",
        #                       "collision_head_link_2", "right_upper_shoulder", "right_lower_shoulder",
        #                       "right_upper_elbow", "right_upper_elbow_visual", "right_lower_elbow",
        #                       "right_upper_forearm", "right_upper_forearm_visual", "right_lower_forearm",
        #                       "right_wrist", "right_hand", "right_gripper_base", "right_gripper",
        #                       "right_gripper_l_finger", "right_gripper_r_finger", "right_gripper_l_finger_tip",
        #                       "right_gripper_r_finger_tip", "left_upper_shoulder", "left_lower_shoulder",
        #                       "left_upper_elbow", "left_upper_elbow_visual", "left_lower_elbow",
        #                       "left_upper_forearm", "left_upper_forearm_visual", "left_lower_forearm",
        #                       "left_wrist", "left_hand", "left_gripper_base", "left_gripper",
        #                       "left_gripper_l_finger", "left_gripper_r_finger", "left_gripper_l_finger_tip",
        #                       "left_gripper_r_finger_tip"])
        self.col_links = set(["torso", "head", "sonar_ring", "screen", "collision_head_link_1",
                              "collision_head_link_2", "right_upper_shoulder", "right_lower_shoulder",
                              "right_upper_elbow", "right_upper_elbow_visual", "right_lower_elbow",
                              "right_upper_forearm", "right_upper_forearm_visual", "right_lower_forearm",
                              "right_wrist", "right_hand", "right_gripper_base", "right_gripper",
                              "right_gripper_l_finger", "right_gripper_r_finger", "right_gripper_l_finger_tip",
                              "right_gripper_r_finger_tip", "left_upper_shoulder", "left_lower_shoulder",
                              "left_upper_elbow", "left_upper_elbow_visual", "left_lower_elbow",
                              "left_upper_forearm", "left_upper_forearm_visual", "left_lower_forearm",
                              "left_wrist", "left_hand", "left_gripper_base", "left_gripper",
                              "left_gripper_l_finger", "left_gripper_r_finger", "left_gripper_l_finger_tip",
                              "left_gripper_r_finger_tip"])
        self.dof_map = {"lArmPose": list(range(2,9)), "lGripper": [9], "rArmPose": list(range(10,17)), "rGripper":[17]}
        super(Baxter, self).__init__(baxter_shape)

    def setup(self, robot):
        """
        Need to setup iksolver for baxter
        """
        iktype = IkParameterizationType.Transform6D
        ikmodel = databases.inversekinematics.InverseKinematicsModel(robot, IkParameterizationType.Transform6D, True)
        if not ikmodel.load():
            ikmodel.autogenerate()
        right_manip = robot.GetManipulator('right_arm')
        ikmodel.manip = right_manip
        right_manip.SetIkSolver(ikmodel.iksolver)

        ikmodel = databases.inversekinematics.InverseKinematicsModel(robot, IkParameterizationType.Transform6D, True)
        if not ikmodel.load():
            ikmodel.autogenerate()
        left_manip = robot.GetManipulator('left_arm')
        ikmodel.manip = left_manip
        left_manip.SetIkSolver(ikmodel.iksolver)

