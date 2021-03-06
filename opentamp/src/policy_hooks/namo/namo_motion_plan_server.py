import copy
import sys
import time
import traceback

import numpy as np
import tensorflow as tf

from opentamp.src.policy_hooks.abstract_motion_plan_server import AbstractMotionPlanServer
from opentamp.src.policy_hooks.sample import Sample
from opentamp.src.policy_hooks.utils.policy_solver_utils import *
from opentamp.src.policy_hooks.utils.tamp_eval_funcs import *
from opentamp.src.policy_hooks.namo.namo_policy_solver import NAMOPolicySolver


class DummyPolicyOpt(object):
    def __init__(self, prob):
        self.traj_prob = prob

class NAMOMotionPlanServer(AbstractMotionPlanServer):
    def __init__(self, hyperparams):
        self.solver = NAMOPolicySolver(hyperparams)
        super(NAMOMotionPlanServer, self).__init__(hyperparams)
