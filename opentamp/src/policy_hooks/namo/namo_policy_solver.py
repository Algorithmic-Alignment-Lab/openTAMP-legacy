import os

import numpy as np

import gurobipy as grb

from sco_py.expr import BoundExpr, QuadExpr, AffExpr
from sco_py.prob import Prob
from sco_py.solver import Solver
from sco_py.variable import Variable

from gps.algorithm.cost.cost_utils import *

from core.util_classes.namo_predicates import ATTRMAP
from pma.namo_solver import NAMOSolver
# from opentamp.src.policy_hooks.namo.multi_task_main import GPSMain
from opentamp.src.policy_hooks.namo.vector_include import *
from opentamp.src.policy_hooks.utils.load_task_definitions import *
from opentamp.src.policy_hooks.namo.namo_agent import NAMOSortingAgent
# import policy_hooks.namo.namo_hyperparams as namo_hyperparams
# import policy_hooks.namo.namo_optgps_hyperparams as namo_hyperparams
from opentamp.src.policy_hooks.namo.namo_policy_predicates import NAMOPolicyPredicate
from opentamp.src.policy_hooks.utils.policy_solver_utils import *
from opentamp.src.policy_hooks.namo.sorting_prob_2 import *
from opentamp.src.policy_hooks.task_net import tf_binary_network, tf_classification_network
from opentamp.src.policy_hooks.mcts import MCTS
from opentamp.src.policy_hooks.state_traj_cost import StateTrajCost
from opentamp.src.policy_hooks.action_traj_cost import ActionTrajCost
from opentamp.src.policy_hooks.traj_constr_cost import TrajConstrCost
from opentamp.src.policy_hooks.cost_product import CostProduct
from opentamp.src.policy_hooks.sample import Sample
from opentamp.src.policy_hooks.policy_solver import get_base_solver

BASE_DIR = os.getcwd() + '/policy_hooks/'
EXP_DIR = BASE_DIR + '/experiments'

# N_RESAMPLES = 5
# MAX_PRIORITY = 3
# DEBUG=False

BASE_CLASS = get_base_solver(NAMOSolver)

class NAMOPolicySolver(BASE_CLASS):
    pass
