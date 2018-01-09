import main

import numpy as np

import rospy

from core.parsing import parse_domain_config, parse_problem_config
from pma.hl_solver import FFSolver
from ros_interface.basket.basket_predict import BasketPredict
from ros_interface.cloth.cloth_grid_predict import ClothGridPredict
from ros_interface.controllers import EEController, TrajectoryController



DOMAIN_FILE = "../domains/laundry_domain/domain.pddl"

class HLLaundryState(object):
    # TODO: This class is here for temporary usage but needs generalization
    #       Ideally integreate with existing HLState class and Env Monitor
    def __init__(self):
        self.region_poses = [[], [], [[0,0,0]], []]
        self.basket_pose = [0.8, -0.3, 2*np.pi/3] # The far location

        self.robot_region = 1
        self.cloth_in_basket = False
        self.cloth_in_washer = False
        self.basket_near_washer = False
        self.washer_door_open = False
        self.near_loc_clear = True
        self.far_loc_clear = True

        # For constructing high level plans
        self.prob_domain = "(:domain laundry_domain)\n"
        self.objects = "(:objects cloth washer basket)\n"
        self.goal = "(:goal (ClothInWasher cloth washer))\n"

    def reset_region_poses(self):
        self.region_poses = [[], [], [], []]

    def get_abs_prob(self):
        prob_str = "(define (problem laundry_prob)\n"
        prob_str += self.prob_domain
        prob_str += self.objects
        prob_str += self.get_init_state()
        prob_str += self.goal
        prob_str += ")\n"
        return prob_str

    def get_init_state(self):
        state_str = "(:init\n"

        regions_occupied = False
        for i in range(len(self.region_poses)):
            r = self.region_poses[i]
            if len(r):
                state_str += "(ClothInRegion{0} cloth)\n".format(i+1)
                regions_occupied = True if i > 0 else False

        if self.cloth_in_basket:
            state_str += "(ClothInBasket cloth basket)\n"
        if self.cloth_in_washer:
            state_str += "(ClothInWasher cloth washer)\n"
        if self.basket_near_washer:
            state_str += "(BasketNearWasher basket washer)\n"
        if self.near_loc_clear:
            state_str += "(BasketNearLocClear basket cloth)\n"
        if self.far_loc_clear:
            state_str += "(BasketFarLocClear basket cloth)\n"
        if self.washer_door_open:
            state_str += "(WasherDoorOpen washer)\n"

        state_str += ")\n"
        return state_str


class LaundryEnvironmentMonitor(object):
    def __init__(self):
        self.state = HLLaundryState()

        with open(DOMAIN_FILE, 'r+') as f:
            self.abs_domain = f.read()
        self.hl_solver = FFSolver(abs_domain=self.abs_domain)
        self.cloth_predictor = ClothGridPredict()
        # self.basket_predictor = BasketPredict()
        self.ee_control = EEController()
        self.traj_control = TrajectoryController()

    def run_baxter(self):
        hl_plan_str = self.solve_hl_prob()


    def solve_hl_prob(self):
        abs_prob = self.state.get_abs_prob()
        return self.hl_solver._run_planner(self.abs_domain, abs_prob)

    def execute_plan(self, plan, active_ts):
        current_ts = active_ts[0]
        while (current_ts < active_ts[1] and current_ts < plan.horizon):
            cur_action = plan.actions.filter(lambda a: a.active_timesteps[0] == current_ts)[0]
            if cur_action.name == "open_door":
                self.state.washer_door_open = True # TODO: Integrate door open/close prediction
            elif cur_action.name == "close_door":
                self.state.washer_door_open = False # TODO: Integrate door open/close prediction
            elif cur_action.name == "basket_putdown_with_cloth":
                self.state.basket_near_washer = cur_action.params[2].name == "basket_near_target"

            if cur_action.name == "center_over_basket":
                pass
            elif cur_action.name == "center_over_cloth":
                pass
            elif cur_action.name == "center_over_washer_handle":
                pass
            elif cur_action.name.startswith("rotate"):
                self.robot_region = int(cur_action.params[-1].name[-1]) + 1 # TODO: Make this more generalized
                # TODO: Add rotation integration here
            else:
                self.traj_control.execute_plan(plan, active_ts=cur_action.active_timesteps)
            current_ts = cur_action.active_timesteps[1]

    def predict_cloth_locations(self):
        self.state.reset_region_poses()
        locs = self.cloth_predictor.predict()
        for loc in locs:
            self.state.region_poses[loc[1]].append(loc[0])

    def predict_basket_location(self):
        self.state.basket_pose = self.basket_predictor.predict()

    def update_plan(self, plan):
        plan.params['basket'].pose[:2, 0] = self.state.basket_pose[:2]
        plan.params['basket'].rotation[2, 0] = self.state.basket_pose[2]
        cur_cloth_n = 0
        for region in self.state.region_poses:
            for pose in region:
                plan.params['cloth_{0}'.format(cur_cloth_n)].pose[:2] = pose
                cur_cloth_n += 1

    def plan_from_str(self, ll_plan_str):
        '''Convert a plan string (low level) into a plan object.'''
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = FFSolver(d_c)
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/baxter_laundry.prob'.format(num_cloths))
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
        return hls.get_plan(plan_str, domain, problem)

    def hl_to_ll(self, plan_str):
        '''Parses a high level plan into a sequence of low level actions.'''
        ll_plan_str = ''
        act_num = 0
        cur_cloth_n = 0
        last_pose = 'ROBOT_INIT_POSE'
        # TODO: Fill in eval functions for basket setdown region

        for action in plan_str:
            act_type = self._get_action_type(action)
            i = 0
            if act_type == 'load_basket_from_region_1':
                init_i = i
                ll_plan_str += '{0}: ROTATE BAXTER {1} ROBOT_REGION_1_POSE_{2} REGION_1 \n'.format(act_num, last_pose, i)
                act_num += 1
                while i < num_cloths_r1 + init_i:
                    ll_plan_str += '{0}: MOVETO BAXTER ROBOT_REGION_1_POSE_{1} CLOTH_GRASP_BEGIN_{2} \n'.format(act_num, i, i)
                    act_num += 1
                    ll_plan_str += '{0}: CLOTH_GRASP BAXTER CLOTH_{1} CLOTH_TARGET_BEGIN_{2} CLOTH_GRASP_BEGIN_{3} CG_EE_{4} CLOTH_GRASP_END_{5} \n'.format(act_num, cur_cloth_n, i, i, i, i)
                    act_num += 1
                    ll_plan_str += '{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_{1} CLOTH_PUTDOWN_BEGIN_{2} CLOTH_{3} \n'.format(act_num, i, i, cur_cloth_n)
                    act_num += 1
                    ll_plan_str += '{0}: PUT_INTO_BASKET BAXTER CLOTH_{1} BASKET CLOTH_TARGET_END_{2} BASKET_NEAR_TARGET CLOTH_PUTDOWN_BEGIN_{3} CP_EE_{4} CLOTH_PUTDOWN_END_{5} \n'.format(act_num, cur_cloth_n, i, i, i, i)
                    act_num += 1
                    i += 1
                    cur_cloth_n += 1
                last_pose = 'CLOTH_PUTDOWN_END_{0}'.format(i-1)


            elif act_type == 'load_basket_from_region_2':
                init_i = i
                while i < num_cloths_r4 + init_i:
                    ll_plan_str += '{0}: ROTATE BAXTER {1} ROBOT_REGION_2_POSE_{2} REGION_2 \n'.format(act_num, last_pose, i)
                    act_num += 1
                    ll_plan_str += '{0}: MOVETO BAXTER ROBOT_REGION_2_POSE_{1} CLOTH_GRASP_BEGIN_{2} \n'.format(act_num, i, i)
                    act_num += 1
                    ll_plan_str += '{0}: CLOTH_GRASP BAXTER CLOTH_{1} CLOTH_TARGET_BEGIN_{2} CLOTH_GRASP_BEGIN_{3} CG_EE_{4} CLOTH_GRASP_END_{5} \n'.format(act_num, cur_cloth_n, i, i, i, i)
                    act_num += 1
                    ll_plan_str += '{0}: ROTATE_HOLDING_CLOTH BAXTER CLOTH CLOTH_GRASP_END_{1} ROBOT_REGION_2_POSE_{2} REGION_2 \n'.format(act_num, i, i)
                    act_num += 1
                    ll_plan_str += '{0}: MOVEHOLDING_CLOTH BAXTER ROBOT_REGION_2_POSE_{1} CLOTH_PUTDOWN_BEGIN_{2} CLOTH_{3} \n'.format(act_num, i, i, cur_cloth_n)
                    act_num += 1
                    ll_plan_str += '{0}: PUT_INTO_BASKET BAXTER CLOTH_{1} BASKET CLOTH_TARGET_END_{2} BASKET_NEAR_TARGET CLOTH_PUTDOWN_BEGIN_{3} CP_EE_{4} CLOTH_PUTDOWN_END_{5} \n'.format(act_num, cur_cloth_n, i, i, i, i)
                    act_num += 1
                    i += 1
                    cur_cloth_n += 1
                last_pose = 'CLOTH_PUTDOWN_END_{0}'.format(i-1)

            # The basket, when not near the washer, is in region 3
            elif act_type == 'load_basket_from_region_3':
                init_i = i
                ll_plan_str += '{0}: ROTATE BAXTER {1} ROBOT_REGION_3_POSE_{2} REGION_3 \n'.format(act_num, last_pose, i)
                act_num += 1
                while i < num_cloths_r3 + init_i:
                    ll_plan_str += '{0}: MOVETO BAXTER ROBOT_REGION_3_POSE_{1} CLOTH_GRASP_BEGIN_{2} \n'.format(act_num, i, i)
                    act_num += 1
                    ll_plan_str += '{0}: CLOTH_GRASP BAXTER CLOTH_{1} CLOTH_TARGET_BEGIN_{2} CLOTH_GRASP_BEGIN_{3} CG_EE_{4} CLOTH_GRASP_END_{5} \n'.format(act_num, cur_cloth_n, i, i, i, i)
                    act_num += 1
                    ll_plan_str += '{0}: MOVEHOLDING_CLOTH BAXTER ROBOT_REGION_2_POSE_{1} CLOTH_PUTDOWN_BEGIN_{2} CLOTH_{3} \n'.format(act_num, i, i, cur_cloth_n)
                    act_num += 1
                    ll_plan_str += '{0}: PUT_INTO_BASKET BAXTER CLOTH_{1} BASKET CLOTH_TARGET_END_{2} BASKET_NEAR_TARGET CLOTH_PUTDOWN_BEGIN_{3} CP_EE_{4} CLOTH_PUTDOWN_END_{5} \n'.format(act_num, cur_cloth_n, i, i, i, i)
                    act_num += 1
                    i += 1
                    cur_cloth_n += 1
                last_pose = 'CLOTH_PUTDOWN_END_{0}'.format(i-1)

            # TODO: Add right handed grasp functionality
            elif act_type == 'load_basket_from_region_4':
                init_i = i
                while i < num_cloths_r4 + init_i:
                    ll_plan_str += '{0}: ROTATE BAXTER {1} ROBOT_REGION_4_POSE_{2} REGION_4 \n'.format(act_num, last_pose, i)
                    act_num += 1
                    ll_plan_str += '{0}: MOVETO BAXTER ROBOT_REGION_4_POSE_{1} CLOTH_GRASP_BEGIN_{2} \n'.format(act_num, i, i)
                    act_num += 1
                    ll_plan_str += '{0}: CLOTH_GRASP BAXTER CLOTH_{1} CLOTH_TARGET_BEGIN_{2} CLOTH_GRASP_BEGIN_{3} CG_EE_{4} CLOTH_GRASP_END_{5} \n'.format(act_num, cur_cloth_n, i, i, i, i)
                    act_num += 1
                    ll_plan_str += '{0}: ROTATE_HOLDING_CLOTH BAXTER CLOTH CLOTH_GRASP_END_{1} ROBOT_REGION_2_POSE_{2} REGION_2 \n'.format(act_num, i, i)
                    act_num += 1
                    ll_plan_str += '{0}: MOVEHOLDING_CLOTH BAXTER ROBOT_REGION_2_POSE_{1} CLOTH_PUTDOWN_BEGIN_{2} CLOTH_{3} \n'.format(act_num, i, i, cur_cloth_n)
                    act_num += 1
                    ll_plan_str += '{0}: PUT_INTO_BASKET BAXTER CLOTH_{1} BASKET CLOTH_TARGET_END_{2} BASKET_NEAR_TARGET CLOTH_PUTDOWN_BEGIN_{3} CP_EE_{4} CLOTH_PUTDOWN_END_{5} \n'.format(act_num, cur_cloth_n, i, i, i, i)
                    act_num += 1
                    i += 1
                    cur_cloth_n += 1
                last_pose = 'CLOTH_PUTDOWN_END_{0}'.format(i-1)


            elif act_type == 'move_basket_to_washer':
                ll_plan_str += '{0}: ROTATE BAXTER {1} ROBOT_REGION_3_POSE_{2} REGION_3 \n'.format(act_num, last_pose, i)
                act_num += 1
                ll_plan_str += '{0}: MOVETO BAXTER ROBOT_REGION_3_POSE_{1} BASKET_GRASP_BEGIN_{2} \n'.format(act_num, i, i)
                act_num += 1
                ll_plan_str += '{0}: BASKET_GRASP_WITH_CLOTH BAXTER BASKET BASKET_FAR_TARGET BASKET_GRASP_BEGIN_{1} BG_EE_LEFT_{2} BG_EE_RIGHT_{3} BASKET_GRASP_END_{4} \n'.format(act_num, i, i, i)
                act_num += 1
                ll_plan_str += '{0}: ROTATE_HOLDING_BASKET_WITH_CLOTH BAXTER BASKET BASKET_GRASP_END_{1} ROBOT_REGION_1_POSE_{2} REGION_1 \n'.format(act_num, i, i)
                act_num += 1
                ll_plan_str += '{0}: MOVEHOLDING_BASKET_WITH_CLOTH BAXTER ROBOT_REGION_1_POSE_{1} BASKET_PUTDOWN_BEGIN_{2} BASKET \n'.format(act_num, i, i)
                act_num += 1
                ll_plan_str += '{0}: BASKET_PUTDOWN_WITH_CLOTH BAXTER BASKET BASKET_NEAR_TARGET BASKET_PUTDOWN_BEGIN_{1} BP_EE_LEFT_{2} BP_EE_RIGHT_{3} BASKET_PUTDOWN_END_{4} \n'.format(act_num, i, i, i)
                last_pose = 'BASKET_PUTDOWN_END_{0}'.format(i)
                i += 1


            elif act_type == 'move_basket_from_washer':
                ll_plan_str += '{0}: ROTATE BAXTER {1} ROBOT_REGION_1_POSE_{2} REGION_1 \n'.format(act_num, last_pose, i)
                act_num += 1
                ll_plan_str += '{0}: MOVETO BAXTER ROBOT_REGION_1_POSE_{1} BASKET_GRASP_BEGIN_{2} \n'.format(act_num, i, i)
                act_num += 1
                ll_plan_str += '{0}: BASKET_GRASP_WITH_CLOTH BAXTER BASKET BASKET_NEAR_TARGET BASKET_GRASP_BEGIN_{1} BG_EE_LEFT_{2} BG_EE_RIGHT_{3} BASKET_GRASP_END_{4} \n'.format(act_num, i, i, i)
                act_num += 1
                ll_plan_str += '{0}: ROTATE_HOLDING_BASKET_WITH_CLOTH BAXTER BASKET BASKET_GRASP_END_{1} ROBOT_REGION_3_POSE_{2} REGION_3 \n'.format(act_num, i, i)
                act_num += 1
                ll_plan_str += '{0}: MOVEHOLDING_BASKET_WITH_CLOTH BAXTER ROBOT_REGION_3_POSE_{1} BASKET_PUTDOWN_BEGIN_{2} BASKET \n'.format(act_num, i, i)
                act_num += 1
                ll_plan_str += '{0}: BASKET_PUTDOWN_WITH_CLOTH BAXTER BASKET BASKET_FAR_TARGET BASKET_PUTDOWN_BEGIN_{1} BP_EE_LEFT_{2} BP_EE_RIGHT_{3} BASKET_PUTDOWN_END_{4} \n'.format(act_num, i, i, i)
                last_pose = 'BASKET_PUTDOWN_END_{0}'.format(i)
                i += 1


            elif act_type == 'open_washer':
                ll_plan_str += '{0}: MOVETO BAXTER {1} OPEN_DOOR_BEGIN_{2} \n'.format(act_num, last_pose, i)
                act_num += 1
                ll_plan_str += '{0}: OPEN_DOOR BAXTER WASHER OPEN_DOOR_BEGIN_{1} OPEN_DOOR_EE_APPROACH_{2} OPEN_DOOR_EE_RETEREAT_{3} OPEN_DOOR_END_{4} WASHER_OPEN_POSE_{5} WASHER_CLOSE_POSE_{6} \n'.format(act_num, i, i, i, i, i, i)
                act_num += 1
                i += 1


            elif act_type == 'close_washer':
                ll_plan_str += '{0}: MOVETO BAXTER {1} CLOSE_DOOR_BEGIN_{2} \n'.format(act_num, last_pose, i)
                act_num += 1
                ll_plan_str += '{0}: CLOSE_DOOR BAXTER WASHER CLOSE_DOOR_BEGIN_{1} CLOSE_DOOR_EE_APPROACH_{2} CLOSE_DOOR_EE_RETEREAT_{3} CLOSE_DOOR_END_{4} WASHER_OPEN_POSE_{5} WASHER_CLOSE_POSE_{6} \n'.format(act_num, i, i, i, i, i, i)
                act_num += 1
                i += 1


            elif act_type == 'load_washer':
                init_i = i
                cur_cloth_n = 0
                ll_plan_str += '{0}: ROTATE BAXTER {1} ROBOT_REGION_1_POSE_{2} REGION_1 \n'.format(act_num, last_pose, i)
                act_num += 1
                while i < num_cloths + init_i:
                    ll_plan_str += '{0}: MOVETO BAXTER ROBOT_REGION_1_POSE_{1} CLOTH_GRASP_BEGIN_{2} \n'.format(act_num, i, i)
                    act_num += 1
                    ll_plan_str += '{0}: CLOTH_GRASP BAXTER CLOTH_{1} CLOTH_TARGET_BEGIN_{2} CLOTH_GRASP_BEGIN_{3} CG_EE_{4} CLOTH_GRASP_END_{5} \n'.format(act_num, cur_cloth_n, i, i, i, i)
                    act_num += 1
                    ll_plan_str += '{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_{1} CLOTH_PUTDOWN_BEGIN_{2} CLOTH_{3} \n'.format(act_num, i, i, cur_cloth_n)
                    act_num += 1
                    ll_plan_str += '{0}: PUT_INTO_WASHER BAXTER WASHER WASHER_POSE_{1} CLOTH_{2} CLOTH_TARGET_END_{3} CLOTH_PUTDOWN_BEGIN_{4} CP_EE_{5} CLOTH_PUTDOWN_END_{6} \n'.format(act_num, i, cur_cloth_n, i, i, i, i)
                    act_num += 1
                    i += 1
                    cur_cloth_n += 1
                i -= 1
                ll_plan_str += '{0}: MOVETO BAXTER CLOTH_PUTDOWN_END_{1} CLOSE_DOOR_BEGIN_{2} \n'.format(act_num, i, i)
                act_num += 1
                ll_plan_str += '{0}: CLOSE_DOOR BAXTER WASHER CLOSE_DOOR_BEGIN_{1} CLOSE_DOOR_EE_APPROACH_{2} CLOSE_DOOR_EE_RETEREAT_{3} CLOSE_DOOR_END_{4} WASHER_OPEN_POSE_{5} WASHER_CLOSE_POSE_{6} \n'.format(act_num, i, i, i, i, i, i)
                act_num += 1
                last_pose = 'CLOSE_DOOR_END_{0}'.format(i)
                i += 1


            elif act_type == 'unload_washer':
                init_i = i
                cur_cloth_n = 0
                ll_plan_str += '{0}: ROTATE BAXTER {1} ROBOT_REGION_1_POSE_{2} REGION_1 \n'.format(act_num, last_pose, i)
                act_num += 1
                while i < num_cloths + init_i:
                    ll_plan_str += '{0}: MOVETO BAXTER ROBOT_REGION_1_POSE_{1} CLOTH_GRASP_BEGIN_{2} \n'.format(act_num, i, i)
                    act_num += 1
                    ll_plan_str += '{0}: TAKE_OUT_OF_WASHER BAXTER WASHER WASHER_POSE_{1} CLOTH_{2} CLOTH_TARGET_END_{3} CLOTH_GRASP_BEGIN_{4} CP_EE_{5} CLOTH_GRASP_END_{6} \n'.format(act_num, i, cur_cloth_n, i, i, i, i)
                    act_num += 1
                    ll_plan_str += '{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_{1} CLOTH_PUTDOWN_BEGIN_{2} CLOTH_{3} \n'.format(act_num, i, i, cur_cloth_n)
                    act_num += 1
                    ll_plan_str += '{0}: PUT_INTO_BASKET BAXTER CLOTH_{1} BASKET CLOTH_TARGET_END_{2} BASKET_NEAR_TARGET CLOTH_PUTDOWN_BEGIN_{3} CP_EE_{4} CLOTH_PUTDOWN_END_{5} \n'.format(act_num, cur_cloth_n, i, i, i, i)
                    act_num += 1
                    i += 1
                    cur_cloth_n += 1
                i -= 1
                ll_plan_str += '{0}: MOVETO BAXTER CLOTH_PUTDOWN_END_{1} CLOSE_DOOR_BEGIN_{2} \n'.format(act_num, i, i)
                act_num += 1
                ll_plan_str += '{0}: CLOSE_DOOR BAXTER WASHER CLOSE_DOOR_BEGIN_{1} CLOSE_DOOR_EE_APPROACH_{2} CLOSE_DOOR_EE_RETEREAT_{3} CLOSE_DOOR_END_{4} WASHER_OPEN_POSE_{5} WASHER_CLOSE_POSE_{6} \n'.format(act_num, i, i, i, i, i, i)
                act_num += 1
                last_pose = 'CLOSE_DOOR_END_{0}'.format(i)
                i += 1


        ll_plan_str += '{0}: MOVETO BAXTER {1} ROBOT_END_POSE \n'.format(act_num, last_pose)

        return ll_plan_str

    def _get_action_type(self, action):
        pass
