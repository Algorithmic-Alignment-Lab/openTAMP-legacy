dom_str = """
# AUTOGENERATED. DO NOT EDIT.
# Configuration file for CAN domain. Blank lines and lines beginning with # are filtered out.

# implicity, all types require a name
Types: Can, Target, RobotPose, Robot, Grasp, Obstacle, Rotation

# Define the class location of each non-standard attribute type used in the above parameter type descriptions.

Attribute Import Paths: RedCircle core.util_classes.items, BlueCircle core.util_classes.items, GreenCircle core.util_classes.items, Vector1d core.util_classes.matrix, Vector2d core.util_classes.matrix, Wall core.util_classes.items, TwoLinkArm core.util_classes.robots

Predicates Import Path: core.util_classes.namo_arm_predicates

"""

prim_pred_str = 'Primitive Predicates: geom, Can, RedCircle; pose, Can, Vector2d; geom, Target, BlueCircle; value, Target, Vector2d; pose, Robot, Vector2d; value, RobotPose, Vector2d; gripper, RobotPose, Vector1d; geom, RobotPose, BlueCircle; geom, Robot, TwoLinkArm; joint1, Robot, Vector1d; joint2, Robot, Vector1d; wrist, Robot, Vector1d; joint1, RobotPose, Vector1d; joint2, RobotPose, Vector1d; wrist, RobotPose, Vector1d; gripper, Robot, Vector1d; value, Grasp, Vector2d; geom, Obstacle, Wall; pose, Obstacle, Vector2d; value, Rotation, Vector1d' 
dom_str += prim_pred_str + '\n\n'

der_pred_str = 'Derived Predicates: At, Can, Target; AtInit, Can, Target; InGripper, Robot, Can, Grasp; Obstructs, Robot, Can, Can, Can; ObstructsHolding, Robot, Target, Target, Can, Can; WideObstructsHolding, Robot, Target, Target, Can, Can; StationaryWrist, Robot; StationaryRot, Robot; Stationary, Can; RobotStationary, Robot; StationaryNEq, Can, Can; IsMP, Robot; GripperClosed, Robot; Near, Can, Target; WideObstructs, Robot, Can, Can, Can; AtNEq, Can, Can, Target; TargetCanGraspCollides, Target, Can, Grasp; HLAtGrasp, Robot, Can, Grasp; InGraspAngle, Robot, Can; ApproachGraspAngle, Robot, Can;  NearGraspAngle, Robot, Can; ThetaDirValid, Robot; ForThetaDirValid, Robot; RevThetaDirValid, Robot; ScalarVelValid, Robot; HLGraspFailed, Can; HLTransferFailed, Can, Target; HLPlaceFailed, Target; RobotInBounds, Robot'
dom_str += der_pred_str + '\n'

dom_str += """

# The first set of parentheses after the colon contains the
# parameters. The second contains preconditions and the third contains
# effects. This split between preconditions and effects is only used
# for task planning purposes. Our system treats all predicates
# similarly, using the numbers at the end, which specify active
# timesteps during which each predicate must hold

"""

class Action(object):
    def __init__(self, name, timesteps, pre=None, post=None):
        pass

    def to_str(self):
        time_str = ''
        cond_str = '(and '
        for pre, timesteps in self.pre:
            cond_str += pre + ' '
            time_str += timesteps + ' '
        cond_str += ')'

        cond_str += '(and '
        for eff, timesteps in self.eff:
            cond_str += eff + ' '
            time_str += timesteps + ' '
        cond_str += ')'

        return "Action " + self.name + ' ' + str(self.timesteps) + ': ' + self.args + ' ' + cond_str + ' ' + time_str

class MoveTo(Action):
    def __init__(self):
        self.name = 'moveto'
        self.timesteps = 25
        et = self.timesteps - 1
        self.args = '(?robot - Robot ?can - Can ?target - Target ?end - Target)' 
        self.pre = [\
                ('(At ?can ?target)', '0:0'),
                #('(forall (?gr - Grasp) (forall (?obj - Can) (not (NearGraspAngle ?robot ?obj))))', '0:-1'),
                ('(not (GripperClosed ?robot))', '1:{0}'.format(et-1)),
                ('(forall (?obj - Can) (Stationary ?obj))', '0:{0}'.format(et-1)),
                ('(IsMP ?robot)', '0:{0}'.format(et-1)),
                ('(RobotInBounds ?robot)', '0:{0}'.format(et)),
                ('(ApproachGraspAngle ?robot ?can)', '{0}:{1}'.format(et-2, et-2)),
                ('(forall (?obj - Can) (not (Obstructs ?robot ?can ?can ?obj)))', '0:-1'),
                ('(forall (?obj - Can) (not (WideObstructs ?robot ?can ?can ?obj)))', '0:-1'),
                ('(forall (?obj - Can) (not (WideObstructs ?robot ?can ?can ?obj)))', '1:{0}'.format(et-4)),
                ('(forall (?obj - Can) (not (Obstructs ?robot ?can ?can ?obj)))', '{0}:{1}'.format(et-3, et-3)),
                ('(forall (?obj - Can) (not (ObstructsHolding ?robot ?target ?target ?obj ?can)))', '{0}:{1}'.format(et-3, et-1)),
        ]
        self.eff = [\
                ('(InGraspAngle ?robot ?can)', '{0}:{1}'.format(et, et)),
                ('(forall (?obj - Can / ?can) (not (NearGraspAngle ?robot ?obj)))', '{0}:{1}'.format(et, et-1)),
                ('(forall (?obj - Can) (Stationary ?obj))', '{0}:{1}'.format(et, et-1)),
                #('(StationaryRot ?robot)', '{0}:{1}'.format(et-2, et-1)),
                ('(RobotStationary ?robot)', '{0}:{0}'.format(et-1)),
        ]

class Transfer(Action):
    def __init__(self):
        self.name = 'transfer'
        self.timesteps = 25
        et = self.timesteps - 1
        self.args = '(?robot - Robot ?c - Can ?t - Target ?init - Target)'
        self.pre = [\
                ('(At ?c ?init)', '0:0'),
                ('(forall (?obj - Can) (not (AtInit ?obj ?t)))', '0:-1'),
                ('(RobotInBounds ?robot)', '0:{0}'.format(et)),
                ('(RobotStationary ?robot)', '0:0'),
                ('(InGraspAngle ?robot ?c)', '0:0'),
                ('(forall (?obj - Can) (not (Near ?obj ?t)))', '0:0'),
                ('(GripperClosed ?robot)', '1:{0}'.format(et-2)),
                ('(not (GripperClosed ?robot))', '{0}:{0}'.format(et-1)),
                ('(InGraspAngle ?robot ?c)', '{0}:{1}'.format(et-2, et-1)),
                ('(RobotStationary ?robot)', '{0}:{0}'.format(et-2)),
                ('(forall (?obj - Can) (not (ObstructsHolding ?robot ?t ?t ?obj ?c)))', '0:{0}'.format(0)),
                ('(forall (?obj - Can) (not (WideObstructsHolding ?robot ?t ?t ?obj ?c)))', '2:{0}'.format(et-2)),
                ('(forall (?obj - Can) (not (ObstructsHolding ?robot ?t ?t ?obj ?c)))', '1:{0}'.format(et-2)),
                ('(forall (?obj - Can) (not (ObstructsHolding ?robot ?t ?t ?obj ?c)))', '{0}:{0}'.format(et-1)),
                ('(forall (?obj - Can) (StationaryNEq ?obj ?c))', '0:{0}'.format(et-1)), 
                ('(IsMP ?robot)', '0:{0}'.format(et-1)),
                #('(StationaryWrist ?robot)', '{0}:{0}'.format(et-1)),
               ]

        self.eff = [\
                ('(NearGraspAngle ?robot ?c)', '{0}:{1}'.format(et, et)),
                ('(forall (?obj - Can / ?c) (not (NearGraspAngle ?robot ?obj)))', '{0}:{1}'.format(et, et-1)),
                ('(At ?c ?t)', '{0}:{1}'.format(et-2, et)),
                ('(forall (?obj - Can / ?c) (not (NearGraspAngle ?robot ?obj)))', '{0}:{1}'.format(et, et-1)),
                ('(Near ?c ?t)', '{0}:{0}'.format(et)),
                ('(not (Near ?c ?init))', '{0}:{1}'.format(et, et-1)),
                ('(not (At ?c ?init))', '{0}:{1}'.format(et, et-1)),
                ('(forall (?obj - Can) (not (Obstructs ?robot ?obj ?obj ?c)))', '{0}:{1}'.format(et, et-1)),
                ('(forall (?obj - Can) (not (WideObstructs ?robot ?obj ?obj ?c)))', '{0}:{1}'.format(et, et-1)),
                ('(forall (?obj - Can) (forall (?targ - Target) (not (WideObstructsHolding ?robot ?targ ?targ ?c ?obj))))', '{0}:{1}'.format(et, et-1)),
                ('(forall (?obj - Can) (forall (?targ - Target) (not (ObstructsHolding ?robot ?targ ?targ ?c ?obj))))', '{0}:{1}'.format(et, et-1)),
                ('(forall (?obj - Can) (Stationary ?obj))', '{0}:{1}'.format(et, et-1)),
        ]

actions = [MoveTo(), Transfer()]
for action in actions:
    dom_str += '\n\n'
    dom_str += action.to_str()

# removes all the extra spaces
dom_str = dom_str.replace('            ', '')
dom_str = dom_str.replace('    ', '')
dom_str = dom_str.replace('    ', '')

print(dom_str)
f = open('namo_current_arm.domain', 'w')
f.write(dom_str)
