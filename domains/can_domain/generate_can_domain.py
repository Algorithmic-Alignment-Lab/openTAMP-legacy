import sys
sys.path.insert(0, '../../src/')
from core.util_classes.pr2_predicates import EEREACHABLE_STEPS

dom_str = """
# AUTOGENERATED. DO NOT EDIT.
# Configuration file for CAN domain. Blank lines and lines beginning with # are filtered out.

# implicity, all types require a name
Types: Can, Target, RobotPose, Robot, EEPose, Obstacle

# Define the class location of each non-standard attribute type used in the above parameter type descriptions.
Attribute Import Paths: RedCan core.util_classes.can, BlueCan core.util_classes.can, PR2 core.util_classes.pr2, Vector1d core.util_classes.matrix, Vector3d core.util_classes.matrix, PR2ArmPose core.util_classes.matrix, Table core.util_classes.table, Box core.util_classes.box

Predicates Import Path: core.util_classes.pr2_predicates

"""

class PrimitivePredicates(object):
    def __init__(self):
        self.attr_dict = {}

    def add(self, name, attrs):
        self.attr_dict[name] = attrs

    def get_str(self):
        prim_str = 'Primitive Predicates: '
        first = True
        for name, attrs in self.attr_dict.iteritems():
            for attr_name, attr_type in attrs:
                pred_str = attr_name + ', ' + name + ', ' + attr_type
                if first:
                    prim_str += pred_str
                    first = False
                else:
                    prim_str += '; ' + pred_str
        return prim_str

pp = PrimitivePredicates()
pp.add('Can', [('geom', 'RedCan'), ('pose', 'Vector3d'), ('rotation', 'Vector3d')])
pp.add('Target', [('geom', 'BlueCan'), ('value', 'Vector3d'), ('rotation', 'Vector3d')])
pp.add('RobotPose', [('value', 'Vector3d'),
                    ('backHeight', 'Vector1d'),
                    ('lArmPose', 'PRcan_1234_02ArmPose'),
                    ('rGripper', 'Vector1d')])
pp.add('Robot', [('geom', 'PR2'),
                ('pose', 'Vector3d'),
                ('backHeight', 'Vector1d'),
                ('lArmPose', 'PR2ArmPose'),
                ('lGripper', 'Vector1d'),
                ('rArmPose', 'PR2ArmPose'),
                ('rGripper', 'Vector1d')])
pp.add('EEPose', [('value', 'Vector3d'), ('rotation', 'Vector3d')])
# pp.add('Grasp', [('value', 'Vector2d')])
pp.add('Obstacle', [('geom', 'Box'), ('pose', 'Vector3d'), ('rotation', 'Vector3d')])
dom_str += pp.get_str() + '\n\n'

class DerivatedPredicates(object):
    def __init__(self):
        self.pred_dict = {}

    def add(self, name, args):
        self.pred_dict[name] = args

    def get_str(self):
        prim_str = 'Derived Predicates: '

        first = True
        for name, args in self.pred_dict.iteritems():
            pred_str = name
            for arg in args:
                pred_str += ', ' + arg

            if first:
                prim_str += pred_str
                first = False
            else:
                prim_str += '; ' + pred_str
        return prim_str

dp = DerivatedPredicates()
dp.add('At', ['Can', 'Target'])
dp.add('RobotAt', ['Robot', 'RobotPose'])
dp.add('EEReachable', ['Robot', 'RobotPose', 'EEPose'])
dp.add('EEReachableRot', ['Robot', 'RobotPose', 'EEPose'])
dp.add('InGripper', ['Robot', 'Can'])
dp.add('InGripperRot', ['Robot', 'Can'])
dp.add('InContact', ['Robot', 'EEPose', 'Target'])
dp.add('Obstructs', ['Robot', 'RobotPose', 'RobotPose', 'Can'])
dp.add('ObstructsHolding', ['Robot', 'RobotPose', 'RobotPose', 'Can', 'Can'])
dp.add('GraspValid', ['EEPose', 'Target'])
dp.add('GraspValidRot', ['EEPose', 'Target'])
dp.add('Stationary', ['Can'])
dp.add('StationaryW', ['Obstacle'])
dp.add('StationaryNEq', ['Can', 'Can'])
dp.add('StationaryArms', ['Robot'])
dp.add('StationaryBase', ['Robot'])
dp.add('IsMP', ['Robot'])
dp.add('WithinJointLimit', ['Robot'])
dp.add('Collides', ['Can', 'Obstacle'])
dp.add('RCollides', ['Robot', 'Obstacle'])

dom_str += dp.get_str() + '\n'

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

class Move(Action):
    def __init__(self):
        self.name = 'moveto'
        self.timesteps = 30
        end = self.timesteps - 1
        self.args = '(?robot - Robot ?start - RobotPose ?end - RobotPose)'
        self.pre = [\
            ('(forall (?c - Can)\
                (not (InGripper ?robot ?c))\
            )', '0:0'),
            ('(forall (?c - Can)\
                (not (InGripperRot ?robot ?c))\
            )', '0:0'),
            ('(RobotAt ?robot ?start)', '0:0'),
            ('(forall (?obj - Can )\
                (not (Obstructs ?robot ?start ?end ?obj)))', '0:{}'.format(end-1)),
            ('(forall (?obj - Can)\
                (Stationary ?obj))', '0:{}'.format(end-1)),
            ('(forall (?w - Obstacle) (StationaryW ?w))', '0:{}'.format(end-1)),
            ('(StationaryArms ?robot)', '0:{}'.format(end-1)),
            ('(StationaryW ?w)', '0:{}'.format(end-1)),
            ('(IsMP ?robot)', '0:{}'.format(end-1)),
            ('(WithinJointLimit ?robot)', '0:{}'.format(end)),
            # ('(forall (?w     - Obstacle)\
            #     (forall (?obj - Can)\
            #         (not (Collides ?obj ?w))\
            #     ))','0:19')
            ('(forall (?w - Obstacle) (not (RCollides ?robot ?w)))', '0:{}'.format(end))
        ]
        self.eff = [\
            ('(not (RobotAt ?robot ?start))', '{}:{}'.format(end, end)),
            ('(RobotAt ?robot ?end)', '{}:{}'.format(end, end))]

class MoveHolding(Action):
    def __init__(self):
        self.name = 'movetoholding'
        self.timesteps = 20
        self.args = '(?robot - Robot ?start - RobotPose ?end - RobotPose ?c - Can)'
        self.pre = [\
            ('(RobotAt ?robot ?start)', '0:0'),
            ('(InGripper ?robot ?c)', '0:19'),
            ('(InGripperRot ?robot ?c)', '0:19'),
            # ('(forall (?obj - Can)\
            #     (not (ObstructsHolding ?robot ?start ?end ?obj ?c))\
            # )', '0:19'),
            ('(forall (?obj - Can) (StationaryNEq ?obj ?c))', '0:18'),
            ('(forall (?w - Obstacle) (StationaryW ?w))', '0:18'),
            ('(StationaryArms ?robot)', '0:18'),
            ('(IsMP ?robot)', '0:18'),
            ('(WithinJointLimit ?robot)', '0:19')
            # ('(forall (?w - Obstacle)\
            #     (forall (?obj - Can)\
            #         (not (Collides ?obj ?w))\
            #     )\
            # )', '0:19')
            # ('(forall (?w - Obstacle) (not (RCollides ?robot ?w)))', '0:19')
        ]
        self.eff = [\
            ('(not (RobotAt ?robot ?start))', '19:19'),
            ('(RobotAt ?robot ?end)', '19:19')
        ]

class Grasp(Action):
    def __init__(self):
        self.name = 'grasp'
        self.timesteps = 40
        self.args = '(?robot - Robot ?can - Can ?target - Target ?sp - RobotPose ?ee - EEPose ?ep - RobotPose)'
        steps = EEREACHABLE_STEPS
        grasp_time = self.timesteps/2
        approach_time = grasp_time-steps
        retreat_time = grasp_time+steps
        self.pre = [\
            ('(At ?can ?target)', '0:0'),
            ('(RobotAt ?robot ?sp)', '0:0'),
            ('(EEReachable ?robot ?sp ?ee)', '{}:{}'.format(grasp_time, grasp_time)),
            # ('(EEReachableRot ?robot ?sp ?ee)', '{}:{}'.format(grasp_time, grasp_time)),
            ('(EEReachableRot ?robot ?sp ?ee)', '{}:{}'.format(approach_time, retreat_time)),
            # ('(EEReachableRot ?robot ?sp ?ee)', '16:24'),
            # TODO: not sure why InContact to 39 fails
            ('(InContact ?robot ?ee ?target)', '{}:38'.format(grasp_time)),
            ('(GraspValid ?ee ?target)', '{}:{}'.format(grasp_time, grasp_time)),
            ('(GraspValidRot ?ee ?target)', '{}:{}'.format(grasp_time, grasp_time)),
            ('(forall (?obj - Can)\
                (not (InGripper ?robot ?obj))\
            )', '0:0'),
            ('(forall (?obj - Can)\
                (not (InGripperRot ?robot ?obj))\
            )', '0:0'),
            ('(forall (?obj - Can) \
                (Stationary ?obj)\
            )', '0:{}'.format(grasp_time-1)),
            ('(forall (?obj - Can) (StationaryNEq ?obj ?can))', '{}:38'.format(grasp_time)),
            ('(forall (?w - Obstacle)\
                (StationaryW ?w)\
            )', '0:38'),
            # ('(StationaryBase ?robot)', '17:22'),
            ('(StationaryBase ?robot)', '{}:{}'.format(approach_time, retreat_time-1)),
            # ('(StationaryBase ?robot)', '0:38'),
            # ('(IsMP ?robot)', '0:38'),
            ('(WithinJointLimit ?robot)', '0:39'),
            # ('(forall (?w - Obstacle)\
            #     (forall (?obj - Can)\
            #         (not (Collides ?obj ?w))\
            #     )\
            # )', '0:38'),
            ('(forall (?w - Obstacle)\
                (not (RCollides ?robot ?w))\
            )', '0:39'),
            ('(forall (?obj - Can)\
                (not (Obstructs ?robot ?sp ?ep ?obj))\
            )', '0:{}'.format(approach_time)),
            ('(forall (?obj - Can)\
                (not (ObstructsHolding ?robot ?sp ?ep ?obj ?can))\
            )', '{}:39'.format(approach_time+1))
        ]
        self.eff = [\
            ('(not (At ?can ?target))', '39:38') ,
            ('(not (RobotAt ?robot ?sp))', '39:38'),
            ('(RobotAt ?robot ?ep)', '39:39'),
            ('(InGripper ?robot ?can)', '{}:39'.format(grasp_time+1)),
            ('(InGripperRot ?robot ?can)', '{}:39'.format(grasp_time+1)),
            ('(forall (?sym1 - RobotPose)\
                (forall (?sym2 - RobotPose)\
                    (not (Obstructs ?robot ?sym1 ?sym2 ?can))\
                )\
            )', '39:38'),
            ('(forall (?sym1 - Robotpose)\
                (forall (?sym2 - RobotPose)\
                    (forall (?obj - Can) (not (ObstructsHolding ?robot ?sym1 ?sym2 ?can ?obj)))\
                )\
            )', '39:38')
        ]

class Putdown(Action):
    def __init__(self):
        self.name = 'putdown'
        self.timesteps = 20
        self.args = '(?robot - Robot ?can - Can ?target - Target ?sp - RobotPose ?ee - EEPose ?ep - RobotPose)'
        self.pre = [\
            ('(RobotAt ?robot ?sp)', '0:0'),
            ('(EEReachable ?robot ?sp ?ee)', '10:10'),
            ('(EEReachableRot ?robot ?sp ?ee)', '10:10'),
            ('(InContact ?robot ?ee ?target)', '0:10'),
            ('(GraspValid ?ee ?target)', '0:0'),
            ('(GraspValidRot ?ee ?target)', '0:0'),
            ('(InGripper ?robot ?can)', '0:10'),
            ('(InGripperRot ?robot ?can)', '0:10'),
            # is the line below so that we have to use a new ee with the target?
            # ('(not (InContact ?robot ?ee ?target))', '0:0'),
            ('(forall (?obj - Can)\
                (Stationary ?obj)\
            )', '10:18'),
            ('(forall (?obj - Can) (StationaryNEq ?obj ?can))', '0:9'),
            ('(forall (?w - Obstacle)\
                (StationaryW ?w)\
            )', '0:18'),
            ('(StationaryBase ?robot)', '0:18'),
            ('(IsMP ?robot)', '0:18'),
            ('(WithinJointLimit ?robot)', '0:19')
            # ('(forall (?w - Obstacle)\
            #     (forall (?obj - Can)\
            #         (not (Collides ?obj ?w))\
            #     )\
            # )', '0:18'),
            # ('(forall (?w - Obstacle)\
            #     (not (RCollides ?robot ?w))\
            # )', '0:19'),
            # ('(forall (?obj - Can)\
            #     (not (ObstructsHolding ?robot ?sp ?ep ?obj ?can))\
            # )', '0:19'),
            # ('(forall (?obj - Can)\
            #     (not (Obstructs ?robot ?sp ?ep ?obj))\
            # )', '19:19')
        ]
        self.eff = [\
            ('(not (RobotAt ?robot ?sp))', '19:19'),
            ('(RobotAt ?robot ?ep)', '19:19'),
            ('(At ?can ?target)', '19:19'),
            ('(not (InGripper ?robot ?can))', '19:19'),
            ('(not (InGripperRot ?robot ?can))', '19:19')
        ]

actions = [Move(), MoveHolding(), Grasp(), Putdown()]
for action in actions:
    dom_str += '\n\n'
    dom_str += action.to_str()

# removes all the extra spaces
dom_str = dom_str.replace('            ', '')
dom_str = dom_str.replace('    ', '')
dom_str = dom_str.replace('    ', '')

print dom_str
f = open('can.domain', 'w')
f.write(dom_str)