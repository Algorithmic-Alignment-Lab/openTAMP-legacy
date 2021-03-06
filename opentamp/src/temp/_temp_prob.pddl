; AUTOGENERATED. DO NOT EDIT.

(define (problem ff_prob)
(:domain robotics)
(:objects
can0_init_target - Target
can0 - Can
can0_end_target - Target
can1_init_target - Target
can1 - Can
can1_end_target - Target
end_target_0 - Target
end_target_1 - Target
end_target_2 - Target
end_target_3 - Target
end_target_4 - Target
end_target_5 - Target
end_target_6 - Target
end_target_7 - Target
pr2 - Robot
grasp0 - Grasp
grasp1 - Grasp
grasp2 - Grasp
grasp3 - Grasp
robot_init_pose - RobotPose
robot_end_pose - RobotPose
obs0 - Obstacle
)

(:init
(AtInit can1 can1_init_target )
(StationaryW obs0 )
(Near can1 can1_init_target )
(AtInit can0 can0_end_target )
(AtInit can1 can1_end_target )
(At can0 can0_init_target )
(IsMP pr2 )
(At can1 can1_init_target )
(AtInit can0 can0_init_target )
(RobotAt pr2 robot_init_pose )
(Near can0 can0_init_target )
)

(:goal
(and (and (Near can0 end_target_0)(Near can1 end_target_1)))
)
)