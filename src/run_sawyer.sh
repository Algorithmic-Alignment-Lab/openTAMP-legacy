for N in 1 2 3 4 5
do
    for S in third
    do

        python3.6 -W ignore policy_hooks/run_training.py -c policy_hooks.robosuite.grasp_hyp \
                                                       -no 1 -nt 1 -llus 1000  -hlus 1000 \
                                                       -ff 1. -mask -hln 2 -hldim 64 -lldim 96 \
                                                       -eta 5 \
                                                       -goal_type grasp \
                                                       -vel 0.02 \
                                                       -obs_del -hist_len 1 -prim_first_wt 1 -lr 0.001 \
                                                       -hllr 0.001 -lldec 0.001 -hldec 0.000 \
                                                       -add_noop 5 --permute_hl 0 \
                                                       -post -pre -mid \
                                                       -expl_wt 10 -expl_eta 4 \
                                                       -motion 16 \
                                                       -task 2 \
                                                       -rollout 16 \
                                                       -roll_hl \
                                                       --load_render \
                                                       -warm 100 \
                                                       -descr sun_sawyer_grasp & 
        sleep 5h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s

    done
done

