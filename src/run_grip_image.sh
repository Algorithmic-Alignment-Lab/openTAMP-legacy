for N in 1 2 3 4 5
do
    for S in third
    do

        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v98 \
                                                       -no 2 -nt 2 -spl -llus 10000  -hlus 10000 \
                                                       -ff 1. -hln 3 -mask -hldim 64 -lldim 64 \
                                                       -retime -vel 0.5 -eta 5 \
                                                       -obs_del -hist_len 3 \
                                                       -fail -failmode random \
                                                       -prim_first_wt 10 -lr 0.0002 \
                                                       -hllr 0.0002 -lldec 0.0001 -hldec 0.000 \
                                                       -add_noop 4 --permute_hl 0 \
                                                       -post -render -hl_image \
                                                       -imwidth 64 -imheight 64 \
                                                       -expl_wt 10 -expl_eta 4 \
                                                       -col_coeff 0.05 \
                                                       -motion 32 \
                                                       -task 4 \
                                                       -rollout 8 \
                                                       -roll_hl \
                                                       -descr fixed_conttask_grip_images_64_hl & 

        sleep 5h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s


        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v98 \
                                                       -no 2 -nt 2 -spl -llus 10000  -hlus 50000 \
                                                       -ff 1. -hln 2 -mask -hldim 64 -lldim 64 \
                                                       -retime -vel 0.4 -eta 2 \
                                                       -obs_del -hist_len 1 \
                                                       -fail -failmode random \
                                                       -prim_first_wt 20 -lr 0.0002 \
                                                       -hllr 0.0002 -lldec 0.0001 -hldec 0.000 \
                                                       -add_noop 1 --permute_hl 0 \
                                                       -post -render -hl_image \
                                                       -imwidth 64 -imheight 64 \
                                                       -expl_wt 20 -expl_eta 4 \
                                                       -col_coeff 0.1 \
                                                       -motion 34 \
                                                       -task 4 \
                                                       -rollout 10 \
                                                       -roll_hl \
                                                       -warm 300 \
                                                       -descr moredata_conttask_grip_images_64_hl & 

        sleep 5h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s

    done
done

