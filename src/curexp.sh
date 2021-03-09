for N in 1 2 3 4 5
do
    for S in third
    do

        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v98 \
                                                       -no 2 -nt 2 -spl -llus 10000 -hlus 10000 \
                                                       -ff 1. -hln 2 -mask -hldim 48 -lldim 64 \
                                                       -retime -vel 0.3 -eta 7 -softev \
                                                       -obs_del -hist_len 1 \
                                                       -prim_first_wt 20 -lr 0.0002 \
                                                       -lr_policy adaptive \
                                                       -hllr 0.0004 -lldec 0.000 -hldec 0.00 \
                                                       -add_noop 2 --permute_hl 1 \
                                                       -render -hl_image \
                                                       -imwidth 96 -imheight 96 \
                                                       -expl_wt 10 -expl_eta 4 \
                                                       -col_coeff 0.1 \
                                                       -motion 32 \
                                                       -task 4 \
                                                       -rollout 12 \
                                                       -pre -post -mid \
                                                       -roll_hl \
                                                       -warm 200 \
                                                       -descr verify_redo_annotate_mon_image_hl & 

        sleep 5h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s

    done
done

