for N in 1 2 3 4 5
do
    for S in third
    do

        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v98 \
                                                       -no 2 -nt 2 -spl -llus 1000 -hlus 5000 \
                                                       -ff 1. -hln 2 -mask -hldim 64 -lldim 64 \
                                                       -retime -vel 0.3 -eta 7 -softev \
                                                       -obs_del -hist_len 2 \
                                                       -fail -failmode random \
                                                       -end2end 0.25 \
                                                       -prim_first_wt 20 -lr 0.0002 \
                                                       -lr_policy adaptive \
                                                       -hllr 0.001 -lldec 1. -hldec 1. \
                                                       -add_noop 0 --permute_hl 1 \
                                                       -post -pre -mid \
                                                       -render -hl_image \
                                                       -imwidth 96 -imheight 96 \
                                                       -expl_wt 10 -expl_eta 4 \
                                                       -col_coeff 0.05 \
                                                       -motion 32 \
                                                       -task 8 \
                                                       -rollout 12 \
                                                       -roll_hl \
                                                       -warm 200 \
                                                       -descr end2end_adaptive_reg & 

        sleep 6h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s

    done
done

