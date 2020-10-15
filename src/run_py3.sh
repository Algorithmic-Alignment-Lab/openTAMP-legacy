for N in 1 2 3 4 5
do
    for S in third
    do

        python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v85 -no 2 -nt 2 -spl -llus 10000  -hlus 5000  -ff 1. -retime -hln 2 -hldim 64 -lldim 64 -eta 4 -obs_del -hist_len 2 -prim_first_wt 10 -lr 0.0005 -hllr 0.0005 -lldec 0.0001 -hldec 0.001 -add_noop 5 --permute_hl 1 -fail -failmode random -expl_wt 10 -expl_eta 3 -expl_n 15 -expl_suc 8 -expl_m 4 -col_coeff 0. -descr obsdeltas_polresample_N15_s8_m4 & 
        sleep 5h
        pkill -f run_train -9
        pkill -f ros -9
        sleep 5s

    done
done

