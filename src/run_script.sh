for N in 1 2 3 4
do
    python -W ignore policy_hooks/run_training.py -c namo.hyperparams_v20 -no 1 -nt 1 -cur 10 -ncur 5 -her&
    sleep 2h 20m
    pkill -f run_train -9
    pkill -f ros -9
    sleep 10s
done

