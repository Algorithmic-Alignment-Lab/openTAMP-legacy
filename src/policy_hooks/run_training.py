import argparse
import copy
import imp
import importlib
import os
import random
import sys
import time

import rospy
from std_msgs.msg import Float32MultiArray, String

from policy_hooks.multiprocess_main import MultiProcessMain
from policy_hooks.multiprocess_pretrain_main import MultiProcessPretrainMain


def get_dir_name(base, no, nt, ind, descr, args=None):
    dir_name = base + 'objs{0}_{1}/exp_id{2}_{3}'.format(no, nt, ind, descr)
    if args is not None:
        useq = '_qfunc' if args.qfunc else ''
        useHer = '_her' if args.her else ''
        expand = '_expand' if args.expand else ''
        neg = '_negExs' if args.negative else ''
        curric = '_curric{0}_{1}'.format(args.cur_thresh, args.n_thresh) if args.cur_thresh > 0 else ''
        dir_name += '{0}{1}{2}{3}{4}'.format(useq, useHer, expand, curric, neg)
    return dir_name


def load_multi(exp_list, n_objs=None, n_targs=None, args=None):
    exps = []
    for exp in exp_list:
        configs = []
        for i in range(len(exp)):
            if n_objs is None:
                c = exp[i]
                next_config = config_module.config.copy()
                config_module = importlib.import_module('policy_hooks.'+c)
            elif n_targs is None:
                n_targs = n_objs
            if n_objs is not None:
                c = exp[i]
                config_module = importlib.import_module('policy_hooks.'+c)
                next_config = config_module.refresh_config(n_objs, n_targs)
            if args is not None:
                next_config['her'] = args.her
                next_config['use_qfunc'] = args.qfunc
                next_config['split_nets'] = args.split
                next_config['expand_process'] = args.expand
                next_config['curric_thresh'] = args.cur_thresh
                next_config['n_thresh'] = args.n_thresh
                next_config['negative'] = args.negative
            next_config['weight_dir'] = get_dir_name(next_config['base_weight_dir'], next_config['num_objs'], next_config['num_targs'], i, next_config['descr'], args)
            next_config['server_id'] = '{0}'.format(str(random.randint(0, 2**16)))
            next_config['mp_server'] = True 
            next_config['pol_server'] = True
            next_config['mcts_server'] = True
            next_config['use_local'] = True
            next_config['log_server'] = False
            next_config['view_server'] = False
            next_config['use_local'] = True
            next_config['log_timing'] = False
            configs.append((next_config, config_module))

        exps.append(configs)
    return exps


def load_config(args, config=None, reload_module=None):
    config_file = args.config
    if reload_module is not None:
        config_module = reload_module
        imp.reload(config_module)
    else:
        config_module = importlib.import_module('policy_hooks.'+config_file)
    config = config_module.config
    config['use_local'] = not args.remote
    config['num_conds'] = args.nconds if args.nconds > 0 else config['num_conds']
    config['common']['num_conds'] = config['num_conds']
    config['algorithm']['conditions'] = config['num_conds']
    config['num_objs'] = args.nobjs if args.nobjs > 0 else config['num_objs']
    config['weight_dir'] = config['base_weight_dir'] + str(config['num_objs'])
    config['log_timing'] = args.timing
    # config['pretrain_timeout'] = args.pretrain_timeout
    config['hl_timeout'] = args.hl_timeout if args.hl_timeout > 0 else config['hl_timeout']
    config['mcts_server'] = args.mcts_server or args.all_servers
    config['mp_server'] = args.mp_server or args.all_servers
    config['pol_server'] = args.policy_server or args.all_servers
    config['log_server'] = args.log_server or args.all_servers
    config['view_server'] = args.view_server
    config['pretrain_steps'] = args.pretrain_steps if args.pretrain_steps > 0 else config['pretrain_steps']
    config['viewer'] = args.viewer
    config['server_id'] = args.server_id if args.server_id != '' else str(random.randint(0,2**32))
    return config, config_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-p', '--pretrain', action='store_true', default=False)
    parser.add_argument('-nf', '--nofull', action='store_true', default=False)
    parser.add_argument('-n', '--nconds', type=int, default=0)
    parser.add_argument('-no', '--nobjs', type=int, default=0)
    parser.add_argument('-nt', '--ntargs', type=int, default=0)
    # parser.add_argument('-ptt', '--pretrain_timeout', type=int, default=300)
    parser.add_argument('-hlt', '--hl_timeout', type=int, default=0)
    parser.add_argument('-k', '--killall', action='store_true', default=True)
    parser.add_argument('-r', '--remote', action='store_true', default=False)
    parser.add_argument('-t', '--timing', action='store_true', default=False)
    parser.add_argument('-mcts', '--mcts_server', action='store_true', default=False)
    parser.add_argument('-mp', '--mp_server', action='store_true', default=False)
    parser.add_argument('-pol', '--policy_server', action='store_true', default=False)
    parser.add_argument('-log', '--log_server', action='store_true', default=False)
    parser.add_argument('-vs', '--view_server', action='store_true', default=False)
    parser.add_argument('-all', '--all_servers', action='store_true', default=False)
    parser.add_argument('-ps', '--pretrain_steps', type=int, default=0)
    parser.add_argument('-v', '--viewer', action='store_true', default=False)
    parser.add_argument('-id', '--server_id', type=str, default='')
    parser.add_argument('-f', '--file', type=str, default='')
    parser.add_argument('-her', '--her', action='store_true', default=False)
    parser.add_argument('-e', '--expand', action='store_true', default=False)
    parser.add_argument('-neg', '--negative', action='store_true', default=False)
    parser.add_argument('-spl', '--split', action='store_true', default=True)
    parser.add_argument('-q', '--qfunc', action='store_true', default=False)
    parser.add_argument('-cur', '--cur_thresh', type=int, default=-1)
    parser.add_argument('-ncur', '--n_thresh', type=int, default=10)

    args = parser.parse_args()

    exps = None
    if args.file == "":
        exps = [[args.config]]

    if False:#args.file == "":
        config, config_module = load_config(args)

    else:
        print('LOADING {0}'.format(args.file))
        current_id = 0
        if exps is None:
            exps = []
            with open(args.file, 'r+') as f:
                exps = eval(f.read())
        n_objs = args.nobjs if args.nobjs > 0 else None
        n_targs = args.ntargs if args.ntargs > 0 else None
        exps = load_multi(exps, n_objs, n_targs, args)
        for exp in exps:
            mains = []
            for c, cm in exp:
                print('\n\n\n\n\n\nLOADING NEXT EXPERIMENT\n\n\n\n\n\n')
                while os.path.isdir('tf_saved/'+c['weight_dir']+str(current_id)):
                    current_id += 1
                c['group_id'] = current_id
                c['weight_dir'] = c['weight_dir']+'{0}'.format(current_id)
                m = MultiProcessMain(c)
                m.monitor = False # If true, m will wait to finish before moving on
                m.group_id = current_id
                with open('tf_saved/'+c['weight_dir']+'/exp_info.txt', 'w+') as f:
                    f.write(str(cm))
                
                m.start()
                mains.append(m)
                time.sleep(1)
            active = True
            
            start_t = time.time()
            while active:
                time.sleep(120.)
                print('RUNNING...')
                active = False
                for m in mains:
                    p_info = m.check_processes()
                    print('PINFO {0}'.format(p_info))
                    active = active or any([code is None for code in p_info])
                    if active: m.expand_rollout_servers()

                if not active:
                    for m in mains:
                        m.kill_processes()

        print('\n\n\n\n\n\n\n\nEXITING')
        sys.exit(0)

    if args.pretrain:
        pretrain = MultiProcessPretrainMain(config)
        pretrain.run()
        config, config_module = load_config(args, reload_module=config_module)
        print '\n\n\nPretraining Complete.\n\n\n'

    if not args.nofull:
        main = MultiProcessMain(config)
        main.start(kill_all=args.killall)

if __name__ == '__main__':
    main()
