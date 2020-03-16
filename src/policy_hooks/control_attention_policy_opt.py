""" This file defines policy optimization for a tensorflow policy. """
import copy
import json
import logging
import os
import sys
import tempfile
import time
import traceback

import numpy as np
import tensorflow as tf

from gps.algorithm.policy_opt.config import POLICY_OPT_TF
from gps.algorithm.policy.tf_policy import TfPolicy
from gps.algorithm.policy_opt.policy_opt import PolicyOpt
from gps.algorithm.policy_opt.tf_utils import TfSolver


MAX_QUEUE_SIZE = 50000

class ControlAttentionPolicyOpt(PolicyOpt):
    """ Policy optimization using tensor flow for DAG computations/nonlinear function approximation. """
    def __init__(self, hyperparams, dO, dU, dPrimObs, dValObs, primBounds):
        import tensorflow as tf
        self.scope = hyperparams['scope'] if 'scope' in hyperparams else None
        # tf.reset_default_graph()
        
        config = copy.deepcopy(POLICY_OPT_TF)
        config.update(hyperparams)
        
        self.split_nets = hyperparams.get('split_nets', False)
        self.valid_scopes = ['control'] if not self.split_nets else list(config['task_list'])

        PolicyOpt.__init__(self, config, dO, dU)

        tf.set_random_seed(self._hyperparams['random_seed'])

        self.tf_iter = 0
        self.batch_size = self._hyperparams['batch_size']

        self._dPrim = primBounds[-1][-1]
        self._dPrimObs = dPrimObs
        self._dValObs = dValObs
        self._primBounds = primBounds
        self.task_map = {}
        self.device_string = "/cpu:0"
        if self._hyperparams['use_gpu'] == 1:
            self.gpu_device = self._hyperparams['gpu_id']
            self.device_string = "/gpu:" + str(self.gpu_device)
        self.act_op = None  # mu_hat
        self.feat_op = None # features
        self.loss_scalar = None
        self.obs_tensor = None
        self.precision_tensor = None
        self.action_tensor = None  # mu true
        self.solver = None
        self.feat_vals = None
        self.init_network()
        self.init_solver()
        self.var = {task: self._hyperparams['init_var'] * np.ones(dU) for task in self.task_map}
        self.var[""] = self._hyperparams['init_var'] * np.ones(dU)
        self.distilled_var = self._hyperparams['init_var'] * np.ones(dU)
        self.weight_dir = self._hyperparams['weight_dir']
        self.scope = self._hyperparams['scope'] if 'scope' in self._hyperparams else None 

        self.gpu_fraction = self._hyperparams['gpu_fraction']
        if not self._hyperparams['allow_growth']:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_fraction)
        else:
            gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
        init_op = tf.initialize_all_variables()
        self.sess.run(init_op)
        self.init_policies(dU)
        if self.scope is not None:
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
            self.saver = tf.train.Saver(variables)
            try:
                self.saver.restore(self.sess, 'tf_saved/'+self.weight_dir+'/'+self.scope+'.ckpt')
                if self.scope in self.task_map:
                    self.task_map[self.scope]['policy'].scale = np.load('tf_saved/'+self.weight_dir+'/'+self.scope+'_scale.npy')
                    self.task_map[self.scope]['policy'].bias = np.load('tf_saved/'+self.weight_dir+'/'+self.scope+'_bias.npy')
                    self.var[self.scope] = np.load('tf_saved/'+self.weight_dir+'/'+self.scope+'_variance.npy')
                    self.task_map[self.scope]['policy'].chol_pol_covar = np.diag(np.sqrt(self.var[self.scope]))
            except Exception as e:
                pass
                # print '\n\nCould not load previous weights for {0} from {1}\n\n'.format(self.scope, self.weight_dir)

        else:
            for scope in self.valid_scopes + ['value', 'primitive', 'image']:
                variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
                if len(variables):
                    self.saver = tf.train.Saver(variables)
                    try:
                        self.saver.restore(self.sess, 'tf_saved/'+self.weight_dir+'/'+scope+'.ckpt')
                        if scope in self.task_map:
                            self.task_map[scope]['policy'].scale = np.load('tf_saved/'+self.weight_dir+'/'+scope+'_scale.npy')
                            self.task_map[scope]['policy'].bias = np.load('tf_saved/'+self.weight_dir+'/'+scope+'_bias.npy')
                            self.var[scope] = np.load('tf_saved/'+self.weight_dir+'/'+scope+'_variance.npy')
                            self.task_map[scope]['policy'].chol_pol_covar = np.diag(np.sqrt(self.var[scope]))
                    except Exception as e:
                        print '\n\nCould not load previous weights for {0} from {1}\n\n'.format(scope, self.weight_dir)
        
        # List of indices for state (vector) data and image (tensor) data in observation.
        self.x_idx, self.img_idx, i = [], [], 0
        if 'obs_image_data' not in self._hyperparams['network_params']:
            self._hyperparams['network_params'].update({'obs_image_data': []})
        for sensor in self._hyperparams['network_params']['obs_include']:
            dim = self._hyperparams['network_params']['sensor_dims'][sensor]
            if sensor in self._hyperparams['network_params']['obs_image_data']:
                self.img_idx = self.img_idx + list(range(i, i+dim))
            else:
                self.x_idx = self.x_idx + list(range(i, i+dim))
            i += dim

        self.prim_x_idx, self.prim_img_idx, i = [], [], 0
        for sensor in self._hyperparams['primitive_network_params']['obs_include']:
            dim = self._hyperparams['primitive_network_params']['sensor_dims'][sensor]
            if sensor in self._hyperparams['primitive_network_params']['obs_image_data']:
                self.prim_img_idx = self.prim_img_idx + list(range(i, i+dim))
            else:
                self.prim_x_idx = self.prim_x_idx + list(range(i, i+dim))
            i += dim

        self.val_x_idx, self.val_img_idx, i = [], [], 0
        for sensor in self._hyperparams['value_network_params']['obs_include']:
            dim = self._hyperparams['value_network_params']['sensor_dims'][sensor]
            if sensor in self._hyperparams['value_network_params']['obs_image_data']:
                self.val_img_idx = self.val_img_idx + list(range(i, i+dim))
            else:
                self.val_x_idx = self.val_x_idx + list(range(i, i+dim))
            i += dim

        self.update_count = 0
        self.update_size = self._hyperparams['update_size']
        self.mu = {}
        self.obs = {}
        self.prc = {}
        self.wt = {}

        self.val_mu = {}
        self.val_obs = {}
        self.val_prc = {}
        self.val_wt = {}

        self.train_iters = 0
        self.average_losses = []
        self.average_val_losses = []
        self.N = 0
        self.buffer_size = MAX_QUEUE_SIZE

        self.share_buffers = self._hyperparams.get('share_buffer', False)
        if self._hyperparams.get('share_buffer', False):
            self.buffers = self._hyperparams['buffers']
            self.buf_sizes = self._hyperparams['buffer_sizes']


    def write_shared_weights(self, scopes=None):
        if scopes is None:
            scopes = self.valid_scopes + ['primitive','image']

        for scope in scopes:
            wts = self.serialize_weights([scope])
            with self.buf_sizes[scope].get_lock():
                self.buf_sizes[scope].value = len(wts)
                self.buffers[scope][:len(wts)] = wts


    def read_shared_weights(self, scopes=None):
        if scopes is None:
            scopes = self.valid_scopes + ['primitive','image']

        for scope in scopes:
            with self.buf_sizes[scope].get_lock():
                wts = self.buffers[scope][:self.buf_sizes[scope].value]

            try:
                self.deserialize_weights(wts)
            except Exception as e:
                pass
                # print(e)
                # print('Could not load {0} weights'.format(scope))


    def serialize_weights(self, scopes=None):
        if scopes is None:
            scopes = self.valid_scopes + ['value', 'primitive', 'image']

        # print 'Serializing', scopes
        var_to_val = {}
        for scope in scopes:
            variables = self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
            for v in variables:
                var_to_val[v.name] = self.sess.run(v).tolist()

        scales = {task: self.task_map[task]['policy'].scale.tolist() for task in scopes if task in self.task_map}
        biases = {task: self.task_map[task]['policy'].bias.tolist() for task in scopes if task in self.task_map}
        variances = {task: self.var[task].tolist() for task in scopes if task in self.task_map}
        scales[''] = []
        biases[''] = []
        variances[''] = []
        return json.dumps([scopes, var_to_val, scales, biases, variances])

    def deserialize_weights(self, json_wts, save=True):
        scopes, var_to_val, scales, biases, variances = json.loads(json_wts)

        # print 'Deserializing', scopes
        for scope in scopes:
            variables = self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
            for var in variables:
                var.load(var_to_val[var.name], session=self.sess)

            if scope not in self.valid_scopes: continue
            # if save:
            #     np.save('tf_saved/'+self.weight_dir+'/control'+'_scale', scales['control'])
            #     np.save('tf_saved/'+self.weight_dir+'/control'+'_bias', biases['control'])
            #     np.save('tf_saved/'+self.weight_dir+'/control'+'_variance', variances['control'])
            self.task_map[scope]['policy'].chol_pol_covar = np.diag(np.sqrt(np.array(variances[scope])))
            self.task_map[scope]['policy'].scale = np.array(scales[scope])
            self.task_map[scope]['policy'].bias = np.array(biases[scope])
            self.var[scope] = np.array(variances[scope])
        if save: self.store_scope_weights(scopes=scopes)
        # print 'Weights for {0} successfully deserialized and stored.'.format(scopes)

    def update_weights(self, scope, weight_dir=None):
        if weight_dir is None:
            weight_dir = self.weight_dir
        self.saver.restore(self.sess, 'tf_saved/'+weight_dir+'/'+scope+'.ckpt')

    def store_scope_weights(self, scopes, weight_dir=None):
        if weight_dir is None:
            weight_dir = self.weight_dir
        for scope in scopes:
            try:
                variables = self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
                saver = tf.train.Saver(variables)
                saver.save(self.sess, 'tf_saved/'+weight_dir+'/'+scope+'.ckpt')
            except:
                print 'Saving variables encountered an issue but it will not crash:'
                traceback.print_exception(*sys.exc_info())

            if scope in self.task_map:
                policy = self.task_map[scope]['policy']
                np.save('tf_saved/'+weight_dir+'/'+scope+'_scale', policy.scale)
                np.save('tf_saved/'+weight_dir+'/'+scope+'_bias', policy.bias)
                np.save('tf_saved/'+weight_dir+'/'+scope+'_variance', self.var[scope])

    def store_weights(self, weight_dir=None):
        if self.scope is None:
            self.store_scope_weights(self.valid_scopes+['value', 'primitive', 'image'], weight_dir)
        else:
            self.store_scope_weights([self.scope], weight_dir)

    def store(self, obs, mu, prc, wt, net, task, update=False, val_ratio=0.05):
        # print('TF got data for', task, 'will update?', update)
        keep_inds = None
        store_val = np.random.uniform() < val_ratio
        inds = np.where(np.abs(np.reshape(wt, [wt.shape[0]*wt.shape[1]])) < 1e-8)
        for dct1, dct2, data in zip([self.mu, self.obs, self.prc, self.wt], [self.val_mu, self.val_obs, self.val_prc, self.val_wt], [mu, obs, prc, wt]):
            if store_val:
                dct = dct2
            else:
                dct = dct1

            if task != 'primitive':
                data = data.reshape([data.shape[0]*data.shape[1]] + list(data.shape[2:]))
            data = np.delete(data, inds[0], axis=0)
            if net not in dct:
                dct[net] = {task: np.array(data)}
            elif task not in dct[net]:
                dct[net][task] = np.array(data)
            else:
                dct[net][task] = np.r_[dct[net][task], data]

            s = MAX_QUEUE_SIZE if not store_val else 1000
            if len(dct[net][task]) > s:
                if keep_inds is None:
                    keep_inds = np.random.choice(range(len(dct[net][task])), s, replace=False)
                dct[net][task] = dct[net][task][keep_inds]
        '''    
        if net not in self.mu or net not in self.obs or net not in self.prc or net not in self.wt:
            self.mu[net] = np.array(mu)
            self.obs[net] = np.array(obs)
            self.prc[net] = np.array(prc)
            self.wt[net] = np.array(wt)
        else:
            #keep_inds = np.where(self.wt[net][:-MAX_QUEUE_SIZE] > 1e0)
            #keep_mu = self.mu[net][keep_inds]
            #keep_obs = self.obs[net][keep_inds]
            #keep_prc = self.prc[net][keep_inds]
            #keep_wt = self.wt[net][keep_inds]
            self.mu[net] = np.r_[self.mu[net], np.array(mu)]
            self.mu[net] = self.mu[net][-MAX_QUEUE_SIZE:]
            self.obs[net] = np.r_[self.obs[net], np.array(obs)]
            self.obs[net] = self.obs[net][-MAX_QUEUE_SIZE:]
            self.prc[net] = np.r_[self.prc[net], np.array(prc)]
            self.prc[net] = self.prc[net][-MAX_QUEUE_SIZE:]
            self.wt[net] = np.r_[self.wt[net], np.array(wt)]
            self.wt[net] = self.wt[net][-MAX_QUEUE_SIZE:]

            #self.mu[net] = np.r_[self.mu[net], keep_mu]
            #self.obs[net] = np.r_[self.obs[net], keep_obs]
            #self.prc[net] = np.r_[self.prc[net], keep_prc]
            #self.wt[net] = np.r_[self.wt[net], keep_wt]
        '''
        self.update_count += len(mu)
        self.N += len(mu)
        return False 

        
        if update: #len(self.mu) > self.update_size:
            # print 'Updating', net
            # Possibility that no good information has come yet
            if net not in self.mu or (task in self.mu[net] and np.all(self.mu[net][task] == self.mu[net][task][0])):
                # print('Insufficient variance for update on', net)
                return False
            obs = np.concatenate(self.obs[net].values(), axis=0)
            mu = np.concatenate(self.mu[net].values(), axis=0)
            prc = np.concatenate(self.prc[net].values(), axis=0)
            wt = np.concatenate(self.wt[net].values(), axis=0)
            
            if len(mu) > self.update_size:
                print('TF got data for', net, 'will update?', update)
                self.update(obs, mu, prc, wt, net)
                if net in self.val_obs:
                    val_obs = np.concatenate(self.val_obs[net].values(), axis=0)
                    val_mu = np.concatenate(self.val_mu[net].values(), axis=0)
                    val_prc = np.concatenate(self.val_prc[net].values(), axis=0)
                    val_wt = np.concatenate(self.val_wt[net].values(), axis=0)
                    self.check_validation(val_obs, val_mu, val_prc, val_wt, net)
                self.store_scope_weights(scopes=[net])
                self.update_count = 0
                return True
            # del self.mu[net]
            # del self.obs[net]
            # del self.prc[net]
            # del self.wt[net]

        return False


    def run_update(self, nets=None):
        if nets is None:
            nets = self.obs.keys()

        updated = False
        for net in nets:
            if net not in self.mu: 
                # print('Insufficient variance for update on', net)
                continue
            obs = np.concatenate(self.obs[net].values(), axis=0)
            mu = np.concatenate(self.mu[net].values(), axis=0)
            prc = np.concatenate(self.prc[net].values(), axis=0)
            wt = np.concatenate(self.wt[net].values(), axis=0)
            
            if len(mu) > self.update_size:
                print('TF updating on data for', net)
                self.update(obs, mu, prc, wt, net)
                self.store_scope_weights(scopes=[net])
                self.update_count = 0
                updated = True
            if net in self.val_obs:
                val_obs = np.concatenate(self.val_obs[net].values(), axis=0)
                val_mu = np.concatenate(self.val_mu[net].values(), axis=0)
                val_prc = np.concatenate(self.val_prc[net].values(), axis=0)
                val_wt = np.concatenate(self.val_wt[net].values(), axis=0)
                if len(val_mu) > self.update_size:
                    self.check_validation(val_obs, val_mu, val_prc, val_wt, net)

        return updated


    def init_network(self):
        """ Helper method to initialize the tf networks used """

        input_tensor = None
        if self._hyperparams['image_network_model'] is not None and (self.scope is None or 'image' == self.scope):
            with tf.variable_scope('image'):
                tf_map_generator = self._hyperparams['image_network_model']
                dIm = self._dIm
                tf_map, fc_vars, last_conv_vars = tf_map_generator(dim_input=dIm, dim_output=1, batch_size=self.batch_size,
                                          network_config=self._hyperparams['image_network_params'])
                self.image_obs_tensor = tf_map.get_input_tensor()
                self.image_precision_tensor = tf_map.get_precision_tensor()
                self.image_action_tensor = tf_map.get_target_output_tensor()
                self.image_act_op = tf_map.get_output_op()
                self.image_feat_op = tf_map.get_feature_op()
                self.image_loss_scalar = tf_map.get_loss_op()
                self.image_fc_vars = fc_vars
                self.image_last_conv_vars = last_conv_vars

                # Setup the gradients
                self.image_grads = [tf.gradients(self.image_act_op[:,u], self.image_obs_tensor)[0] for u in range(1)]

                # input_tensor = self.image_act_op

        if self.scope is None or 'primitive' == self.scope:
            with tf.variable_scope('primitive'):
                tf_map_generator = self._hyperparams['primitive_network_model']
                tf_map, fc_vars, last_conv_vars = tf_map_generator(dim_input=self._dPrimObs, dim_output=self._dPrim, batch_size=self.batch_size,
                                          network_config=self._hyperparams['primitive_network_params'], input_layer=input_tensor)
                self.primitive_obs_tensor = tf_map.get_input_tensor()
                self.primitive_precision_tensor = tf_map.get_precision_tensor()
                self.primitive_action_tensor = tf_map.get_target_output_tensor()
                self.primitive_act_op = tf_map.get_output_op()
                self.primitive_feat_op = tf_map.get_feature_op()
                self.primitive_loss_scalar = tf_map.get_loss_op()
                self.primitive_fc_vars = fc_vars
                self.primitive_last_conv_vars = last_conv_vars

                # Setup the gradients
                self.primitive_grads = [tf.gradients(self.primitive_act_op[:,u], self.primitive_obs_tensor)[0] for u in range(self._dPrim)]

        if self.scope is None or 'value' == self.scope:
            with tf.variable_scope('value'):
                tf_map_generator = self._hyperparams['value_network_model']
                dValObs = self._dValObs
                tf_map, fc_vars, last_conv_vars = tf_map_generator(dim_input=dValObs, dim_output=1, batch_size=self.batch_size,
                                          network_config=self._hyperparams['value_network_params'], input_layer=input_tensor)
                self.value_obs_tensor = tf_map.get_input_tensor()
                self.value_precision_tensor = tf_map.get_precision_tensor()
                self.value_action_tensor = tf_map.get_target_output_tensor()
                self.value_act_op = tf_map.get_output_op()
                self.value_feat_op = tf_map.get_feature_op()
                self.value_loss_scalar = tf_map.get_loss_op()
                self.value_fc_vars = fc_vars
                self.value_last_conv_vars = last_conv_vars

                # Setup the gradients
                self.value_grads = [tf.gradients(self.value_act_op[:,u], self.value_obs_tensor)[0] for u in range(1)]

        # with tf.variable_scope('distilled'):
        #     tf_map_generator = self._hyperparams['distilled_network_model']
        #     tf_map, fc_vars, last_conv_vars = tf_map_generator(dim_input=self._dPrimObs, dim_output=self._dU, batch_size=self.batch_size,
        #                               network_config=self._hyperparams['distilled_network_params'])
        #     self.distilled_obs_tensor = tf_map.get_input_tensor()
        #     self.distilled_precision_tensor = tf_map.get_precision_tensor()
        #     self.distilled_action_tensor = tf_map.get_target_output_tensor()
        #     self.distilled_act_op = tf_map.get_output_op()
        #     self.distilled_feat_op = tf_map.get_feature_op()
        #     self.distilled_loss_scalar = tf_map.get_loss_op()
        #     self.distilled_fc_vars = fc_vars
        #     self.distilled_last_conv_vars = last_conv_vars

        #     # Setup the gradients
        #     self.distilled_grads = [tf.gradients(self.distilled_act_op[:,u], self.distilled_obs_tensor)[0] for u in range(self._dU)]

        for scope in self.valid_scopes:
            if self.scope is None or scope == self.scope:
                with tf.variable_scope(scope):
                    self.task_map[scope] = {}
                    tf_map_generator = self._hyperparams['network_model']
                    tf_map, fc_vars, last_conv_vars = tf_map_generator(dim_input=self._dO, dim_output=self._dU, batch_size=self.batch_size,
                                              network_config=self._hyperparams['network_params'], input_layer=input_tensor)
                    self.task_map[scope]['obs_tensor'] = tf_map.get_input_tensor()
                    self.task_map[scope]['precision_tensor'] = tf_map.get_precision_tensor()
                    self.task_map[scope]['action_tensor'] = tf_map.get_target_output_tensor()
                    self.task_map[scope]['act_op'] = tf_map.get_output_op()
                    self.task_map[scope]['feat_op'] = tf_map.get_feature_op()
                    self.task_map[scope]['loss_scalar'] = tf_map.get_loss_op()
                    self.task_map[scope]['fc_vars'] = fc_vars
                    self.task_map[scope]['last_conv_vars'] = last_conv_vars

                    # Setup the gradients
                    self.task_map[scope]['grads'] = [tf.gradients(self.task_map[scope]['act_op'][:,u], self.task_map[scope]['obs_tensor'])[0] for u in range(self._dU)]

    def init_solver(self):
        """ Helper method to initialize the solver. """
        if self.scope is None or 'primitive' == self.scope:
            vars_to_opt = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='primitive')
            self.primitive_solver = TfSolver(loss_scalar=self.primitive_loss_scalar,
                                               solver_name=self._hyperparams['solver_type'],
                                               base_lr=self._hyperparams['lr'],
                                               lr_policy=self._hyperparams['lr_policy'],
                                               momentum=self._hyperparams['momentum'],
                                               weight_decay=0.,
                                               fc_vars=self.primitive_fc_vars,
                                               last_conv_vars=self.primitive_last_conv_vars,
                                               vars_to_opt=vars_to_opt)

        if self.scope is None or 'value' == self.scope:
            vars_to_opt = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='value')
            self.value_solver = TfSolver(loss_scalar=self.value_loss_scalar,
                                           solver_name=self._hyperparams['solver_type'],
                                           base_lr=self._hyperparams['lr'],
                                           lr_policy=self._hyperparams['lr_policy'],
                                           momentum=self._hyperparams['momentum'],
                                           weight_decay=0.,
                                           fc_vars=self.value_fc_vars,
                                           last_conv_vars=self.value_last_conv_vars,
                                           vars_to_opt=vars_to_opt)

        if self._hyperparams['image_network_model'] is not None and (self.scope is None or 'image' == self.scope):
            vars_to_opt = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='image_filter')
            self.image_solver = TfSolver(loss_scalar=self.image_loss_scalar,
                                           solver_name=self._hyperparams['solver_type'],
                                           base_lr=self._hyperparams['lr'],
                                           lr_policy=self._hyperparams['lr_policy'],
                                           momentum=self._hyperparams['momentum'],
                                           weight_decay=0.,
                                           fc_vars=self.image_fc_vars,
                                           last_conv_vars=self.image_last_conv_vars,
                                           vars_to_opt=vars_to_opt)

        # vars_to_opt = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='distilled')
        # self.distilled_solver = TfSolver(loss_scalar=self.distilled_loss_scalar,
        #                                    solver_name=self._hyperparams['solver_type'],
        #                                    base_lr=self._hyperparams['lr'],
        #                                    lr_policy=self._hyperparams['lr_policy'],
        #                                    momentum=self._hyperparams['momentum'],
        #                                    weight_decay=self._hyperparams['weight_decay'],
        #                                    fc_vars=self.distilled_fc_vars,
        #                                    last_conv_vars=self.distilled_last_conv_vars,
        #                                    vars_to_opt=vars_to_opt)

        for scope in self.valid_scopes:
            if self.scope is None or scope == self.scope:
                vars_to_opt = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
                self.task_map[scope]['solver'] = TfSolver(loss_scalar=self.task_map[scope]['loss_scalar'],
                                                       solver_name=self._hyperparams['solver_type'],
                                                       base_lr=self._hyperparams['lr'],
                                                       lr_policy=self._hyperparams['lr_policy'],
                                                       momentum=self._hyperparams['momentum'],
                                                       weight_decay=self._hyperparams['weight_decay'],
                                                       fc_vars=self.task_map[scope]['fc_vars'],
                                                       last_conv_vars=self.task_map[scope]['last_conv_vars'],
                                                       vars_to_opt=vars_to_opt)

    def init_policies(self, dU):
        for scope in self.valid_scopes:
            if self.scope is None or scope == self.scope:
                self.task_map[scope]['policy'] = TfPolicy(dU, 
                                                        self.task_map[scope]['obs_tensor'], 
                                                        self.task_map[scope]['act_op'], 
                                                        self.task_map[scope]['feat_op'],
                                                        np.zeros(dU), 
                                                        self.sess, 
                                                        self.device_string, 
                                                        copy_param_scope=None)

            # self.distilled_policy = TfPolicy(dU,
        #                                  self.distilled_obs_tensor,
        #                                  self.distilled_act_op,
        #                                  self.distilled_feat_op,
        #                                  np.zeros(dU),
        #                                  self.sess,
        #                                  self.device_string,
        #                                  copy_param_scope=None)

    def task_distr(self, obs):
        if len(obs.shape) < 2:
            obs = obs.reshape(1, -1)
        distr = self.sess.run(self.primitive_act_op, feed_dict={self.primitive_obs_tensor:obs}).flatten()
        res = []
        for bound in self._primBounds:
            res.append(distr[bound[0]:bound[1]])
        return res

    def value(self, obs):
        if len(obs.shape) < 2:
            obs = obs.reshape(1, -1)
        value = self.sess.run(self.value_act_op, feed_dict={self.value_obs_tensor:obs}).flatten()
        return value.flatten()

    def check_validation(self, obs, tgt_mu, tgt_prc, tgt_wt, task="control", val_ratio=0.2):
        """
        Update policy.
        Args:
            obs: Numpy array of observations, N x T x dO.
            tgt_mu: Numpy array of mean controller outputs, N x T x dU.
            tgt_prc: Numpy array of precision matrices, N x T x dU x dU.
            tgt_wt: Numpy array of weights, N x T.
        Returns:
            A tensorflow object with updated weights.
        """
        # if np.any(np.isnan(tgt_mu)) or np.any(np.abs(tgt_mu) == np.inf):
        #     import ipdb; ipdb.set_trace()
        start_t = time.time()
        NT = obs.shape[0]
        if task == 'primitive':
            dU, dO = self._dPrim, self._dPrimObs
        else:
            dU, dO = self._dU, self._dO

        # TODO - Make sure all weights are nonzero?

        # Save original tgt_prc.
        tgt_prc_orig = np.reshape(tgt_prc, [NT, dU, dU])

        # Renormalize weights.
        assert not (np.sum(tgt_wt) == 0 or np.any(np.isnan(tgt_wt)))
        tgt_wt *= (float(NT) / np.sum(tgt_wt))
        # Allow weights to be at most twice the robust median.
        # mn = np.median(tgt_wt[(tgt_wt > 1e-2).nonzero()])
        mn = np.median(tgt_wt[(tgt_wt > 1e-2).nonzero()])
        # for n in range(N):
        #     for t in range(T):
        #         tgt_wt[n, t] = min(tgt_wt[n, t], 2 * mn)
        # Robust median should be around one.
        tgt_wt /= mn

        # Reshape inputs.
        obs = np.reshape(obs, (NT, dO))
        tgt_mu = np.reshape(tgt_mu, (NT, dU))
        tgt_prc = np.reshape(tgt_prc, (NT, dU, dU))
        tgt_wt = np.reshape(tgt_wt, (NT, 1, 1))

        # Fold weights into tgt_prc.
        tgt_prc = tgt_wt * tgt_prc

        # TODO: Find entries with very low weights?

        # Normalize obs, but only compute normalzation at the beginning.
        if task != 'primitive':
            policy = self.task_map[task]['policy']
            if policy.scale is None or policy.bias is None:
                policy.x_idx = self.x_idx
                # 1e-3 to avoid infs if some state dimensions don't change in the
                # first batch of samples
                # policy.scale = np.diag(
                #     1.0 / np.maximum(np.std(obs[:, self.x_idx], axis=0), 1e-3))
                policy.scale = np.diag(
                    1.0 / np.maximum(np.std(obs[:, self.x_idx], axis=0), 1e-1))
                policy.bias = - np.mean(
                    obs[:, self.x_idx].dot(policy.scale), axis=0)

                np.save('tf_saved/'+self.weight_dir+'/'+task+'_scale', policy.scale)
                np.save('tf_saved/'+self.weight_dir+'/'+task+'_bias', policy.bias)
            obs[:, self.x_idx] = obs[:, self.x_idx].dot(policy.scale) + policy.bias
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isnan(tgt_mu))
        assert not np.any(np.isnan(tgt_prc))
        # Assuming that N*T >= self.batch_size.

        batches_per_epoch = np.maximum(np.floor(NT / self.batch_size), 1)
        idx = range(NT)
        average_loss = 0
        np.random.shuffle(idx)
        # actual training.
        # print "\nEntering Tensorflow Training Loop"
        for i in range(10):
            self.train_iters += 1
            # Load in data for this batch.
            start_idx = int(i * self.batch_size %
                            (batches_per_epoch * self.batch_size))
            idx_i = idx[start_idx:start_idx+self.batch_size]
            if task == 'primitive':
                feed_dict = {self.primitive_obs_tensor: obs[idx_i],
                             self.primitive_action_tensor: tgt_mu[idx_i],
                             self.primitive_precision_tensor: tgt_prc[idx_i]}
                val_loss = self.primitive_solver(feed_dict, self.sess, device_string=self.device_string, train=False)
            else:
                feed_dict = {self.task_map[task]['obs_tensor']: obs[idx_i],
                             self.task_map[task]['action_tensor']: tgt_mu[idx_i],
                             self.task_map[task]['precision_tensor']: tgt_prc[idx_i]}
                val_loss = self.task_map[task]['solver'](feed_dict, self.sess, device_string=self.device_string, train=False)
            average_loss += val_loss

        self.average_val_losses.append({
                'loss': average_loss / 10.,
                'iter': self.tf_iter,
                'N': self.N})


     
    def update(self, obs, tgt_mu, tgt_prc, tgt_wt, task="control", val_ratio=0.2):
        """
        Update policy.
        Args:
            obs: Numpy array of observations, N x T x dO.
            tgt_mu: Numpy array of mean controller outputs, N x T x dU.
            tgt_prc: Numpy array of precision matrices, N x T x dU x dU.
            tgt_wt: Numpy array of weights, N x T.
        Returns:
            A tensorflow object with updated weights.
        """
        if task == 'primitive':
            return self.update_primitive_filter(obs, tgt_mu, tgt_prc, tgt_wt, val_ratio=val_ratio)
        if task == 'value':
            return self.update_value(obs, tgt_mu, tgt_prc, tgt_wt)
        if task == 'image':
            return self.update_image_net(obs, tgt_mu, tgt_prc, tgt_wt)
        # if np.any(np.isnan(tgt_mu)) or np.any(np.abs(tgt_mu) == np.inf):
        #     import ipdb; ipdb.set_trace()
        start_t = time.time()
        #N, T = obs.shape[:2]
        NT = obs.shape[0]
        dU, dO = self._dU, self._dO

        # TODO - Make sure all weights are nonzero?

        # Save original tgt_prc.
        tgt_prc_orig = np.reshape(tgt_prc, [NT, dU, dU])

        # Renormalize weights.
        assert not (np.sum(tgt_wt) == 0 or np.any(np.isnan(tgt_wt)))
        tgt_wt *= (float(NT) / np.sum(tgt_wt))
        # Allow weights to be at most twice the robust median.
        # mn = np.median(tgt_wt[(tgt_wt > 1e-2).nonzero()])
        mn = np.median(tgt_wt[(tgt_wt > 1e-2).nonzero()])
        # for n in range(N):
        #     for t in range(T):
        #         tgt_wt[n, t] = min(tgt_wt[n, t], 2 * mn)
        # Robust median should be around one.
        tgt_wt /= mn

        # Reshape inputs.
        obs = np.reshape(obs, (NT, dO))
        tgt_mu = np.reshape(tgt_mu, (NT, dU))
        tgt_prc = np.reshape(tgt_prc, (NT, dU, dU))
        tgt_wt = np.reshape(tgt_wt, (NT, 1, 1))

        # Fold weights into tgt_prc.
        tgt_prc = tgt_wt * tgt_prc

        # TODO: Find entries with very low weights?

        # Normalize obs, but only compute normalzation at the beginning.
        policy = self.task_map[task]['policy']
        if policy.scale is None or policy.bias is None:
            policy.x_idx = self.x_idx
            # 1e-3 to avoid infs if some state dimensions don't change in the
            # first batch of samples
            # policy.scale = np.diag(
            #     1.0 / np.maximum(np.std(obs[:, self.x_idx], axis=0), 1e-3))
            policy.scale = np.diag(
                1.0 / np.maximum(np.std(obs[:, self.x_idx], axis=0), 1e-1))
            policy.bias = - np.mean(
                obs[:, self.x_idx].dot(policy.scale), axis=0)

            np.save('tf_saved/'+self.weight_dir+'/'+task+'_scale', policy.scale)
            np.save('tf_saved/'+self.weight_dir+'/'+task+'_bias', policy.bias)
        obs[:, self.x_idx] = obs[:, self.x_idx].dot(policy.scale) + policy.bias
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isnan(tgt_mu))
        assert not np.any(np.isnan(tgt_prc))
        # Assuming that N*T >= self.batch_size.

        batches_per_epoch = np.maximum(np.floor(NT / self.batch_size), 1)
        idx = np.array(range(NT))
        average_loss = 0
        np.random.shuffle(idx)

        if self._hyperparams['fc_only_iterations'] > 0:
            feed_dict = {self.obs_tensor: obs}
            num_values = obs.shape[0]
            conv_values = self.task_map[task]['solver'].get_last_conv_values(self.sess, feed_dict, num_values, self.batch_size)
            for i in range(self._hyperparams['fc_only_iterations'] ):
                start_idx = int(i * self.batch_size %
                                (batches_per_epoch * self.batch_size))
                idx_i = idx[start_idx:start_idx+self.batch_size]
                feed_dict = {self.task_map[task]['last_conv_vars']: conv_values[idx_i],
                             self.task_map[task]['action_tensor']: tgt_mu[idx_i],
                             self.task_map[task]['precision_tensor']: tgt_prc[idx_i]}
                train_loss = self.task_map[task]['solver'](feed_dict, self.sess, device_string=self.device_string, use_fc_solver=True)
                average_loss += train_loss

                # if (i+1) % 500 == 0:
                #     LOGGER.info('tensorflow iteration %d, average loss %f',
                #                     i+1, average_loss / 500)
                #     average_loss = 0
            average_loss = 0

        # actual training.
        # print "\nEntering Tensorflow Training Loop"
        for i in range(self._hyperparams['iterations']):
            self.train_iters += 1
            # Load in data for this batch.
            start_idx = int(i * self.batch_size %
                            (batches_per_epoch * self.batch_size))
            idx_i = idx[start_idx:start_idx+self.batch_size]

            idx_i = idx[np.random.choice(idx, self.batch_size, replace=False)]
            feed_dict = {self.task_map[task]['obs_tensor']: obs[idx_i],
                         self.task_map[task]['action_tensor']: tgt_mu[idx_i],
                         self.task_map[task]['precision_tensor']: tgt_prc[idx_i]}
            train_loss = self.task_map[task]['solver'](feed_dict, self.sess, device_string=self.device_string)

            if np.isnan(train_loss) or np.isinf(train_loss):
                print('\n\nINVALID NETWORK UPDATE: RESTORING MODEL FROM CKPT (iteration {0})'.format(i))
                try:
                    self.saver.restore(self.sess, 'tf_saved/'+self.weight_dir+'/'+task+'.ckpt')
                    self.task_map[task]['policy'].scale = np.load('tf_saved/'+self.weight_dir+'/'+task+'_scale.npy')
                    self.task_map[task]['policy'].bias = np.load('tf_saved/'+self.weight_dir+'/'+task+'_bias.npy')
                    self.var[task] = np.load('tf_saved/'+self.weight_dir+'/'+task+'_variance.npy')
                    self.task_map[task]['policy'].chol_pol_covar = np.diag(np.sqrt(self.var[task]))
                except Exception as e:
                    print '\n\nCould not load previous weights for {0} from {1} after invalid update\n\n'.format(task, self.weight_dir)


            average_loss += train_loss
            # if (i+1) % 50 == 0:
            #     LOGGER.info('tensorflow iteration %d, average loss %f',
            #                  i+1, average_loss / 50)
            #     average_loss = 0
        # print "Leaving Tensorflow Training Loop\n"

        self.tf_iter += self._hyperparams['iterations']
        self.average_losses.append({
                'loss': average_loss / self._hyperparams['iterations'],
                'iter': self.tf_iter,
                'N': self.N})

        feed_dict = {self.obs_tensor: obs}
        num_values = obs.shape[0]
        if self.task_map[task]['feat_op'] is not None:
            self.task_map[task]['feat_vals'] = self.task_map[task]['solver'].get_var_values(self.sess, self.task_map[task]['feat_op'], feed_dict, num_values, self.batch_size)
        # Keep track of tensorflow iterations for loading solver states.

        # Optimize variance.
        A = np.sum(tgt_prc_orig, 0) + 2 * NT * \
                self._hyperparams['ent_reg'] * np.ones((dU, dU))
        A = A / np.sum(tgt_wt)

        # TODO - Use dense covariance?
        self.var[task] = 1 / np.diag(A)
        policy.chol_pol_covar = np.diag(np.sqrt(self.var[task]))
        print('Time to run policy update:', time.time() - start_t)

        return policy

    def update_primitive_filter(self, obs, tgt_mu, tgt_prc, tgt_wt, val_ratio=0.2):
        """
        Update policy.
        Args:
            obs: Numpy array of observations, N x T x dO.
            tgt_mu: Numpy array of mean filter outputs, N x T x dP.
            tgt_prc: Numpy array of precision matrices, N x T x dP x dP.
            tgt_wt: Numpy array of weights, N x T.
        Returns:
            A tensorflow object with updated weights.
        """
        # print 'Updating primitive network...'
        N = obs.shape[0]
        dP, dO = self._dPrim, self._dPrimObs

        # TODO - Make sure all weights are nonzero?

        # Save original tgt_prc.
        tgt_prc_orig = np.reshape(tgt_prc, [N, dP, dP])

        # Renormalize weights.
        tgt_wt *= (float(N) / np.sum(tgt_wt))
        # Allow weights to be at most twice the robust median.
        mn = np.median(tgt_wt[(tgt_wt > 1e-2).nonzero()])
        for n in range(N):
            tgt_wt[n] = min(tgt_wt[n], 2 * mn)
        # Robust median should be around one.
        tgt_wt /= mn

        # Reshape inputs.
        obs = np.reshape(obs, (N, dO))
        tgt_mu = np.reshape(tgt_mu, (N, dP))
        tgt_prc = np.reshape(tgt_prc, (N, dP, dP))
        tgt_wt = np.reshape(tgt_wt, (N, 1, 1))

        # Fold weights into tgt_prc.
        tgt_prc = tgt_wt * tgt_prc

        # Assuming that N*T >= self.batch_size.
        batch_size = np.minimum(self.batch_size, N)
        batches_per_epoch = np.maximum(np.floor(N / batch_size), 1)
        idx = range(N)
        average_loss = 0
        np.random.shuffle(idx)

        if self._hyperparams['fc_only_iterations'] > 0:
            feed_dict = {self.obs_tensor: obs}
            num_values = obs.shape[0]
            conv_values = self.primitive_solver.get_last_conv_values(self.sess, feed_dict, num_values, batch_size)
            for i in range(self._hyperparams['fc_only_iterations'] ):
                start_idx = int(i * batch_size %
                                (batches_per_epoch * batch_size))
                idx_i = idx[start_idx:start_idx+batch_size]
                feed_dict = {self.primitive_last_conv_vars: conv_values[idx_i],
                             self.primitive_action_tensor: tgt_mu[idx_i],
                             self.primitive_precision_tensor: tgt_prc[idx_i]}
                train_loss = self.primitive_solver(feed_dict, self.sess, device_string=self.device_string, use_fc_solver=True)
                average_loss += train_loss

                # if (i+1) % 500 == 0:
                #     LOGGER.info('tensorflow iteration %d, average loss %f',
                #                     i+1, average_loss / 500)
                #     average_loss = 0
            average_loss = 0

        # actual training.
        for i in range(self._hyperparams['iterations']):
            # Load in data for this batch.
            self.train_iters += 1
            start_idx = int(i * self.batch_size %
                            (batches_per_epoch * self.batch_size))
            idx_i = idx[start_idx:start_idx+self.batch_size]
            feed_dict = {self.primitive_obs_tensor: obs[idx_i],
                         self.primitive_action_tensor: tgt_mu[idx_i],
                         self.primitive_precision_tensor: tgt_prc[idx_i]}
            train_loss = self.primitive_solver(feed_dict, self.sess, device_string=self.device_string)

            average_loss += train_loss

        self.average_losses.append(average_loss / self._hyperparams['iterations'])
        feed_dict = {self.obs_tensor: obs}
        num_values = obs.shape[0]
        if self.primitive_feat_op is not None:
            self.primitive_feat_vals = self.primitive_solver.get_var_values(self.sess, self.primitive_feat_op, feed_dict, num_values, self.batch_size)
        # print 'Updated primitive network.\n'


    def update_value(self, obs, tgt_mu, tgt_prc, tgt_wt, val_ratio=0.2):
        """
        Update policy.
        Args:
            obs: Numpy array of observations, N x T x dO.
            tgt_mu: Numpy array of mean filter outputs, N x T x dP.
            tgt_prc: Numpy array of precision matrices, N x T x dP x dP.
            tgt_wt: Numpy array of weights, N x T.
        Returns:
            A tensorflow object with updated weights.
        """
        # print 'Updating value network...'
        N = obs.shape[0]
        dP, dO = 1, self._dValObs

        # TODO - Make sure all weights are nonzero?

        # Save original tgt_prc.
        tgt_prc_orig = np.reshape(tgt_prc, [N, dP, dP])

        # Renormalize weights.
        tgt_wt *= (float(N) / np.sum(tgt_wt))
        # Allow weights to be at most twice the robust median.
        mn = np.median(tgt_wt[(tgt_wt > 1e-2).nonzero()])
        for n in range(N):
            tgt_wt[n] = min(tgt_wt[n], 2 * mn)
        # Robust median should be around one.
        tgt_wt /= mn

        # Reshape inputs.
        obs = np.reshape(obs, (N, dO))
        tgt_mu = np.reshape(tgt_mu, (N, dP))
        tgt_prc = np.reshape(tgt_prc, (N, dP, dP))
        tgt_wt = np.reshape(tgt_wt, (N, 1, 1))

        # Fold weights into tgt_prc.
        tgt_prc = tgt_wt * tgt_prc

        # Assuming that N*T >= self.batch_size.
        batch_size = np.minimum(self.batch_size, N)
        batches_per_epoch = np.maximum(np.floor(N / batch_size), 1)
        idx = range(N)
        average_loss = 0
        np.random.shuffle(idx)

        if self._hyperparams['fc_only_iterations'] > 0:
            feed_dict = {self.obs_tensor: obs}
            num_values = obs.shape[0]
            conv_values = self.value_solver.get_last_conv_values(self.sess, feed_dict, num_values, batch_size)
            for i in range(self._hyperparams['fc_only_iterations'] ):
                start_idx = int(i * batch_size %
                                (batches_per_epoch * batch_size))
                idx_i = idx[start_idx:start_idx+batch_size]
                feed_dict = {self.value_last_conv_vars: conv_values[idx_i],
                             self.value_action_tensor: tgt_mu[idx_i],
                             self.value_precision_tensor: tgt_prc[idx_i]}
                train_loss = self.value_solver(feed_dict, self.sess, device_string=self.device_string, use_fc_solver=True)
                average_loss += train_loss

                # if (i+1) % 500 == 0:
                #     LOGGER.info('tensorflow iteration %d, average loss %f',
                #                     i+1, average_loss / 500)
                #     average_loss = 0
            average_loss = 0

        # actual training.
        for i in range(self._hyperparams['iterations']):
            # Load in data for this batch.
            self.train_iters += 1
            start_idx = int(i * self.batch_size %
                            (batches_per_epoch * self.batch_size))
            idx_i = idx[start_idx:start_idx+self.batch_size]
            feed_dict = {self.value_obs_tensor: obs[idx_i],
                         self.value_action_tensor: tgt_mu[idx_i],
                         self.value_precision_tensor: tgt_prc[idx_i]}
            train_loss = self.value_solver(feed_dict, self.sess, device_string=self.device_string)

            average_loss += train_loss
            if (i+1) % 50 == 0:
                # LOGGER.info('tensorflow iteration %d, average loss %f',
                #              i+1, average_loss / 50)
                average_loss = 0

        self.average_losses.append(average_loss)
        feed_dict = {self.obs_tensor: obs}
        num_values = obs.shape[0]
        if self.value_feat_op is not None:
            self.value_feat_vals = self.value_solver.get_var_values(self.sess, self.value_feat_op, feed_dict, num_values, self.batch_size)
        # print 'Updated value network.'


    def update_distilled(self, obs, tgt_mu, tgt_prc, tgt_wt):
        """
        Update policy.
        Args:
            obs: Numpy array of observations, N x T x dO.
            tgt_mu: Numpy array of mean controller outputs, N x T x dU.
            tgt_prc: Numpy array of precision matrices, N x T x dU x dU.
            tgt_wt: Numpy array of weights, N x T.
        Returns:
            A tensorflow object with updated weights.
        """
        N, _ = obs.shape[:2]
        dU, dO = self._dU, self._dPrimObs

        # TODO - Make sure all weights are nonzero?

        # Save original tgt_prc.
        tgt_prc_orig = np.reshape(tgt_prc, [N, dU, dU])

        # Renormalize weights.
        tgt_wt *= (float(N) / np.sum(tgt_wt))
        # Allow weights to be at most twice the robust median.
        mn = np.median(tgt_wt[(tgt_wt > 1e-2).nonzero()])
        for n in range(N):
            tgt_wt[n] = min(tgt_wt[n], 2 * mn)
        # Robust median should be around one.
        tgt_wt /= mn

        # Reshape inputs.
        obs = np.reshape(obs, (N, dO))
        tgt_mu = np.reshape(tgt_mu, (N, dU))
        tgt_prc = np.reshape(tgt_prc, (N, dU, dU))
        tgt_wt = np.reshape(tgt_wt, (N, 1, 1))

        # Fold weights into tgt_prc.
        tgt_prc = tgt_wt * tgt_prc

        # TODO: Find entries with very low weights?

        # Normalize obs, but only compute normalzation at the beginning.
        if self.distilled_policy.scale is None or self.distilled_policy.bias is None:
            self.distilled_policy.x_idx = self.prim_x_idx
            # 1e-3 to avoid infs if some state dimensions don't change in the
            # first batch of samples
            self.distilled_policy.scale = np.diag(
                1.0 / np.maximum(np.std(obs[:, self.prim_x_idx], axis=0), 1e-3))
            self.distilled_policy.bias = - np.mean(
                obs[:, self.prim_x_idx].dot(self.distilled_policy.scale), axis=0)
        obs[:, self.prim_x_idx] = obs[:, self.prim_x_idx].dot(self.distilled_policy.scale) + self.distilled_policy.bias

        # Assuming that N >= self.batch_size.
        batches_per_epoch = np.maximum(np.floor(N / self.batch_size), 1)
        idx = range(N)
        average_loss = 0
        np.random.shuffle(idx)

        if self._hyperparams['fc_only_iterations'] > 0:
            feed_dict = {self.distilled_obs_tensor: obs}
            num_values = obs.shape[0]
            conv_values = self.distilled_solver.get_last_conv_values(self.sess, feed_dict, num_values, self.batch_size)
            for i in range(self._hyperparams['fc_only_iterations'] ):
                start_idx = int(i * self.batch_size %
                                (batches_per_epoch * self.batch_size))
                idx_i = idx[start_idx:start_idx+self.batch_size]
                feed_dict = {self.distilled_last_conv_vars: conv_values[idx_i],
                             self.distilled_action_tensor: tgt_mu[idx_i],
                             self.distilled_precision_tensor: tgt_prc[idx_i]}
                train_loss = self.distilled_solver(feed_dict, self.sess, device_string=self.device_string, use_fc_solver=True)
                average_loss += train_loss

                # if (i+1) % 500 == 0:
                #     LOGGER.info('tensorflow iteration %d, average loss %f',
                #                     i+1, average_loss / 500)
                #     average_loss = 0
            average_loss = 0

        # actual training.
        for i in range(self._hyperparams['iterations']):
            # Load in data for this batch.
            start_idx = int(i * self.batch_size %
                            (batches_per_epoch * self.batch_size))
            idx_i = idx[start_idx:start_idx+self.batch_size]
            feed_dict = {self.distilled_obs_tensor: obs[idx_i],
                         self.distilled_action_tensor: tgt_mu[idx_i],
                         self.distilled_precision_tensor: tgt_prc[idx_i]}
            train_loss = self.distilled_solver(feed_dict, self.sess, device_string=self.device_string)

            average_loss += train_loss
            # if (i+1) % 50 == 0:
                # LOGGER.info('tensorflow iteration %d, average loss %f',
                #              i+1, average_loss / 50)
                # average_loss = 0

        feed_dict = {self.obs_tensor: obs}
        num_values = obs.shape[0]
        if self.distilled_feat_op is not None:
            self.distilled_feat_vals = self.solver.get_var_values(self.sess, self.distilled_feat_op, feed_dict, num_values, self.batch_size)
        # Keep track of tensorflow iterations for loading solver states.
        self.tf_iter += self._hyperparams['iterations']

        # Optimize variance.
        A = np.sum(tgt_prc_orig, 0) + 2 * N * \
                self._hyperparams['ent_reg'] * np.ones((dU, dU))
        A = A / np.sum(tgt_wt)

        # TODO - Use dense covariance?
        self.distilled_var = 1 / np.diag(A)
        self.distilled_policy.chol_pol_covar = np.diag(np.sqrt(self.distilled_var))

        return self.distilled_policy

    def traj_prob(self, obs, task="control"):
        assert len(obs.shape) == 2 or obs.shape[0] == 1
        mu, sig, prec, det_sig = self.prob(obs, task)
        traj = np.tri(mu.shape[1]).dot(mu[0])
        return np.array([traj]), sig, prec, det_sig

    def policy_initialized(self, task):
        if task in self.valid_scopes:
            return self.task_map[task]['policy'].scale is not None
        return self.task_map['control']['policy'].scale is not None

    def prob(self, obs, task="control"):
        """
        Run policy forward.
        Args:
            obs: Numpy array of observations that is N x T x dO.
        """
        if len(obs.shape) < 3:
            obs = obs.reshape((1, obs.shape[0], obs.shape[1]))
        dU = self._dU
        N, T = obs.shape[:2]

        # Normalize obs.
        if task not in self.valid_scopes:
            task = "control"
        if task in self.task_map:
            policy = self.task_map[task]['policy']
        else:
            policy = getattr(self, '{0}_policy'.format(task))
        if policy.scale is not None:
            # TODO: Should prob be called before update?
            for n in range(N):
                obs[n, :, self.x_idx] = (obs[n, :, self.x_idx].T.dot(policy.scale)
                                         + policy.bias).T

        output = np.zeros((N, T, dU))

        for i in range(N):
            for t in range(T):
                # Feed in data.
                if task in self.task_map:
                    obs_tensor = self.task_map[task]['obs_tensor']
                    act_op = self.task_map[task]['act_op']
                else:
                    obs_tensor = getattr(self, '{0}_obs_tensor'.format(task))
                    act_op = getattr(self, '{0}_act_op'.format(task))
                feed_dict = {obs_tensor: np.expand_dims(obs[i, t], axis=0)}
                # with tf.device(self.device_string):
                #     output[i, t, :] = self.sess.run(act_op, feed_dict=feed_dict)
                output[i, t, :] = self.sess.run(act_op, feed_dict=feed_dict)

        if task in self.var:
            pol_sigma = np.tile(np.diag(self.var[task]), [N, T, 1, 1])
            pol_prec = np.tile(np.diag(1.0 / self.var[task]), [N, T, 1, 1])
            pol_det_sigma = np.tile(np.prod(self.var[task]), [N, T])
        else:
            var = getattr(self, '{0}_var'.format(task))
            pol_sigma = np.tile(np.diag(var), [N, T, 1, 1])
            pol_prec = np.tile(np.diag(1.0 / var), [N, T, 1, 1])
            pol_det_sigma = np.tile(np.prod(var), [N, T])

        return output, pol_sigma, pol_prec, pol_det_sigma

    def set_ent_reg(self, ent_reg):
        """ Set the entropy regularization. """
        self._hyperparams['ent_reg'] = ent_reg

    def save_model(self, fname):
        # LOGGER.debug('Saving model to: %s', fname)
        self.saver.save(self.sess, fname, write_meta_graph=False)

    def restore_model(self, fname):
        self.saver.restore(self.sess, fname)
        # LOGGER.debug('Restoring model from: %s', fname)

    # For pickling.
    def __getstate__(self):
        with tempfile.NamedTemporaryFile('w+b', delete=True) as f:
            self.save_model(f.name) # TODO - is this implemented.
            f.seek(0)
            with open(f.name, 'r') as f2:
                wts = f2.read()
        return {
            'hyperparams': self._hyperparams,
            'dO': self._dO,
            'dU': self._dU,
            'scale': {task:self.task_map[task]['policy'].scale for task in self.task_map},
            'bias': {task:self.task_map[task]['policy'].bias for task in self.task_map},
            'tf_iter': self.tf_iter,
            'x_idx': {task:self.task_map[task]['policy'].x_idx for task in self.task_map},
            'chol_pol_covar': {task:self.task_map[task]['policy'].chol_pol_covar for task in self.task_map},
            'wts': wts,
        }

    # For unpickling.
    def __setstate__(self, state):
        from tensorflow.python.framework import ops
        ops.reset_default_graph()  # we need to destroy the default graph before re_init or checkpoint won't restore.
        self.__init__(state['hyperparams'], state['dO'], state['dU'])
        for task in self.task_map:
            self.policy[task].scale = state['scale']
            self.policy[task].bias = state['bias']
            self.policy[task].x_idx = state['x_idx']
            self.policy[task].chol_pol_covar = state['chol_pol_covar']
        self.tf_iter = state['tf_iter']

        with tempfile.NamedTemporaryFile('w+b', delete=True) as f:
            f.write(state['wts'])
            f.seek(0)
            self.restore_model(f.name)

