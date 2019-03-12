import numpy as np
import rospy
import time

from std_msgs.msg import Float32MultiArray, String

from tamp_ros.msg import *
from tamp_ros.srv import *


class ValueServer(object):
    def __init__(self, hyperparams):
        import tensorflow as tf
        rospy.init_node('value_update_server')
        hyperparams['policy_opt']['scope'] = 'value'
        self.policy_opt = hyperparams['policy_opt']['type'](
            hyperparams['policy_opt'], 
            hyperparams['dO'],
            hyperparams['dU'],
            hyperparams['dObj'],
            hyperparams['dTarg'],
            hyperparams['dPrimObs'],
            hyperparams['dValObs'],
            hyperparams['prim_bounds']
        )
        self.task = 'value'
        # self.value_service = rospy.Service('qvalue', QValue, self.value)
        self.updater = rospy.Subscriber('value_update', PolicyUpdate, self.update, queue_size=2)
        self.weight_publisher = rospy.Publisher('tf_weights', UpdateTF, queue_size=1)
        self.stop = rospy.Subscriber('terminate', String, self.end, queue_size=1)
        self.stopped = False
        self.time_log = 'tf_saved/'+hyperparams['weight_dir']+'/timing_info.txt'
        self.log_timing = hyperparams['log_timing']
        self.log_publisher = rospy.Publisher('log_update', String, queue_size=1)

        self.update_queue = []

        # rospy.spin()


    # def run(self):
    #     while not self.stopped:
    #         rospy.sleep(0.01)


    # def end(self, msg):
    #     self.stopped = True
    #     rospy.signal_shutdown('Received notice to terminate.')


    # def update(self, msg):
    #     mu = np.array(msg.mu)
    #     mu_dims = (msg.n, msg.rollout_len, msg.dU)
    #     mu = mu.reshape(mu_dims)

    #     obs = np.array(msg.obs)
    #     obs_dims = (msg.n, msg.rollout_len, msg.dValObs)
    #     obs = obs.reshape(obs_dims)

    #     prc = np.array(msg.prc)
    #     prc_dims = (msg.n, msg.rollout_len, msg.dU, msg.dU)
    #     prc = prc.reshape(prc_dims)

    #     wt = msg.wt

    #     start_time = time.time()
    #     update = self.policy_opt.store(obs, mu, prc, wt, 'value')
    #     end_time = time.time()

    #     if update and self.log_timing:
    #         with open(self.time_log, 'a+') as f:
    #             f.write('Time to update value neural net on {0} data points: {1}\n'.format(self.policy_opt.update_size, end_time, start_time))

    #     print 'Weights updated:', update, 'value'
    #     if update:
    #         self.weight_publisher.publish(self.policy_opt.serialize_weights(['value']))


    def run(self):
        while not self.stopped:
            rospy.sleep(0.01)
            self.parse_update_queue()


    def end(self, msg):
        self.stopped = True
        rospy.signal_shutdown('Received notice to terminate.')


    def update(self, msg):
        mu = np.array(msg.mu)
        mu_dims = (msg.n, msg.rollout_len, msg.dU)
        mu = mu.reshape(mu_dims)

        obs = np.array(msg.obs)
        obs_dims = (msg.n, msg.rollout_len, msg.dO)
        obs = obs.reshape(obs_dims)

        prc = np.array(msg.prc)
        prc_dims = (msg.n, msg.rollout_len, msg.dU, msg.dU)
        prc = prc.reshape(prc_dims)

        wt_dims = (msg.n, msg.rollout_len) if msg.rollout_len > 1 else (msg.n,)
        wt = np.array(msg.wt).reshape(wt_dims)
        self.update_queue.append((obs, mu, prc, wt))


    def parse_update_queue(self):
        while len(self.update_queue):
            obs, mu, prc, wt = self.update_queue.pop()
            start_time = time.time()
            update = self.policy_opt.store(obs, mu, prc, wt, 'value')
            end_time = time.time()

            if update and self.log_timing:
                with open(self.time_log, 'a+') as f:
                    f.write('Time to update {0} neural net on {1} data points: {2}\n'.format(self.task, self.policy_opt.update_size, end_time-start_time))

            rospy.sleep(0.01)
            print 'Weights updated:', update, self.task
            # if update:
            #     self.weight_publisher.publish(self.policy_opt.serialize_weights([self.task]))

            if update:
                msg = UpdateTF()
                msg.scope = str(self.task)
                msg.data = self.policy_opt.serialize_weights([self.task])
                self.weight_publisher.publish(msg)


    def value(self, req):
        value = self.policy_opt.value(np.array([req.obs]))
        return QValueResponse(value)
