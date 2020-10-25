import numpy as np
import gym
import tensorflow as tf
import time
import copy
import pickle
from tools.env_copy import copy_env
import matplotlib.pyplot as plt
from dynamic_model import Dynamic_Net
###2020/07/08###
#####TODO: This problem is only used for offpolicy learning, once loaded the database file, it will not store episodes that created by its own.
#####################  hyper parameters  ######################
from Replay_Buffer import Replay_buffer

TRAIN_FROM_SCRATCH = True # 是否加载模型
MAX_EP_STEPS = 100  # 每条采样轨迹的最大长度
LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.99
# s_dim = env.observation_space.shape[0]
# a_dim = env.action_space.shape[0]
# a_bound = env.action_space.high.shape
VALUE_TRAIN_TIME = 100
ACTOR_TRAIN_TIME = 100
DYNAMIC_TRAIN_TIME = 100

# reward discount
# TAU = 0.01      # soft replacement
TRAIN_TIME = 300
MEMORY_CAPACITY = 15000
BATCH_SIZE = 32
ROLL_OUTS = 20  # PI2并行采样数
SAMPLE_SIZE = 128  # 训练时采样数，分成minibatch后进行训练
ENV_NAME = "InvertedDoublePendulum-v1"
PI2_coefficient = 30
MINI_BATCH = 1 # 训练的时候的minibatch
NUM_EPISODES = 2  # 每次rollout_train采样多少条轨迹
load_model_path = './mpc1023/models.ckpt'
save_model_path = './mpc1023/models.ckpt'
"""
=========================流程==================================
self.learn()函数包含一次采样（rollout_train）和一次训练（update）
rollout_trian函数使用self.pi2_critic函数选取动作，这个函数输入状态，根据actor产生动作，然后使用PI2的方法产生合成动作
update分为critic update和action update。
"""


class PI2_Critic(object):
    def __init__(self, a_dim, s_dim, a_bound, env=None, buffer=None):
        self.dynamic_memory = np.zeros((MEMORY_CAPACITY, s_dim + s_dim + a_dim), dtype=np.float32)
        # 1(the last dimension) for reward
        self.num_episodes = NUM_EPISODES
        self.minibatch = MINI_BATCH
        self.sample_size = SAMPLE_SIZE
        self.trainfromscratch = TRAIN_FROM_SCRATCH
        self.sess = tf.Session()
        self.env = copy_env(env)
        self.reset_env = copy_env(env)
        self.globaltesttime = 0
        self.vtrace_losses = []
        self.dlosses = []
        self.alosses = []
        self.dynamic_model = Dynamic_Net(s_dim, a_dim,'dm')
        if buffer == None:
            self.buffer = Replay_buffer(buffer_size=200)
        else:
            self.buffer = buffer
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.target_action = tf.placeholder(tf.float32, [None, self.a_dim])
        self.target_value = tf.placeholder(tf.float32, [None, 1])
        self.current_action = tf.placeholder(tf.float32, [None, self.a_dim])
        self.generate_sample_from_outside_buffer = False
        with tf.variable_scope('Actor'):
            self.a, self.a_mu = self._build_a(self.S, scope='evaal', trainable=True)
        self.action = tf.clip_by_value(tf.squeeze(self.a.sample(1), axis=0), -a_bound[0], a_bound[0])
        self.action_prob = self.a.prob(self.current_action)
        with tf.variable_scope('Critic'):
            self.q = self._build_c(self.S, scope='vtrace', trainable=True)
            self.q_compare = self._build_c(self.S, scope='td_lambda', trainable=True)
        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/evaal')
        self.cv_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/vtrace')
        self.ctd_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/td_lambda')

        self.vtrace_error = tf.losses.mean_squared_error(labels=self.target_value, predictions=self.q)
        self.td_error = tf.losses.mean_squared_error(labels=self.target_value, predictions=self.q_compare)
        self.vtrace_train = tf.train.AdamOptimizer(LR_C).minimize(self.vtrace_error, var_list=self.cv_params)
        self.td_train = tf.train.AdamOptimizer(LR_C).minimize(self.td_error, var_list=self.ctd_params)
        self.a_loss = tf.losses.mean_squared_error(labels=self.target_action, predictions=self.a_mu)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(self.a_loss, var_list=self.ae_params)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if not self.trainfromscratch:
            self.buffer.load_data()

    def rollout_paths(self, test=True, max_length=MAX_EP_STEPS):
        # 采样一条轨迹
        # Warning: maxlength和Max_EP是一个东西， 都是200
        obs = self.env.reset()
        done = False
        R = 0
        t = 0
        path = []
        while not done:
            if t >= max_length:
                break
            act = self.MPC(obs, self.env)
            new_obs, r, done, _ = self.env.step(act)
            prob = self.get_probability(obs.reshape([-1, s_dim]), act.reshape([-1, a_dim]))
            temp_transition = np.hstack((obs, act, [r], new_obs, prob[0]))
            path.append(temp_transition)
            R += r  # TODO: 是否需要加gamma算折扣累计回报?
            t += 1
            obs = new_obs
        print('sample path reward', R)
        return path, R

    def rollout_train(self, num_episodes=NUM_EPISODES, max_length=None):
        # 采样num_episode条轨迹
        print('=============================Sampling=============================')
        count = 0
        returnsum = 0
        path_number = 0
        while (path_number < num_episodes):
            print('==============Generating Paths==============')
            path, path_return = self.rollout_paths(test=False, max_length=max_length)
            self.store_path(path, path_return)
            count += len(path)
            returnsum += path_return
            path_number += 1
        avg_return = returnsum / path_number
        return avg_return, path_number, count

    def store_path(self, path, path_return):
        # 将轨迹存储近buffer
        pointer = self.buffer.store_episode(path, path_return)
        return pointer

    def save_model(self, model_path=save_model_path):
        self.saver.save(self.sess, model_path)

    def restore_model(self, model_path=load_model_path):
        self.saver.restore(self.sess, model_path)

    def choose_action(self, s):
        s = s.reshape(1, -1)
        return self.sess.run(self.a, {self.S: s})[0]

    def sample_action(self, s):
        action = self.sess.run(self.a_mu, {self.S: s})[0]
        return action

    def get_state_value(self, s):
        # 获得状态对应的V
        s = s.reshape([-1, self.s_dim])
        v0 =  self.sess.run(self.q, {self.S: s})
        #v1 = self.sess.run(self.q_compare, {self.S: s})
        return v0

    def get_probability(self, s, a):
        # 获得状态动作对应的概率
        return self.sess.run(self.action_prob, {self.current_action: a, self.S: s})


    def parse_episode(self, epi):
        # 输入一串episode，该函数会按顺序返回states，actions，reward， probability
        epi = copy.deepcopy(epi)
        length = len(epi)
        states = np.zeros([length, self.s_dim])
        actions = np.zeros([length,1])
        rewards = np.zeros([length])
        next_states = np.zeros([length, self.s_dim])
        probs = np.zeros([length])
        for i in range(length):
            pair = copy.deepcopy(epi[i])
            state = pair[:self.s_dim]
            action =pair[self.s_dim:self.s_dim + self.a_dim]
            reward = pair[self.s_dim + self.a_dim: self.s_dim + self.a_dim+1]
            next_state = pair[self.s_dim + self.a_dim+1: self.s_dim + self.a_dim+1+self.s_dim]
            probability = pair[self.s_dim + self.a_dim+1+self.s_dim]
            states[i] = state
            actions[i][0] = action
            rewards[i] = reward
            next_states[i] = next_state
            probs[i] = probability
        return states, actions, rewards, next_states, probs

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            # 1.2.策略网络第一层隐含层
            a_f1 = tf.layers.dense(inputs=s, units=128, activation=tf.nn.relu, trainable=trainable)
            # 1.3 第二层，均值
            a_mu = self.a_bound * tf.layers.dense(inputs=a_f1, units=self.a_dim, activation=tf.nn.tanh,
                                             trainable=trainable)
            # 1.3 第二层，标准差
            a_sigma = tf.ones(1)
            normal_dist = tf.contrib.distributions.Normal(a_mu, a_sigma)
            # 根据正态分布采样一个动作
        return normal_dist, a_mu

    def _build_c(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net_1 = tf.layers.dense(inputs=s, units=128, activation=tf.nn.relu,
                                    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                    bias_initializer=tf.constant_initializer(0.1),
                                    trainable=trainable)
            net_2 = tf.layers.dense(inputs=net_1, units=64, activation=tf.nn.relu,
                                    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                    bias_initializer=tf.constant_initializer(0.1),
                                    trainable=trainable)
            return tf.layers.dense(net_2, 1, trainable=trainable)  # Q(s,a)

    def compute_vtrace_target(self, states, val_t, rewards, probs, actions):
        path_len = len(states)
        vtrace_target = np.zeros(path_len)
        # truncated_c_rho_threshold = 1.0
        truncated_rho_threshold = 1.0
        pi = np.squeeze(np.array(self.get_probability(states, actions)))
        mu = np.array(copy.deepcopy(probs))
        iw = pi / mu
        last_val = rewards[-1]
        vtrace_target[-1] = last_val
        for i in reversed(range(0, path_len - 1)):
            curr_r = rewards[i]
            next_v = vtrace_target[i + 1]
            weight = min(iw[i], truncated_rho_threshold)
            curr_v = val_t[i] + weight*(curr_r + GAMMA*val_t[i+1] - val_t[i]) + GAMMA*weight*(next_v - val_t[i+1])
            #curr_v = GAMMA * weight * (curr_r + next_v) + (1 - GAMMA * weight) * val_t[i]
            vtrace_target[i] = curr_v
        return vtrace_target

    def sample_dynamic(self, episodes):
        episodes_dynamics = []
        episodes_sactions = []
        for episode in episodes:
            # 输入一串episode，该函数会按顺序返回states，actions，reward， probability
            epi = copy.deepcopy(episode)
            length = len(epi) - 1
            s_actions = np.zeros([length, self.s_dim + self.a_dim])
            dynamics = np.zeros([length, self.s_dim])

            for i in range(length):
                pair = copy.deepcopy(epi[i])
                state = pair[:self.s_dim]
                s_action = pair[:self.s_dim + self.a_dim]
                next_state = pair[self.s_dim + self.a_dim + 1: self.s_dim + self.a_dim + 1 + self.s_dim]
                s_actions[i] = s_action
                dynamics[i] = next_state - state
                episodes_dynamics.append(copy.deepcopy(dynamics))
                episodes_sactions.append(copy.deepcopy(s_actions))
        return episodes_dynamics, episodes_sactions

    def dynamic_training(self, n_sample_size=SAMPLE_SIZE, traintime=DYNAMIC_TRAIN_TIME):

        minibatch = self.minibatch
        n_episodes = self.buffer.get_length()
        if n_sample_size > n_episodes:
            n_sample_size = n_episodes
            print("======Warning: buffer size is less than sample size, need to store more data========")
        #dloss = 0
        indices = np.random.choice(n_episodes, size=n_sample_size)  # 随机生成序号
        episodes = []
        for i in indices:
            episodes.append(copy.deepcopy(self.buffer.buffer_data[i]))
        episodes_dynamics, episodes_sactions = self.sample_dynamic(copy.deepcopy(episodes))
        data_number = len(episodes_dynamics)
        perm = np.random.permutation(data_number)
        # Using BGD
        minibatch = data_number
        target_sactions = []
        total_dynamics = []
        for i in range(0, data_number, minibatch):
            for j in perm[i:i + minibatch]:
                for k in range(len(episodes_dynamics[j])):
                    total_dynamics.append(episodes_dynamics[j][k])
                    target_sactions.append(episodes_sactions[j][k])
            target_dynamics = np.array(total_dynamics)
            target_sactions = np.array(target_sactions)
        for value_train_times in range(traintime):
            dloss = self.dynamic_model.learn(target_sactions, target_dynamics)
            if (value_train_times+1)%10 == 0:
                #dloss = dloss / value_train_times
                self.dlosses.append(dloss)
                print('dynamic is ', dloss)
        return dloss

    def update(self, update_type = 1):
        self.dynamic_training()
    def learn(self):
        # 采样+学习

        self.rollout_train(num_episodes=self.num_episodes, max_length=MAX_EP_STEPS)
        self.update()

    def test(self,test_time=3, use_vtrace=False):
        # 测试，use_hybrid_action表示是否使用PI2动作

        ave_reward = 0
        ave_time = 0
        for i in range(test_time):
            total_reward = 0
            obs = self.env.reset()
            done = False
            t = 0
            while (not done) and (t <= MAX_EP_STEPS):
                # time_start = time.time()
                act = self.MPC(obs, self.env)
                new_obs, r, done, _ = self.env.step(act)
                total_reward += r
                t += 1
                obs = new_obs
                # time_end = time.time()
            ave_reward += total_reward
            ave_time += t
        ave_reward = ave_reward / test_time
        ave_time = ave_time / test_time
        return ave_reward, ave_time

    def MPC(self, initial_start, env, iteration_times=5):
        initial_action = self.sample_action(initial_start.reshape(
                [-1, self.s_dim]))  # initial action is supposed to be [action_dim,] narray object
        sigma = np.ones([(ROLL_OUTS + 1) * iteration_times, self.a_dim])
        sigma[0] = np.zeros_like(sigma[0])
        action_groups = np.squeeze(
            np.clip(np.random.normal(initial_action, sigma), -self.a_bound[0], self.a_bound[0]))
        action_groups = action_groups.reshape((ROLL_OUTS + 1) * iteration_times, self.a_dim)
        rewards = np.zeros([len(action_groups)])

        for j in range(len(action_groups)):
            r = 0
            a = copy.deepcopy(action_groups[j][0])
            s_a = np.zeros([1, self.s_dim + self.a_dim])
            s_a[0][:self.s_dim] = initial_start
            s_a[0][self.s_dim: self.s_dim + self.a_dim] = a
            temp_next_state, temp_reward, done = self.dynamic_model.prediction(s_a)
            r += temp_reward
            timer = 0

            while not done or timer < 100:
                a = self.sample_action(temp_next_state.reshape([-1, self.s_dim]))
                timer += 1
                s_a = np.zeros([1, self.s_dim + self.a_dim])
                s_a[0][:self.s_dim] = temp_next_state
                s_a[0][self.s_dim: self.s_dim + self.a_dim] = a
                temp_next_state, temp_reward, done = self.dynamic_model.prediction(s_a)
                r += temp_reward
            rewards[j] = r
        values = rewards.reshape([(ROLL_OUTS + 1) * iteration_times, 1])
        maxv = np.max(values, axis=0)
        best_action = action_groups[np.argmax(values)]
        best_value = maxv

        return best_action

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed()
epoch = int(1e4)
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high.shape
print("action bound is", a_bound)
s = env.reset()
pi2_critic = PI2_Critic(a_dim, s_dim, a_bound, env)

normal_rewards = []
for i in range(50000):
    pi2_critic.learn()
    if (i+1) % 1== 0:
        print('======================start testing=========================')
        r,t = pi2_critic.test(use_vtrace=True)
        print('==========PI2_Critic ', r,' steps',t,'=============')
        normal_rewards.append(r)
    if (i+1)%10 == 0:
        try:
            print('total interactions:', pi2_critic.buffer.get_total_interactions(), 'total trajectories:',
                  pi2_critic.buffer.get_length())
            pi2_critic.save_model()
            pi2_critic.buffer.save_data(model_path='./mpc1023/buffer_data')
            print('data saved successfully')
            plt.plot(normal_rewards, label='MPC')
            with open('./mpc1023/mpc_data', 'wb') as f:
                pickle.dump(normal_rewards, f, pickle.HIGHEST_PROTOCOL)
            plt.xlabel('every 2 new trajectories with env', fontsize=16)
            plt.ylabel('scores', fontsize=16)
            plt.legend()
            plt.savefig('./mpc1023/mpc.png')
            plt.clf()
            print('plot save successed')
        except:
            print('figure save failed')