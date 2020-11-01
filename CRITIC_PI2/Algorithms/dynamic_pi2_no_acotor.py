'''
todo：
1. v_loss 很大，感觉是不是需要归一化以下
2. sigma 如果不降低是不是不太好(为什么方差会很大)
3. 我希望actor能够更新,方差能够自己学
'''
import numpy as np
import gym
import tensorflow as tf
import time
import copy
from tools.env_copy import copy_env
from dynamic_model import Dynamic_Net
from Replay_Buffer import Replay_buffer
from datetime import datetime
import matplotlib.pyplot as plt
import pickle
import tensorflow_probability as tfp
from tools.plot_data import mkdir
#####################  hyper parameters  ######################
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
ENV_NAME = "InvertedDoublePendulum-v1"
# ENV_NAME = "InvertedPendulum-v1"
# ENV_NAME = "InvertedDoublePendulum-v1" # todo
# ENV_NAME = "Walker2d-v1"
# ENV_NAME = "Pendulum-v0"
# ENV_NAME = "Hopper-v1"
TRAIN_FROM_SCRATCH = True # 是否加载模型
MAX_EP_STEPS = 500  # 每条采样轨迹的最大长度
LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.99
VARINANCE =1

VALUE_TRAIN_TIME = 100
ACTOR_TRAIN_TIME = 100
DYNAMIC_TRAIN_TIME = 100

BATCH_SIZE = 32
ROLL_OUTS = 100  # PI2并行采样数
SAMPLE_SIZE = 64  # 训练时采样数，分成minibatch后进行训练
PI2_coefficient = 30
MINI_BATCH = 128 # 训练的时候的minibatch
DEFAULT_EPISODES_NUMBERS = 2  # 每次rollout_train采样多少条轨迹
load_model_path = './offline_data/InvertedDoublePendulum-v1/2020-10-25T22-29-07/model/199'

##################### END hyper parameters  ######################
class PI2_Critic(object):
    def __init__(self, a_dim, s_dim, a_bound, env=None, buffer=None):
        self.global_step = 0                                # 看现在已经交互了多少条轨迹了
        self.env = copy_env(env)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
        config.gpu_options.allow_growth = True  # 程序按需申请内存
        self.sess = tf.Session(config=config)
        self.summary_writer = tf.summary.FileWriter(f"./log/{ENV_NAME}/{TIMESTAMP}/")
        self.dynamic_model = Dynamic_Net(s_dim, a_dim,'dm',sess=self.sess)
        if buffer is not  None:
            self.buffer = buffer
        else:
            self.buffer = Replay_buffer(buffer_size=500)
        if not TRAIN_FROM_SCRATCH:
            self.buffer.load_data()
        ###########################create Actor - Critic network ###########################
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.target_action = tf.placeholder(tf.float32, [None, a_dim])
        self.target_value = tf.placeholder(tf.float32, [None, 1])
        self.current_action = tf.placeholder(tf.float32, [None, a_dim])
        self.generate_sample_from_outside_buffer = False
        with tf.variable_scope('Actor'):
            self.a, self.a_mu = self._build_a(self.S, scope='eval', trainable=True)
        self.action = tf.clip_by_value(tf.squeeze(self.a.sample(1), axis=0), -a_bound[0], a_bound[0])
        self.action_prob = self.a.prob(self.current_action)
        with tf.variable_scope('Critic'):
            self.q = self._build_c(self.S, scope='vtrace', trainable=True)
            self.q_compare = self._build_c(self.S, scope='td_lambda', trainable=True)
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.cv_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/vtrace')
        self.ctd_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/td_lambda')


        self.vtrace_error = tf.losses.mean_squared_error(labels=self.target_value, predictions=self.q)
        self.td_error = tf.losses.mean_squared_error(labels=self.target_value, predictions=self.q_compare)
        self.vtrace_train = tf.train.AdamOptimizer(LR_C).minimize(self.vtrace_error, var_list=self.cv_params)
        self.td_train = tf.train.AdamOptimizer(LR_C).minimize(self.td_error, var_list=self.ctd_params)
        self.a_loss = tf.losses.mean_squared_error(labels=self.target_action, predictions=self.a_mu)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(self.a_loss, var_list=self.ae_params)


        self.reward_per_episode = tf.placeholder(tf.float32)
        self.summary_vtrace = tf.summary.scalar("vtrace loss",self.vtrace_error)
        self.summary_reward = tf.summary.scalar("reward per episode",self.reward_per_episode)
        self.summary_actor= tf.summary.scalar("Actor Loss",self.a_loss)


        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
# ----------------------- build actor-critic model -------------
    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            # 1.2.策略网络第一层隐含层
            a_f1 = tf.layers.dense(inputs=s, units=128, activation=tf.nn.relu, trainable=trainable)
            # 1.3 第二层，均值
            a_mu = a_bound * tf.layers.dense(inputs=a_f1, units=self.a_dim, activation=tf.nn.tanh,
                                             trainable=trainable)
            # 1.3 第二层，标准差
            # a_sigma = tf.layers.dense(inputs=a_f1, units=self.a_dim*self.a_dim, activation=tf.nn.tanh,
            #                                  trainable=trainable)
            # a_sigma = tf.reshape(a_sigma,shape=[-1,self.a_dim,self.a_dim])
            a_sigma = tf.eye(self.a_dim)*tf.constant(VARINANCE,tf.float32)
            normal_dist = tfp.distributions.MultivariateNormalTriL(loc=a_mu,scale_tril=a_sigma)
            # tf.contrib.distributions.MultivariateNormal
            # normal_dist = tf.contrib.distributions.Normal(a_mu, a_sigma)
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
    def save_model(self, model_dir,model_name):
        mkdir(model_dir)
        self.saver.save(self.sess,model_dir+model_name)
    def restore_model(self, model_path=load_model_path):
        self.saver.restore(self.sess, model_path)
    def sample_action(self, s):
        """
        从分布采集一个action的样本
        """
        action = self.sess.run(self.a_mu, {self.S: s})[0]
        return action
    def get_state_value(self, s):
        # 获得状态对应的V
        s = s.reshape([-1, s_dim])
        v0 = self.sess.run(self.q, {self.S: s})
        return v0
    def get_probability(self, s, a):
        # 获得状态动作对应的概率
        return self.sess.run(self.action_prob, {self.current_action: a, self.S: s})
    def scala_value(self,x):
        x = tf.sign(x) * (tf.sqrt(tf.abs(x)+1)-1) + 0.001*x
        return x
    def descala_value(self,x):
        x = tf.sign(x) *(
            (
                (tf.sqrt(1+4*0.001*
                         (
                             tf.abs(x)+1+0.001
                         ))-1)/(2*0.001)
            )**2-1
        )
        return x
# ----------------------  rollout data and store data ---------------
    def parse_episode(self, epi):
        """
        输入一串episode，该函数会按顺序返回states，actions，reward， probability
        """
        epi = copy.deepcopy(epi)
        length = len(epi)
        states = np.zeros([length, s_dim])
        actions = np.zeros([length,self.a_dim])
        rewards = np.zeros([length])
        next_states = np.zeros([length, s_dim])
        probs = np.zeros([length])
        for i in range(length):
            pair = copy.deepcopy(epi[i])
            state = pair[:s_dim]
            action =pair[s_dim:s_dim + a_dim]
            reward = pair[s_dim + a_dim: s_dim + a_dim+1]
            next_state = pair[s_dim + a_dim+1: s_dim + a_dim+1+s_dim]
            probability = pair[s_dim + a_dim+1+s_dim]
            states[i] = state
            actions[i][:] = action
            rewards[i] = reward
            next_states[i] = next_state
            probs[i] = probability
        return states, actions, rewards, next_states, probs
    def rollout_one_path(self, max_steps_per_episodes=MAX_EP_STEPS):
        """
        跟环境交互一条轨迹然后返回轨迹(s,a,r,s_)以及总回报(不含有折扣）
        """
        obs = self.env.reset()
        done = False
        R = 0
        t = 0
        path = []
        while not done and t < max_steps_per_episodes:
            act = self.dypi2(initial_state=obs)
            new_obs, r, done, _ = self.env.step(act)
            prob = self.get_probability(obs.reshape([-1, s_dim]), act.reshape([-1, a_dim]))
            temp_transition = np.hstack((obs, act, [r], new_obs, prob[0]))
            path.append(temp_transition)
            R += r
            t += 1
            obs = new_obs
        print(f"rollout the {self.global_step}th episode | steps: {t} | total reward: {R} | mean reward: {R/t}")

        return path, R
    def rollout(self, episodes_numbers=DEFAULT_EPISODES_NUMBERS, max_steps_per_episodes=None):
        """
        跟环境交互然后产生num_episodes条轨迹,注意我们这里可以限制每条轨迹最大长度
        """
        print(f"start rollout the {episodes_numbers} episodes from environment")
        count = 0
        returnsum = 0
        path_number = 0
        while (path_number < episodes_numbers):
            path, path_return = self.rollout_one_path(max_steps_per_episodes=max_steps_per_episodes)
            self.buffer.store_episode(path, path_return)
            count += len(path)
            returnsum += path_return
            path_number += 1
            self.global_step +=1
            summary = self.sess.run(self.summary_reward, feed_dict={self.reward_per_episode:
                                                                          path_return})
            self.summary_writer.add_summary(summary, global_step=self.global_step)
        avg_return = returnsum / path_number
        return avg_return, path_number, count
# ---------------------- model training ------------
    def sample_dynamic(self, episodes):
        """
        从buffer中取数据给dynamic训练，数据格式是s_t,a_t,s_{t+1}-s_t
        end to end 的数据
        """
        episodes_dynamics = []
        episodes_sactions = []
        for episode in episodes:
            epi = copy.deepcopy(episode)
            length = len(epi)-1        # todo:这个是为啥子？
            s_actions = np.zeros([length, s_dim + a_dim])
            dynamics =  np.zeros([length, s_dim])
            for i in range(length):
                pair = copy.deepcopy(epi[i])
                state = pair[:s_dim]
                s_action = pair[:s_dim + a_dim]
                next_state = pair[s_dim + a_dim + 1: s_dim + a_dim + 1 + s_dim]
                s_actions[i] = s_action
                dynamics[i] = next_state - state
                episodes_dynamics.append(copy.deepcopy(dynamics))
                episodes_sactions.append(copy.deepcopy(s_actions))
        return episodes_dynamics, episodes_sactions
    def sample_data(self, episodes):
        """
        从buffer得到训练AC网络的数据，s_t->v(s_t), s_t -> a_t
        """
        vtrace_values = []
        updated_vtrace_values = []
        episodes_states = []
        episodes_actions = []
        episodes_probs = []
        for episode in episodes:
            states, actions, rewards, _, probs = self.parse_episode(episode)
            val_t0= self.get_state_value(states)
           # td_target = self.compute_return(episode, val_t1, rewards)
            vtrace_target = self.compute_vtrace_target(states, val_t0, rewards, probs, actions)
            vtrace_values.append(val_t0)
          #  td_values.append(val_t1)
            updated_vtrace_values.append(vtrace_target)
        #    updated_td_values.append(td_target)
            episodes_states.append(states)
            episodes_actions.append(actions)
            episodes_probs.append(probs)
        return episodes_states, vtrace_values, updated_vtrace_values, episodes_actions
    def update(self, update_type = 1):
        if update_type == 1:
            self.train_critic_network()
            self.train_dynamic_network()
            self.train_actor_network()
        elif update_type == 2:
            self.train_dynamic_network()
            self.train_critic_network()
        elif update_type==3:
            self.train_actor_network()
    def train(self, update_type = 1):
        self.rollout(episodes_numbers=DEFAULT_EPISODES_NUMBERS, max_steps_per_episodes=MAX_EP_STEPS)
        self.update(update_type=update_type)
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
            vtrace_target[i] = curr_v
        return vtrace_target
    def train_critic_network(self, n_sample_size=SAMPLE_SIZE, traintime=VALUE_TRAIN_TIME):
        n_episodes = self.buffer.get_length()
        indices = np.random.choice(n_episodes, size=n_sample_size, )  # p=weight)  # 随机生成序号
        episodes = []
        for i in indices:
            episodes.append(copy.deepcopy(self.buffer.buffer_data[i]))
        episodes_states, values, vtrace_values, _ = self.sample_data(copy.deepcopy(episodes))
        data_number = len(values)
        perm = np.random.permutation(data_number)
        minibatch = data_number
        total_states = []
        target_values_v = []

        for i in range(0, data_number, minibatch):
            for j in perm[i:i + minibatch]:
                for k in range(len(episodes_states[j])):
                    total_states.append(copy.deepcopy(episodes_states[j][k]))
                    target_values_v.append(copy.deepcopy([vtrace_values[j][k]]))

            total_states = np.array(total_states)
            target_values_v = np.array(target_values_v)
        for value_train_times in range(traintime):
            summary,_ = self.sess.run([self.summary_vtrace,self.vtrace_train],
                                        feed_dict={self.target_value: target_values_v, self.S: total_states})
            self.summary_writer.add_summary(summary=summary,global_step=self.global_step)
    def train_actor_network(self, n_sample_size=SAMPLE_SIZE, traintime=ACTOR_TRAIN_TIME):
        n_episodes = self.buffer.get_length()
        if n_sample_size > n_episodes:
            n_sample_size = n_episodes
        indices = np.random.choice(n_episodes, size=n_sample_size)  # 随机生成序号
        episodes = []
        for i in indices:
            episodes.append(copy.deepcopy(self.buffer.buffer_data[i]))
        episodes_states, values, _, episodes_actions = self.sample_data(copy.deepcopy(episodes))
        data_number = len(values)
        perm = np.random.permutation(data_number)
        # Using BGD
        minibatch = data_number
        target_actions = None
        total_states = None
        for i in range(0, data_number, minibatch):
            target_actions = []
            total_states = []
            for j in perm[i:i + minibatch]:
                for k in range(len(episodes_states[j])):
                    total_states.append(episodes_states[j][k])
                    target_actions.append(episodes_actions[j][k])
            total_states = np.array(total_states)
            target_actions = np.array(target_actions)
        for value_train_times in range(traintime):
                summary,_ = self.sess.run([self.summary_actor,self.atrain], {self.target_action: target_actions, self.S: total_states})
                self.summary_writer.add_summary(summary,global_step=self.global_step)
    def train_dynamic_network(self, n_sample_size=SAMPLE_SIZE, traintime=DYNAMIC_TRAIN_TIME):
        n_episodes = self.buffer.get_length()
        if n_sample_size > n_episodes:
            n_sample_size = n_episodes
        #dloss = 0
        indices = np.random.choice(n_episodes, size=n_sample_size)  # 随机生成序号
        episodes = []
        for i in indices:
            episodes.append(copy.deepcopy(self.buffer.buffer_data[i]))
        episodes_dynamics, episodes_sactions = self.sample_dynamic(copy.deepcopy(episodes))
        data_number = len(episodes_dynamics)
        perm = np.random.permutation(data_number)
        # Using BGD
        minibatch = data_number  # 轨迹的数量
        target_sactions = []
        target_dynamics = []
        for i in range(0, data_number, minibatch):
            for j in perm[i:i + minibatch]:
                for k in range(len(episodes_dynamics[j])):
                    target_dynamics.append(episodes_dynamics[j][k])
                    target_sactions.append(episodes_sactions[j][k])
            summary = self.dynamic_model.learn(np.array(target_sactions), np.array(target_dynamics),
                                               EPOCH=traintime)
            self.summary_writer.add_summary(summary=summary, global_step=self.global_step)
            target_sactions = []
            target_dynamics = []
# --------------------- different algorithms ---------
    def dypi2(self, initial_state, iteration_times=5):
        current_best_action = None
        current_best_value = -np.inf

        initial_action = self.sample_action(
            initial_state.reshape([-1, s_dim]))

        for i in range(iteration_times):
            sigma = np.ones([ROLL_OUTS, self.a_dim])*VARINANCE
            sigma[0] = np.zeros_like(sigma[0])
            action_groups = np.squeeze(
                np.clip(np.random.normal(loc=initial_action,scale=sigma), -self.a_bound[0], self.a_bound[0]))
            action_groups = action_groups.reshape(ROLL_OUTS, self.a_dim)
            next_stages = []
            rewards = []
            dones = []
            for j in range(len(action_groups)):
                s_a = np.zeros([1, s_dim + a_dim])
                s_a[0][:s_dim] = initial_state
                s_a[0][s_dim: s_dim + a_dim] = action_groups[j]
                s_a = s_a.reshape([-1, s_dim + a_dim])
                temp_next_state, temp_reward, done = self.dynamic_model.prediction(s_a)
                dones.append(done[0])
                next_stages.append(temp_next_state)
                rewards.append(temp_reward)
            state_groups = np.array(next_stages)
            rewards = np.array(rewards)
            next_values_v = np.array(self.get_state_value(state_groups))
            next_values = next_values_v

            values = rewards.reshape([ROLL_OUTS, 1]) + next_values
            probability_weighting = np.zeros((ROLL_OUTS, 1), dtype=np.float64)
            exponential_value_loss = np.zeros((ROLL_OUTS, 1), dtype=np.float64)
            maxv = np.max(values, axis=0)
            minv = np.min(values, axis=0)
            if (maxv - minv) < 1e-4:
                probability_weighting[:] = 1.0 / ROLL_OUTS
            else:
                for k in range(exponential_value_loss.shape[0]):
                    res = (maxv - values[k]) / (maxv - minv)
                    exponential_value_loss[k] = res
                probability_weighting[:] = exponential_value_loss[:] / np.sum(exponential_value_loss)  # 计算soft_max概率
            hybrid_action = np.squeeze(np.dot(action_groups.T, probability_weighting)) # 去掉多余维度
            s_a = np.zeros([1, s_dim + a_dim])
            s_a[0][:s_dim] = initial_state
            s_a[0][s_dim: s_dim + a_dim] = hybrid_action
            temp_next_state, temp_reward, done = self.dynamic_model.prediction(s_a)
            # if not done:
            next_values_v = self.get_state_value(np.array(temp_next_state).reshape([1, self.s_dim]))
            # else:
            #     next_values_v = 0
            next_values = next_values_v

            hybrid_value = temp_reward + next_values
            if hybrid_value >= maxv:
                current_action = hybrid_action
                current_value = hybrid_value
            else:
                current_action = action_groups[np.argmax(values)]
                current_value = maxv
            if current_best_value < current_value:
                current_best_value = copy.deepcopy(current_value)
                current_best_action = copy.deepcopy(current_action)
            initial_action = hybrid_action
        current_best_action = np.reshape(current_best_action,[self.a_dim])
        return current_best_action
    def test(self,test_time=3,if_render=False):
        """
        测试当前agent的performance！
        """
        print(f"start test for {test_time} episodes using ")
        ave_reward = 0
        ave_time = 0
        for i in range(test_time):
            total_reward = 0
            obs = self.env.reset()
            done = False
            t = 0
            while (not done) and (t <= MAX_EP_STEPS):
                act = self.dypi2(initial_state=obs)
                new_obs, r, done, _ = self.env.step(act)
                total_reward += r
                t += 1
                obs = new_obs
                if if_render:
                    self.env.render()
                print(t)
            ave_reward += total_reward
            ave_time += t
        ave_reward = ave_reward / test_time
        ave_time = ave_time / test_time
        return ave_reward, ave_time

# ---------------------- plot data and result --------
if __name__ == '__main__':
    # ---------------------------- env info ------------------------------
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)
    epochs = int(1e4)
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high.shape
    print(f"s dim: {s_dim} | a dim : {a_dim} | a_bound : {a_bound}")
    # --------------------------- interact info -------------------
    s = env.reset()
    pi2_critic = PI2_Critic(a_dim, s_dim, a_bound, env)
    normal_rewards = []
    ############################TRAINING#########################
    print("============Start interact============")
    for epoch in range(epochs):
        pi2_critic.train(update_type=2)
        if (epoch + 1)%10 == 0:
            try:
                print("start save data")
                pi2_critic.save_model(model_dir=f"./offline_data/{ENV_NAME}/{TIMESTAMP}/model/",model_name=f"{epoch}")
                pi2_critic.buffer.save_data(model_dir=f'./offline_data/{ENV_NAME}/{TIMESTAMP}/buffer_data/',model_name=f"{epoch}")
            except:
                print('data or figure save failed')
    # #############################TEST############################
    # pi2_critic.restore_model(model_path="offline_data/2020-10-25T22-29-07/model/199")
    # pi2_critic.test(test_time=1,if_render=True)