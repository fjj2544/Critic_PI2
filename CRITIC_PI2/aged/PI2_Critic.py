import numpy as np
import gym
import tensorflow as tf
import time
import copy
import pickle
from tools.env_copy import copy_env
import matplotlib.pyplot as plt
from PI2_replay_buffer import Replay_buffer
###########################################################################################  hyper parameters  ####################################################
TRAIN_FROM_SCRATCH = True  # 是否加载模型
MAX_EP_STEPS = 200  # 每条采样轨迹的最大长度
LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.99

VALUE_TRAIN_TIME = 50
ACTOR_TRAIN_TIME = 50
# reward discount
# TAU = 0.01      # soft replacement
TRAIN_TIME = 5
MEMORY_CAPACITY = 15000
BATCH_SIZE = 32
ROLL_OUTS = 20  # PI2并行采样数
SAMPLE_SIZE = 32  # 训练时采样数，分成minibatch后进行训练
ENV_NAME = "InvertedDoublePendulum-v1"
PI2_coefficient = 30
MINI_BATCH = 4  # 训练的时候的minibatch
NUM_EPISODES = 2  # 每次rollout_train采样多少条轨迹
model_path = './data_0716/models.ckpt'

###########################################################################END hyper parameters####################################################################################################
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
        self.closses = []
        self.alosses = []
        if buffer == None:
            self.buffer = Replay_buffer()
        else:
            self.buffer = buffer
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
            self.q = self._build_c(self.S, scope='eval', trainable=True)
        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        # 定义损失函数和优化器
        self.td_error = tf.losses.mean_squared_error(labels=self.target_value, predictions=self.q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(self.td_error, var_list=self.ce_params)
        self.a_loss = tf.losses.mean_squared_error(labels=self.target_action, predictions=self.a_mu)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(self.a_loss, var_list=self.ae_params)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if not self.trainfromscratch:
            self.restore_model(model_path)
            self.buffer.load_data(model_path)

    def save_model(self, model_path=model_path):
        self.saver.save(self.sess, model_path)

    def restore_model(self, model_path=model_path):
        self.saver.restore(self.sess, model_path)

    def choose_action(self, s):
        # 这俩函数差不多，，，，choose action为单排state做了reshape处理
        s = s.reshape(1, -1)
        return self.sess.run(self.action, {self.S: s})[0]

    def sample_action(self, s):

        action = self.sess.run(self.a_mu, {self.S: s})[0]
        return action

    def get_state_value(self, s):
        # 获得状态对应的V
        return self.sess.run(self.q, {self.S: s})

    def get_probability(self, s, a):
        # 获得状态动作对应的概率，用于重要性采样
        return self.sess.run(self.action_prob, {self.current_action: a, self.S: s})

    def test(self, use_hybrid_method=False):
        # 测试，use_hybrid_action表示是否使用PI2动作
        # 采十条取均值
        ave_reward = 0
        ave_time = 0
        for i in range(3):
            total_reward = 0
            obs = self.env.reset()
            done = False
            t = 0
            while (not done) and (t <= MAX_EP_STEPS):
              #  time_start = time.time()
                if use_hybrid_method:
                    act = self.pi2_with_critic(obs, self.env)
                else:
                    act = self.choose_action(obs)
                new_obs, r, done, _ = self.env.step(act)
                total_reward += r
                t += 1
                obs = new_obs
                time_end = time.time()
                # print('totally cost:', time_end - time_start)
            ave_reward += total_reward
            ave_time += t
        ave_reward = ave_reward / 10
        ave_time = ave_time / 10
        return ave_reward, ave_time

    def learn(self):
        # 采样+学习
        self.rollout_train(num_episodes=self.num_episodes, max_length=MAX_EP_STEPS)
        self.update()

    def parse_episode(self, epi):
        # 输入一串episode，该函数会按顺序返回states，actions，reward， probability
        epi = copy.deepcopy(epi)
        states = []
        actions = []
        rewards = []
        next_states = []
        probs = []
        for pair in epi:
            state = copy.deepcopy(pair[:s_dim])
            action = copy.deepcopy(pair[s_dim:s_dim + a_dim])
            reward = copy.deepcopy(pair[-s_dim - 1:-s_dim])
            next_state = copy.deepcopy(pair[-s_dim:-1])
            probability = copy.deepcopy(pair[-1])
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            probs.append(probability)
        return states, actions, rewards, next_states, probs

    def _build_a(self, s, scope, trainable):
        # 创建动作网络
        # 注意，老师认为在测试的时候，使用均值来和PI2动作比较，其余场合还是用高斯分布动作（即加入方差）
        with tf.variable_scope(scope):
            # 1.2.策略网络第一层隐含层
            a_f1 = tf.layers.dense(inputs=s, units=128, activation=tf.nn.relu, trainable=trainable)
            # 1.3 第二层，均值
            a_mu = a_bound * tf.layers.dense(inputs=a_f1, units=self.a_dim, activation=tf.nn.tanh,
                                             trainable=trainable)
            # 1.3 第二层，标准差
            a_sigma = tf.ones(1)
            normal_dist = tf.contrib.distributions.Normal(a_mu, a_sigma)
            # 根据正态分布采样一个动作
        return normal_dist, a_mu

    def _build_c(self, s, scope, trainable):
        # 值函数网络
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
            act = self.pi2_with_critic(obs, self.env)
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

    def compute_vtrace_target(self, states, val_t, rewards, probs, actions):
        # 后向视角计算vtrace值，后面用于做标签监督学习值函数
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
            curr_v = val_t[i] + weight * (curr_r + GAMMA * val_t[i + 1] - val_t[i]) + GAMMA * weight * (
                    next_v - val_t[i + 1])
            vtrace_target[i] = curr_v
        return vtrace_target

    def update(self, n_sample_size=SAMPLE_SIZE):
        self.critic_training()
        self.actor_training()

    # def critic_training(self, n_sample_size=SAMPLE_SIZE, traintime=VALUE_TRAIN_TIME):
    #     minibatch = self.minibatch
    #     n_episodes = self.buffer.get_length()
    #     indices = np.random.choice(n_episodes, size=n_episodes )  # p=weight)  # 随机生成序号
    #     episodes = []
    #     for i in indices:
    #         episodes.append(copy.deepcopy(self.buffer.buffer_data[i]))
    #     episodes_states, values, vtrace_values, _ = self.sample_data(copy.deepcopy(episodes))
    #     data_number = len(values)
    #     perm = np.random.permutation(data_number)
    #     minibatch = data_number
    #     total_states = []
    #     target_values_v = []
    #
    #     for i in range(0, data_number, minibatch):
    #         for j in perm[i:i + minibatch]:
    #             for k in range(len(episodes_states[j])):
    #                 total_states.append(copy.deepcopy(episodes_states[j][k]))
    #                 target_values_v.append(copy.deepcopy([vtrace_values[j][k]]))
    #
    #         total_states = np.array(total_states)
    #         target_values_v = np.array(target_values_v)
    #     for value_train_times in range(traintime):
    #         self.sess.run(self.vtrace_train, {self.target_value: target_values_v, self.S: total_states})
    #         vtrace_loss = self.sess.run(self.vtrace_error, {self.target_value: target_values_v, self.S: total_states})
    #         if (value_train_times+1)%5 == 0:
    #
    #             print("vtrace loss is ", vtrace_loss)
    #         self.vtrace_losses.append(vtrace_loss)
    #     return vtrace_loss / traintime
    def critic_training(self, n_sample_size=SAMPLE_SIZE, traintime=VALUE_TRAIN_TIME):
        closs = 0
        minibatch = self.minibatch
        n_episodes = self.buffer.get_length()
        for value_train_times in range(traintime):
            # 每次取n_sample_size条轨迹，分成minibatch次进行minibatch梯度下降
            # weight = self.buffer.rewards[:n_episodes]/np.sum(self.buffer.rewards)
            indices = np.random.choice(n_episodes, size=n_episodes )  # p=weight)  # 随机生成序号
            episodes = []
            for i in indices:
                episodes.append(copy.deepcopy(self.buffer.buffer_data[i]))
            episodes_states, values, new_values, _ = self.sample_data(copy.deepcopy(episodes))
            data_number = len(values)
            perm = np.random.permutation(data_number)

            for i in range(0, data_number, minibatch):
                total_states = []
                target_values = []
                for j in perm[i:i + minibatch]:
                    for k in range(len(episodes_states[j])):
                        total_states.append(copy.deepcopy(episodes_states[j][k]))
                        target_values.append(copy.deepcopy([new_values[j][k]]))
                total_states = np.array(total_states)
                target_values = np.array(target_values)
                self.sess.run(self.ctrain, {self.target_value: target_values, self.S: total_states})
                closs += self.sess.run(self.td_error, {self.target_value: target_values, self.S: total_states})
        print("critic loss is ", closs / VALUE_TRAIN_TIME)
        self.closses.append(closs / VALUE_TRAIN_TIME)
        return closs / VALUE_TRAIN_TIME

    # def actor_training(self, n_sample_size=SAMPLE_SIZE, traintime=ACTOR_TRAIN_TIME):
    #
    #     minibatch = self.minibatch
    #     n_episodes = self.buffer.get_length()
    #     if n_sample_size > n_episodes:
    #         n_sample_size = n_episodes
    #         print("======Warning: buffer size is less than sample size, need to store more data========")
    #     indices = np.random.choice(n_episodes, size=n_sample_size)  # 随机生成序号
    #     episodes = []
    #     for i in indices:
    #         episodes.append(copy.deepcopy(self.buffer.buffer_data[i]))
    #     episodes_states, values, _, episodes_actions = self.sample_data(copy.deepcopy(episodes))
    #     data_number = len(values)
    #     perm = np.random.permutation(data_number)
    #     # Using BGD
    #     minibatch = data_number
    #     for i in range(0, data_number, minibatch):
    #         target_actions = []
    #         total_states = []
    #         for j in perm[i:i + minibatch]:
    #             for k in range(len(episodes_states[j])):
    #                 total_states.append(episodes_states[j][k])
    #                 target_actions.append(episodes_actions[j][k])
    #         total_states = np.array(total_states)
    #         target_actions = np.array(target_actions)
    #     for value_train_times in range(traintime):
    #             self.sess.run(self.atrain, {self.target_action: target_actions, self.S: total_states})
    #             aloss = self.sess.run(self.a_loss, {self.target_action: target_actions, self.S: total_states})
    #             if (value_train_times + 1) % 5 == 0:
    #                 print("aloss is ", aloss)
    #             self.alosses.append(aloss)
    #     return aloss
    def actor_training(self, n_sample_size=SAMPLE_SIZE, traintime=ACTOR_TRAIN_TIME):
        # 和critic training差不多，只是损失函数不一样
        batchsize = BATCH_SIZE
        minibatch = self.minibatch
        n_episodes = self.buffer.get_length()
        if n_sample_size > n_episodes:
            n_sample_size = n_episodes
            print("======Warning: buffer size is less than sample size, need to store more data========")
        aloss = 0
        for value_train_times in range(traintime):
            indices = np.random.choice(n_episodes, size=n_sample_size)  # 随机生成序号
            episodes = []
            for i in indices:
                episodes.append(copy.deepcopy(self.buffer.buffer_data[i]))
            episodes_states, values, _, episodes_actions = self.sample_data(copy.deepcopy(episodes))
            data_number = len(values)
            perm = np.random.permutation(data_number)
            for i in range(0, data_number, minibatch):
                target_actions = []
                total_states = []
                for j in perm[i:i + minibatch]:
                    for k in range(len(episodes_states[j])):
                        total_states.append(episodes_states[j][k])
                        target_actions.append(episodes_actions[j][k])
                total_states = np.array(total_states)
                target_actions = np.array(target_actions)
                self.sess.run(self.atrain, {self.target_action: target_actions, self.S: total_states})
                aloss += self.sess.run(self.a_loss, {self.target_action: target_actions, self.S: total_states})
        aloss = aloss / TRAIN_TIME
        self.alosses.append(aloss)
        print('aloss is ', aloss)
        return aloss

    def sample_data(self, episodes):
        # 输入若k条episode[i,j,k], 分别返回[i*j, -1]格式的s,v(s),v_new(s), action.
        values = []
        updated_values = []
        episodes_states = []
        episodes_actions = []
        episodes_probs = []
        for episode in episodes:
            states, actions, rewards, _, probs = self.parse_episode(episode)
            states = np.array(states)
            actions = np.array(actions)
            probs = np.array(probs)
            # 把states从列表转化为numpy数组
            val_t = self.get_state_value(states)
            # new_val_t = self.compute_return(episode, val_t, rewards)
            vtrace_target = self.compute_vtrace_target(states, val_t, rewards, probs, actions)
            values.append(val_t)
            updated_values.append(vtrace_target)
            episodes_states.append(states)
            episodes_actions.append(actions)
            episodes_probs.append(probs)
        return episodes_states, values, updated_values, episodes_actions

    def pi2_with_critic(self, initial_start, env, iteration_times=5):
        # pi2探索，输入状态，生成混合动作，点优化5次，取最好。
        batch_max_value = np.zeros([iteration_times, 2])
        total_time = time.time()
        for i in range(iteration_times):
            initial_time = time.time()
            envs = [copy_env(env) for j in range(ROLL_OUTS + 1)]  # 复制环境，为计算V(S) = r+v(S_(t+1))做准备
            if i == 0:
                initial_action = self.sample_action(
                    initial_start.reshape([-1, s_dim]))  # initial action is supposed to be [action_dim,] narray object
            sigma = np.ones([ROLL_OUTS, self.a_dim])
            sigma[0] = np.zeros_like(sigma[0])
            action_groups = np.squeeze(
                np.clip(np.random.normal(initial_action, sigma), -self.a_bound[0], self.a_bound[0]))
            action_groups = action_groups.reshape(ROLL_OUTS, self.a_dim)
            next_stages = []
            rewards = []
            initial_time = time.time() - initial_time
            # print('initial took', initial_time, ' s')
            calculate_value_time = time.time()
            for j in range(len(action_groups)):
                temp_next_state, temp_reward, _, _ = envs[j].step(copy.deepcopy(action_groups[j][0]))
                next_stages.append(temp_next_state)
                rewards.append(temp_reward)
            state_groups = np.array(next_stages)
            rewards = np.array(rewards)
            next_values = np.array(self.get_state_value(state_groups))
            values = rewards.reshape([ROLL_OUTS, 1]) + next_values
            calculate_value_time = time.time() - calculate_value_time
            #  print('calculate value time took ', calculate_value_time)
            get_hybrid_action = time.time()
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
            hybrid_action = np.dot(action_groups.T, probability_weighting)
            hybrid_next_state, hybrid_reward, _, _ = envs[-1].step(hybrid_action)
            hybrid_value = hybrid_reward + self.get_state_value(np.array(hybrid_next_state).reshape([1, 11]))
            if hybrid_value >= maxv:
                current_action = hybrid_action
                current_value = hybrid_value
            else:
                current_action = action_groups[np.argmax(values)]
                current_value = maxv
            batch_max_value[i][0] = copy.deepcopy(current_action)
            batch_max_value[i][1] = copy.deepcopy(current_value)
            initial_action = hybrid_action
            get_hybrid_action = time.time() - get_hybrid_action
        #  print('get hybrid action took :', get_hybrid_action)
        index = np.argmax(batch_max_value[:, 1], axis=0)
        total_time = time.time() - total_time
        #   print('pi2_with_critic took ', total_time,' s')
        return batch_max_value[index][0]

    def pi2_tradition(self, initial_start, env, iteration_times=5):
        # pi2探索，输入状态，生成混合动作，点优化5次，去最好。
        #   total_time = time.time()
        batch_max_value = np.zeros([iteration_times, 2])
        for i in range(iteration_times):
            #   initial_time = time.time()
            envs = [copy_env(env) for j in range(ROLL_OUTS + 1)]  # 复制环境，为计算V(S) = r+v(S_(t+1))做准备
            if i == 0:
                initial_action = self.sample_action(
                    initial_start.reshape([-1, s_dim]))  # initial action is supposed to be [action_dim,] narray object
            sigma = np.ones([ROLL_OUTS, self.a_dim])
            sigma[0] = np.zeros_like(sigma[0])
            action_groups = np.squeeze(
                np.clip(np.random.normal(initial_action, sigma), -self.a_bound[0], self.a_bound[0]))
            action_groups = action_groups.reshape(ROLL_OUTS, self.a_dim)
            rewards = np.zeros([len(action_groups)])
            # initial_time = time.time() - initial_time
            #   print('initial took', initial_time, ' s')
            #   calculate_value_time = time.time()
            for j in range(len(action_groups)):

                r = 0
                a = copy.deepcopy(action_groups[j][0])
                temp_next_state, temp_reward, done, _ = envs[j].step(a)
                r += temp_reward
                timer = 0
                while not done or timer < 200:
                    a = pi2_critic.sample_action(temp_next_state.reshape([-1, s_dim]))
                    timer += 1
                    temp_next_state, temp_reward, done, _ = envs[j].step(a)
                    r += temp_reward
                rewards[j] = r
            print(len(rewards))
            # rewards = np.array(rewards)
            # print(len(rewards))
            values = rewards.reshape([ROLL_OUTS, 1])
            #    calculate_value_time = time.time() - calculate_value_time
            #  print('calculate value time took ', calculate_value_time)
            #    time_start = time.time()
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
            hybrid_action = np.dot(action_groups.T, probability_weighting)

            hybrid_value = 0
            a = hybrid_action
            temp_next_state, temp_reward, done, _ = envs[-1].step(a)
            hybrid_value += temp_reward
            while not done:
                a = pi2_critic.sample_action(temp_next_state.reshape([-1, s_dim]))
                temp_next_state, temp_reward, done, _ = envs[-1].step(a)
                hybrid_value += temp_reward
            if hybrid_value >= maxv:
                current_action = hybrid_action
                current_value = hybrid_value
            else:
                current_action = action_groups[np.argmax(values)]
                current_value = maxv
            batch_max_value[i][0] = copy.deepcopy(current_action)
            batch_max_value[i][1] = copy.deepcopy(current_value)
            initial_action = hybrid_action
        #    time_end = time.time()
        #    print('get hybrid action :', time_end - time_start)
        index = np.argmax(batch_max_value[:, 1], axis=0)
        # total_time = time.time() - total_time
        # print('pi2_tradition took ', total_time,' s')
        return batch_max_value[index][0]

if __name__ == '__main__':
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
    hybrid_rewards = []
    for i in range(epoch):
        if True:
            pi2_critic.learn()
            print('======================start testing=========================')
            r, t = pi2_critic.test(use_hybrid_method=False)
            print('==========test tradtition  reward ', r, ' steps', t, '=============')
            normal_rewards.append(r)
            r, t = pi2_critic.test(use_hybrid_method=True)
            print('==========test hybrid  reward ', r, ' steps', t, '=============')
            hybrid_rewards.append(r)
        if (i + 1) % 20 == 0:
            # 保存图片
            try:
                pi2_critic.save_model()
                pi2_critic.buffer.save_data(model_path)
                print('data saved successfully')
                plt.plot(normal_rewards)
                plt.savefig('./data/normal_rewards.jpg')
                plt.clf()
                plt.plot(hybrid_rewards)
                plt.savefig('./data/hybrid_rewards.jpg')
                plt.clf()
                plt.plot(pi2_critic.alosses)
                plt.savefig('./data/alosses.jpg')
                plt.clf()
                plt.plot(pi2_critic.closses)
                plt.savefig('./data/closses.jpg')
                print('figure saved successfully')
                plt.clf()
                with open('./data/PI2', 'wb') as f:
                    pickle.dump(normal_rewards, f, pickle.HIGHEST_PROTOCOL)
                with open('./data/GPS', 'wb') as f:
                    pickle.dump(hybrid_rewards, f, pickle.HIGHEST_PROTOCOL)

            # model_path = './Standard_buffer_data/reward_data'
            except:
                print('figure save failed')
