
import numpy as np
import copy
from PI2_RL.Code.variables import GAMA,EXPLORATION_TOTAL_STEPS,roll_outs
from PI2_RL.Code.tools.env_copy import copy_env

def standardization(data):
    mu = np.mean(data, axis=0)                                      #  均值
    sigma = np.std(data, axis=0)                                    #  方差
    return (data - mu) / sigma

class DynamicBuffer:
    def __init__(self, ob_dim, act_dim, is_norm=False, buffer_size=15000):
        # -------------- Config --------------
        self.ob_dim = ob_dim
        self.act_dim = act_dim
        self.is_norm = is_norm
        self.buffer_size = buffer_size                              # The size of buffer
        # -------------- Data --------------
        self.obs_act = []                                           # The obs and act
        self.delta_obs = []                                         # The delta of sequential obs
        # -------------- Mean --------------
        self.mu_delta = np.zeros(ob_dim)                            # 均值
        self.mu_obs_act = np.zeros(ob_dim+act_dim)
        # -------------- Stdev --------------
        self.sigma_delta = np.ones(ob_dim)                          # 方差
        self.sigma_obs_act = np.ones(ob_dim+act_dim)

    def get_standardization(self):
        # ----------- Mean -----------
        self.mu_obs_act = np.mean(self.obs_act, axis=0)
        self.mu_delta = np.mean(self.delta_obs, axis=0)
        # ----------- Stdev -----------
        self.sigma_obs_act = np.std(self.obs_act, axis=0)
        self.sigma_delta = np.std(self.delta_obs, axis=0)
        for i in range(self.ob_dim + self.act_dim):
            if self.sigma_obs_act[i] == 0:
                self.sigma_obs_act[i] = 1.0
        for i in range(self.ob_dim):
            if self.sigma_delta[i] == 0:
                self.sigma_delta[i] = 1.0

    # <<<<<<<<<<增加奖励<<<<<<<<<<<<<<
    def addExperience(self, obs_now_batch, act_batch, obs_next_batch):
        if len(obs_now_batch)+len(self.obs_act) > self.buffer_size:
            self.obs_act[1:len(obs_now_batch)] = []
            self.delta_obs[1:len(obs_now_batch)] = []

        for obs, act, obs_ in zip(obs_now_batch, act_batch, obs_next_batch):
            o_a_ = np.hstack((obs, act))
            self.obs_act.append(o_a_)
            self.delta_obs.append(obs_ - obs)
        # ----------- normalization -----------
        if self.is_norm:
            self.get_standardization()

    def getExperience(self, number):
        if number == 0:
            return [], [], []
        obs_act = []
        delta_obs = []
        index = np.random.choice(range(len(self.obs_act)), number)                                                      # 直接选择经验不需要优先回放
        for i in index:
            obs_act_data = self.obs_act[i]
            delta_obs_data = self.delta_obs[i]
            if self.is_norm:
                obs_act_data = (self.obs_act[i] - self.mu_obs_act) / self.sigma_obs_act
                delta_obs_data = (self.delta_obs[i] - self.mu_delta) / self.sigma_delta
            delta_obs.append(delta_obs_data)
            obs_act.append(obs_act_data)
        obs_act = np.reshape(obs_act, (number, len(obs_act[0])))
        delta_obs = np.reshape(delta_obs, (number, len(delta_obs[0])))
        return obs_act, delta_obs

    def buffer_empty(self):
        return len(self.obs_act) == 0

    def clear_buffer(self):
        self.obs_act = []
        self.delta_obs = []
        return

    def get_len(self):
        return len(self.delta_obs)

class PolicyBuffer:
    def __init__(self, buffer_size=1500):
        self.buffer_size = buffer_size                              # buffer 大小
        self.obs = []                                               # 存储状态
        self.act = []                                               # 存储动作
        self.rwd = []                                               # 存储奖励

    # <<<<<<<<<<增加奖励<<<<<<<<<<<<<<
    def addExperience(self,obs_batch, act_batch,red_batch):
        if len(obs_batch)+len(self.obs) > self.buffer_size:
            self.rwd[1:len(obs_batch)] = []                         #清空奖励池
            self.obs[1:len(obs_batch)] = []                         #清空状态池
            self.act[1:len(obs_batch)] = []                         #清空动作池
        for obs, act, red in zip(obs_batch, act_batch, red_batch):
            self.rwd.append(red)                                    # 存储奖励
            self.obs.append(np.squeeze(obs))                        # 存储状态
            self.act.append(act)                                    # 存储动作

    # <<<<<<<<<<<<采经验<<<<<<<<<<
    def getExperience(self, number, if_priority=False):
        if number == 0:
            return [], [], []
        obs = []
        act = []
        reward = []
        if if_priority:                                                                                                 # 优先采样
            p = np.squeeze(self.get_pro())                                                                              # 计算动作关于奖励的概率
            index = np.random.choice(range(len(self.act)), number, p=p)                                                 # 计算Idx
        else:
            index = np.random.choice(range(len(self.act)), number)                                                      # 直接选择经验不需要优先回放
        for i in index:
            obs.append(self.obs[i])                                                                                     # 存储状态
            act.append(self.act[i])                                                                                     # 储存动作
            reward.append(self.rwd[i])                                                                                  # 储存奖励
        obs = np.reshape(obs, (number, len(obs[0])))                                                                    # 重构尺寸
        act = np.reshape(act, (number, len(act[0])))                                                                    # 重构尺寸
        reward = np.reshape(reward, (number, len(reward[0])))                                                           # 重构尺寸
        return obs, act, reward

    def get_pro(self):
        loss = np.reshape(self.rwd,newshape=[len(self.rwd)])
        exponential_value_loss = np.zeros([len(self.rwd)],dtype=np.float64)
        probability_weighting = np.zeros([len(self.rwd)],dtype=np.float64)
        if (loss.max() - loss.min() <= 1e-4):
            probability_weighting[:] = 1.0 / len(self.rwd)
        else:
            exponential_value_loss[:] = np.exp(-30 * (loss.max()-loss[:])/(loss.max()-loss.min()))
            probability_weighting[:] = exponential_value_loss[:] / np.sum(exponential_value_loss)
        return probability_weighting

    def buffer_empty(self):
        return len(self.obs) == 0                                                                                       # 查看是否经验池为空

    def clear_buffer(self):
        self.obs = []                                                                                                   # 清空经验池
        self.act = []                                                                                                   # 清空经验池
        self.rwd = []                                                                                                   # 清空经验池
        return

    def get_len(self):
        return len(self.rwd)                                                                                            # 得到经验池的尺寸

class Sample():
    def __init__(self,ob_dim,ac_dim):
        self.ob_dim = ob_dim                                                                                            # 状态空间维度
        self.ac_dim = ac_dim                                                                                            # 动作空间维度

    def get_episode_reward_with_sample(self
                                       ,policy ,current_env ,current_obs,current_action,
                                       action_bound,
                                       sigma_list = np.ones([roll_outs,1,1]),
                                       total_num = roll_outs,
                                       total_step = EXPLORATION_TOTAL_STEPS,with_noise = True):
        self.env = copy_env(current_env)                                                                                # 复制当前环境
        sigma = np.reshape(copy.deepcopy(sigma_list), (total_num, self.ac_dim))                                         # reshape 尺寸
        obs_out = [current_obs]                                                                                         # 状态空间
        reward_out = []
        reward_list = []
        env_list = []
        obs_list = []
        obs_n_list = []
        done_list = []
        #<<<<<<<<<复制多个ENV<<<<<<<
        for i in range(total_num):
            env_list.append(copy_env(self.env))                                                                         #环境列表
            reward_list.append([])                                                                                      #记录reward
            obs_list.append(copy.deepcopy(current_obs))                                                                 #记录当前的状态
        current_action = np.squeeze(current_action)
        action = np.clip(np.random.normal(current_action,sigma),action_bound[0],action_bound[1])                        #第一个动作人为给定
        if(with_noise==False):
            sigma *=0
        for i in range(total_num):
            observation_next, reward_, done, info = env_list[i].step(action[i])                                         # 交互一次
            obs_n_list.append(observation_next)                                                                         # 记录状态
            done_list.append(done)                                                                                      # 记录done
            reward_list[i].append(reward_)                                                                              # 记录reward
        action_out = np.reshape(action, [total_num, self.ac_dim])                                                       # 记录第一个动作

        finish_traj = sum(done_list)                                                                                    # 看是否炸掉
        for j in range(total_step-1):
            obs_list = np.reshape(obs_n_list, (total_num, self.ob_dim))                                                 # reshape 状态
            action = policy.policy_sample( obs_list,sigma=sigma)                                                        # 得到采样网络的采样
            action = action[0]

            for i in range(total_num):
                if finish_traj == total_num:
                    break
                if done_list[i]:
                    continue
                observation_next, reward_, done, info = env_list[i].step(action[i])                                     # 与环境交互
                obs_n_list[i] = observation_next                                                                        # 记录状态
                if done:
                    finish_traj += 1
                    done_list[i] = True
                    reward_list[i].append(reward_)
                else:
                    reward_list[i].append(reward_)
        # <<<<<<<<<计算折扣累计汇报<<<<<<<<<<<<<<<<<
        discount_r = np.zeros((total_num,total_step),np.float64)
        for id_ in range(total_num):
            running_add = 0
            len_traj = len(reward_list[id_])
            for t in reversed(list(range(0,len_traj))):
                running_add = running_add * GAMA + reward_list[id_][t]
                discount_r[id_][t] = running_add
            reward_out.append(discount_r[id_][0])
        reward_out = np.reshape(reward_out,newshape=[total_num,1])                                                      # 得到折扣累计回报
        return obs_out, action_out, reward_out

    def get_episode_reward_with_dynamic(self,
                                        policy, current_env,
                                        current_obs, current_action,
                                        dynamic_net,
                                        action_bound,
                                        sigma_list=np.ones([roll_outs, 1, 1]),
                                        total_num=roll_outs,
                                        total_step=EXPLORATION_TOTAL_STEPS,
                                        with_noise=True):
        self.env = copy_env(current_env)                                                                                # 复制当前环境
        sigma = np.reshape(copy.deepcopy(sigma_list), (total_num, self.ac_dim))                                         # reshape 尺寸
        obs_out = [current_obs]                                                                                         # 状态空间
        reward_out = []
        reward_list = []
        env_list = []
        obs_list = []
        #<<<<<<<<<复制多个ENV<<<<<<<
        for i in range(total_num):
            env_list.append(copy_env(self.env))                                                                         #环境列表
            reward_list.append([])                                                                                      #记录reward
            obs_list.append(copy.deepcopy(np.squeeze(current_obs)))                                                     #记录当前的状态
        obs_list = np.array(obs_list)
        current_action = np.squeeze(current_action)
        action = np.squeeze(np.clip(np.random.normal(current_action, sigma), action_bound[0], action_bound[1]))         #第一个动作人为给定
        if with_noise is False:
            sigma *= 0
        state_act = np.hstack((obs_list[0], action[0]))
        for i in range(1, roll_outs):
            data_1 = np.hstack((obs_list[i], action[i]))
            state_act = np.vstack((state_act, data_1))
        obs_np, reward_np, done_list, info_np = dynamic_net.prediction(state_act)
        for i in range(total_num):
            reward_list[i].append(reward_np[i])                                                                         # 记录reward
        action_out = np.reshape(action, [total_num, self.ac_dim])                                                       # 记录第一个动作
        done_traj = sum(done_list)
        for j in range(total_step-1):
            if done_traj == total_num:
                break

            obs_np = np.reshape(obs_np, (roll_outs, self.ob_dim))
            action = policy.policy_sample(obs_np, sigma=sigma)
            action = action[0]
            state_act = np.hstack((obs_np, action))
            obs_np, reward_np, done, info_np = dynamic_net.prediction(state_act)
            for i in range(total_num):
                if done[i] is True and done_list[i] is False:
                    done_traj += 1
                    done_list[i] = True
                    reward_list[i].append(0.0)
                if done[i] is False and done_list[i] is False:
                    reward_list[i].append(reward_np[i])
        discount_r = np.zeros((total_num, total_step), np.float64)

        for id_ in range(total_num):
            running_add = 0
            len_traj = len(reward_list[id_])
            for t in reversed(list(range(0,len_traj))):
                running_add = running_add * GAMA + reward_list[id_][t]
                discount_r[id_][t] = running_add
            reward_out.append(float(discount_r[id_][0]))
        reward_out = np.reshape(reward_out, newshape=[total_num,1])                                                     # 得到折扣累计回报
        return obs_out, action_out, reward_out, reward_list
