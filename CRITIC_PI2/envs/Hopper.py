import numpy as np
import math
import gym

def is_done(s,a):
    done = []
    for predict,a_ in zip(s,a):
        done_ = (np.isfinite(predict).all() and (np.abs(predict[1:]) < 100).all() and (predict[0] > 0.7) and (abs(predict[1])<0.2))
        done.append(done_)
    return done
def reward_function(s,s_,a):
    reward = []
    for predict,a_ in zip(s_,a):
        posafter = 0.0
        posbefore = 0.0
        reward_ = (posafter - posbefore) / 0.008 + 1.0 - 1e-3 * np.square(a_).sum()
        reward.append(reward_)
    return reward


if __name__ == '__main__':
    env = gym.make("Hopper-v1")
    env.seed(1)  #
    np.random.seed(1)  # KL

    env = env.unwrapped  # 解封装环境
    # ------------查看状态空间范围--------------
    print(env.observation_space)  # 输出状态空间
    print(env.observation_space.shape[0])  # 输出状态空间维度
    print(env.observation_space.high, env.observation_space.low)  # 输出状态空间范围
    ob_dim = env.observation_space.shape[0]  # 记录状态空间
    # ----------查看动作空间范围---------------
    try:
        action_bound = np.vstack((env.action_space.low, env.action_space.high))
        ac_dim = env.action_space.shape[0]
        print(env.action_space)
        print(env.action_space.shape[0])
    except:
        action_bound = np.vstack((0, env.action_space.n - 1))
        ac_dim = 1
        print(env.action_space)
        print(env.action_space.n)

    env.reset()
    print(np.concatenate([[env.model.data.qpos.flat[0]],env._get_obs()]))
    # print(env._get_obs())

    print(env.dt)
    # s1 = env.model.data.qpos[0]
    # # print(env.model.data.qpos)
    # #
    # # print(env.model.data.qvel)
    #
    # a= [1,1,1]
    # print(env.step(a)[1]+ 1e-3*np.square(a).sum()-1)
    # s2 = env.model.data.qpos[0]
    # print((s2-s1)/0.08)
    # # print(env.model.data.qpos)
    # #
    # # print(env.model.data.qvel)