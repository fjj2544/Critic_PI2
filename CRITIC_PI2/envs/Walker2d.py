import numpy as np
import math
import gym
dt = 0.008

def reward_function(s_, a):
    height, ang = s_[0, 0:2]
    alive_bonus = 1.0
    reward = 0
    reward += alive_bonus
    reward -= 1e-3 * np.square(a).sum()
    return reward
def is_done(s_, a):
    height,ang = s_[0, 0:2]
    done = not(height > 0.8 and height < 2.0 and
               ang > -1.0 and ang < 1.0)
    return done

if __name__ == '__main__':
    env = gym.make("Walker2d-v1")
    env = env.unwrapped
    env.reset()

    print(env.dt)
    print()



    print(env.model.data.qpos.shape)
    print(env.model.data.qvel.shape)

    print(env.model.data.qpos[0,0])

    height, ang = env.model.data.qpos[1:3,0]

    print(env.model.data.qpos[0:3,0])


    print(height,ang)
    print(height * np.sin(ang))

