import numpy as np
import math
def reward_function(s_, a):
    reward = np.ones((len(s_), 1))
    return reward
def is_done(s_, a):
    dones = []
    for predict in s_:
        done = not (np.isfinite(predict).all() and (np.abs(predict[1]) <= .2))
        dones.append(done)
    return dones
if __name__ == '__main__':
    s_ = np.array([[1,np.inf,3],[4,5,6]])
    print(is_done(s_,1))