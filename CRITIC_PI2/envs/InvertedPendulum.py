import numpy as np
import math
def reward_function(s_, a):
    reward = np.ones((len(s_), 1))
    return reward
def is_done(s_, a):
    done = []
    for predict in s_:
        done.append(not (np.isfinite(predict).all() and (np.abs(predict[1]) <= .2)))
    return done