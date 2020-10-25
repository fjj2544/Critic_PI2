import numpy as np
import math
def reward_function(s, a):
    reward = np.ones((len(s), 1))
    return reward
def is_done(s,a):
    done = []
    for predict in s:
        done.append(not (np.isfinite(predict).all() and (np.abs(predict[1]) <= .2)))
    return done