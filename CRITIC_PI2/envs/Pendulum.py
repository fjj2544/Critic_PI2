import numpy as np
import math
def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
def reward_function(s_, a):
    max_torque = 2
    th  = []
    for cos,sin in zip(s_[:, 0], s_[:, 1]):
        th.append(math.atan2(sin,cos))
    thdot = s_[:, 2]
    a = np.clip(a,-max_torque,max_torque)[0]
    th = np.array(th)
    thdot = np.array(thdot)

    costs = angle_normalize(th)**2 + .1*thdot**2 +.001*(a**2)
    return -costs
def is_done(s,a):
    return [False]