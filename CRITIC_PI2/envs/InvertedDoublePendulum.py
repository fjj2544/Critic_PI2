import numpy as np
import math
def reward_function(s_, a):
    theta1 = np.arctan2(s_[:, 1], s_[:, 3])
    theta2 = np.arctan2(s_[:, 2], s_[:, 4])
    y = 0.6 * np.cos(theta1) + 0.6 * np.cos(theta1 + theta2)
    x = s_[:, 0] + 0.6 * np.sin(theta1) + 0.6 * np.sin(theta1 + theta2)

    v1 = s_[:, 6]
    v2 = s_[:, 7]
    dist_penalty = 0.01 * x ** 2 + (y-2) **2
    vel_penalty = 1e-3 * v1 **2 + 5e-3 * v2 **2
    alive_bonus = 10
    reward = (alive_bonus - dist_penalty - vel_penalty)
    return reward
def is_done(s_, a):
    theta1 = np.arctan2(s_[:, 1], s_[:, 3])
    theta2 = np.arctan2(s_[:, 2], s_[:, 4])
    y = 0.6 * np.cos(theta1) + 0.6 * np.cos(theta1 + theta2)
    x = s_[:, 0] + 0.6 * np.sin(theta1) + 0.6 * np.sin(theta1 + theta2)
    done = (y <= 1)
    return done