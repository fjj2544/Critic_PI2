import numpy as np
import os
data_name = ["ddpg_data","mpc_data","reward_data","tra_pi2_data"]
env_name = "InvertedPendulum-v1"
def smooth(data, weight=0.96):
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1-weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)
for alo in data_name:
    data = np.load(f"/Users/bytedance/Desktop/大四项目/代码/rl/ICRA/Critic_PI2/experiment//data/experiment1/{env_name}/{alo}",allow_pickle=True)
    data = smooth(data)
    import matplotlib.pyplot as plt
    plt.plot(data)
    plt.show()
    np.save(f"/Users/bytedance/Desktop/大四项目/代码/rl/ICRA/Critic_PI2/experiment//data/experiment1/{env_name}/{alo}_smoothed",data)
