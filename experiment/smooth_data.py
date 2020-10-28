
import numpy as np
data_name = ["ddpg_data","mpc_data","reward_data","tra_pi2_data"]
from plot_data import mkdir

save_dir = f"data/numpy/smoothed/ablation_data/"
save_name = f"optimal_baseline_data"
load_name = save_name
load_dir = f"data/numpy/raw_data/ablation_data/"
def smooth(data, weight=0.96):
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1-weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)
for alo in data_name:
    data = np.load(load_dir+load_name)
    data = smooth(data)
    import matplotlib.pyplot as plt
    plt.plot(data)
    plt.show()
    mkdir(save_dir)
    np.save(save_dir+save_name,data)
#
# def smooth(path,weight=0.96): #weight是平滑度，tensorboard 默认0.6
#     data = pd.read_csv(filepath_or_buffer=csv_path,header=0,names=['Step','Value'],dtype={'Step':np.int,'Value':np.float})
#     scalar = data['Value'].values
#     last = scalar[0]
#     smoothed = []
#     for point in scalar:
#         smoothed_val = last * weight + (1 - weight) * point
#         smoothed.append(smoothed_val)
#         last = smoothed_val
#     save = pd.DataFrame({'Step':data['Step'].values,'Value':smoothed})
#     save.to_csv('smooth_'+csv_path)
