import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from plot_data import mkdir

# ENV_NAME ="InvertedPendulum"
ENV_NAME ="InvertedDoublePendulum"
alo_list = ["DDPG","MPC","CPI2_without_action","CPI2"]
def smooth(alo,weight=0.9):
    csv_path = f"data/csv/{ENV_NAME}/{alo}.csv"
    data = pd.read_csv(filepath_or_buffer=csv_path,header=0,names=['Step','Value'],dtype={'Step':np.int,'Value':np.float})
    scalar = data['Value'].values
    print(f"{alo} : {len(scalar)}")

    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    print(f"{alo} : {len(smoothed)}")
    mkdir(f"data/numpy/{ENV_NAME}/")
    np.save(f"data/numpy/{ENV_NAME}/{alo}",np.array(smoothed))
    plt.plot(smoothed,label=alo)
if __name__=='__main__':
    for alo in alo_list:
        smooth(alo)
    plt.legend(loc="best")
    plt.show()