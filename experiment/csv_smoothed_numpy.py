import pandas as pd
import numpy as np
import os

ENV_NAME ="InvertedDoublePendulum"
def smooth(alo,weight=0.85): #weight是平滑度，tensorboard 默认0.6
    csv_path = f"data/csv/{ENV_NAME}/{alo}"
    data = pd.read_csv(filepath_or_buffer=csv_path,header=0,names=['Step','Value'],dtype={'Step':np.int,'Value':np.float})
    scalar = data['Value'].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    np.save(f"data/numpy/{ENV_NAME}/{alo}",np.array(smoothed))

if __name__=='__main__':
    smooth('test.csv')