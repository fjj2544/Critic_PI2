import numpy as np
import copy
import pickle
BUFFERSIZE = int(1e3)
class Replay_buffer():
    def __init__(self, buffer_size= BUFFERSIZE):
        self.path = '.\Standard_buffer_data'
        self.buffersize= buffer_size
        self.buffer_data = []
        self.pointer = 0
        self.rewards = np.zeros([self.buffersize])
        """
        buffer的数据结构：
            [i,j,k]
            i是episode数量
            j是有多少个（s,a,r,s_, prob）序列
            k是（s,a,r,s_）内容
        Rewards对应的是每一条episode对应的回报
        
        """
    def get_length(self):
        print("pointer is:", self.pointer % self.buffersize)
        return len(self.buffer_data)
    def store_episode(self, episode, episode_reward):
        if self.pointer >= self.buffersize:
            self.buffer_data[self.pointer % self.buffersize] = episode
            self.rewards[self.pointer % self.buffersize] = episode_reward
        else:
            self.buffer_data.append(episode)
            self.rewards[self.pointer % self.buffersize] = episode_reward
        self.pointer += 1
        return self.pointer
    def save_data(self, model_path):
        with open(model_path, 'wb') as f:
            pickle.dump(self.buffer_data, f, pickle.HIGHEST_PROTOCOL)
            print(f"save buffer_date in {model_path}")
        with open(model_path, 'wb') as f:
            pickle.dump(self.rewards, f, pickle.HIGHEST_PROTOCOL)
            print(f"save rewards in {model_path}")
    def load_data(self):
        model_path = './Standard_buffer_data/buffer_data'
        with open(model_path, 'rb') as f:
           self.buffer_data = pickle.load(f)
           print(f"load buffer_date in {model_path}")
        model_path = './Standard_buffer_data/reward_data'
        with open(model_path, 'rb') as f:
           self.rewards = pickle.load(f)
           print(f"load reward_data in {model_path}")
        self.pointer = len(self.buffer_data)
