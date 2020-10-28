import matplotlib.pyplot as plt
import numpy as np
ENV_NAME = "InvertedDoublePendulum-v1"
ddpg_data = np.array(np.load(f"/Users/bytedance/Desktop/大四项目/代码/rl/ICRA/Critic_PI2/experiment/data/experiment1/{ENV_NAME}/ddpg_data_smoothed.npy"))
mpc_data = np.array(np.load(f"/Users/bytedance/Desktop/大四项目/代码/rl/ICRA/Critic_PI2/experiment/data/experiment1/{ENV_NAME}/mpc_data_smoothed.npy"))
cpi2_data = np.array(np.load(f"/Users/bytedance/Desktop/大四项目/代码/rl/ICRA/Critic_PI2/experiment/data/experiment1/{ENV_NAME}/reward_data_smoothed.npy"))
pi2_data = np.array(np.load(f"/Users/bytedance/Desktop/大四项目/代码/rl/ICRA/Critic_PI2/experiment/data/experiment1/{ENV_NAME}/tra_pi2_data_smoothed.npy"))
print(f"ddpg: {ddpg_data.shape}\n mpc: {mpc_data.shape} \n cpi2 {cpi2_data.shape}\n pi2 {pi2_data.shape}")

CONVERGE_VALUE = 900
data_number = np.inf
data_list = []
real_data = []
data_list.append(cpi2_data)
data_list.append(mpc_data)
data_list.append(ddpg_data)
data_list.append(pi2_data)
for data in data_list:
    data_number = min(data_number,len(data))
data_number = int(data_number)
for data in data_list:
    real_data.append(data[:data_number])
data_list = real_data

label = ["Critic PI2","MPC","DDPG","PI2"]
color = ["r","g","b","k"]
# line_style = ["-","-.",":","--"]
line_style = ["-","-","-","-"]
linewidth = 1 # 绘图中曲线宽度
fontsize = 5 # 绘图字体大小
markersize = 2.5  # 标志中字体大小
legend_font_size = 5 #图例中字体大小
labelwidth = 10 # 横纵轴

def mkdir(path):
    import os
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False
def save_figure(dir,name):
    mkdir(dir)
    plt.savefig(dir+name,bbox_inches = 'tight')

def plot_result(data_list # [algorithm_id,data,*]
                ,figure_number=3):
    # plt.figure(figsize=(2.8, 1.7), dpi=300)
    '''绘制alpha曲线'''
    fig, ax = plt.subplots(figsize=(2.8, 1.7), dpi=300)
    # 取消边框
    for key, spine in ax.spines.items():
        # 'left', 'right', 'bottom', 'top'
        if key == 'right' or key == 'top':
            spine.set_visible(False)
    plt.xticks(fontproperties='Times New Roman', fontsize=fontsize)
    plt.yticks(fontproperties='Times New Roman', fontsize=fontsize)
    plt.xlabel("Episodes",fontproperties='Times New Roman',fontsize=labelwidth)
    plt.ylabel("Mean Rewards",fontproperties='Times New Roman',fontsize=labelwidth)
    t = np.arange(0, data_number, 1)
    for i in range(figure_number):
        plt.plot(t,data_list[i], label=label[i],color=color[i],linestyle=line_style[i],linewidth=linewidth)
    converge_line = np.array(np.ones_like(data_list[0]))*CONVERGE_VALUE
    print(converge_line.shape)
    plt.plot(t,converge_line,label="converge",color="y",linestyle="--",linewidth=linewidth)
    # plt.legend(loc='best', prop={'family': 'Times New Roman', 'size': legend_font_size})
    plt.title(f"{ENV_NAME}",fontproperties='Times New Roman',fontsize=labelwidth)
    save_figure(f"/Users/bytedance/Desktop/大四项目/代码/rl/ICRA/Critic_PI2/experiment/photo/{ENV_NAME}/", f"{ENV_NAME}.pdf")
    plt.show()
plot_result(data_list,figure_number=4)

#--------------------------plot data with tensorboard--------------------------
# import tensorflow as tf
# from datetime import datetime
# ENV_NAME = "InvertedDoublePendulum-v1" # todo
#
# TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
# summary_writers = []
# for i in range(len(data_list)):
#     summary_writer = tf.summary.FileWriter(f"./log/{ENV_NAME}/{TIMESTAMP}/{label[i]}")
#     summary_writers.append(summary_writer)
# data_tensor = tf.placeholder(tf.float32)
# data_summary = tf.summary.scalar("Reward", data_tensor)
# sess =tf.Session()
# sess.run(tf.global_variables_initializer())
#
# for i in range(250):
#     for j,summary_writer in enumerate(summary_writers):
#         summary = sess.run(data_summary, feed_dict={data_tensor: data_list[j][i] })
#         summary_writer.add_summary(summary,global_step=i)