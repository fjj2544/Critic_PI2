import matplotlib.pyplot as plt
import numpy as np
ddpg_data = np.array(np.load("./InvertedDoublePendulum/ddpg_data.npy"))
mpc_data = np.array(np.load("./InvertedDoublePendulum/mpc_data.npy"))
cpi2_data = np.array(np.load("./InvertedDoublePendulum/reward_data.npy"))
pi2_data = np.array(np.load("./InvertedDoublePendulum/tra_pi2_data.npy"))
print(f"ddpg: {ddpg_data.shape}\n mpc: {mpc_data.shape} \n cpi2 {cpi2_data.shape}\n pi2 {pi2_data.shape}")
data_list = []
real_data = []
data_list.append(cpi2_data[:250])
data_list.append(mpc_data[:250])
data_list.append(ddpg_data[:250])
data_list.append(pi2_data[:250])


label = ["Critic PI2","MPC","DDPG","PI2"]
color = ["r","g","b","k"]
# line_style = ["-","-.",":","--"]
line_style = ["-","-","-","-"]
linewidth = 1 # 绘图中曲线宽度
fontsize = 5 # 绘图字体大小
markersize = 2.5  # 标志中字体大小
legend_font_size = 5 #图例中字体大小
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
    '''绘制alpha曲线'''
    plt.figure(figsize=(2.8, 1.7), dpi=300)
    plt.xticks(fontproperties='Times New Roman', fontsize=fontsize)
    plt.yticks(fontproperties='Times New Roman', fontsize=fontsize)
    plt.xlabel("Episodes",fontproperties='Times New Roman',fontsize=fontsize)
    plt.ylabel("Mean Rewards",fontproperties='Times New Roman',fontsize=fontsize)
    for i in range(figure_number):
        t = np.arange(0,len(data_list[i])*2,2)
        plt.plot(t,data_list[i], label=label[i],color=color[i],linestyle=line_style[i],linewidth=linewidth)
    plt.legend(loc='best', prop={'family': 'Times New Roman', 'size': legend_font_size})
    save_figure("./photo/exp1/", "InvertedDoublePendulum.pdf")
    plt.show()
plot_result(data_list,figure_number=4)

# #--------------------------plot data with tensorboard--------------------------
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