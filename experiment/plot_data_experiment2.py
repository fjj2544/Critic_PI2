import matplotlib.pyplot as plt
import numpy as np
# ENV_NAME = "InvertedPendulum"

ENV_NAME= "InvertedDoublePendulum"
cpi2 = np.array(np.load(f"./data/numpy/smoothed/ablation_data/CPI2.npy"))
without_optimal = np.array(np.load(f"./data/numpy/smoothed/ablation_data/optimal_baseline_data.npy"))
without_v_trace = np.array(np.load(f"./data/numpy/smoothed/ablation_data/vtrace_data.npy"))
without_action = np.array(np.load(f"./data/numpy/smoothed/ablation_data/CPI2_without_action.npy"))
print(f"ddpg: {cpi2.shape}\n mpc: {without_optimal.shape} \n cpi2 {without_v_trace.shape} \n without action:{without_action.shape}" )

CONVERGE_VALUE = 4700
data_number = np.inf
data_list = []
real_data = []
data_list.append(cpi2)
data_list.append(without_optimal)
data_list.append(without_v_trace)
data_list.append(without_action)
for data in data_list:
    data_number = min(data_number,len(data))
data_number = int(data_number)
for data in data_list:
    real_data.append(data[:data_number])
data_list = real_data

label = ["Critic PI2","Without Optimal Action","Without Critic","Without Training Actor"]
color = ["r","g","b","k"]
# line_style = ["-","-.",":","--"]
line_style = ["-","-","-","-"]
linewidth = 1 # 绘图中曲线宽度
fontsize = 10 # 绘图字体大小
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
    fig, ax = plt.subplots(figsize=(5,1.7), dpi=1000)

    plt.gcf().set_facecolor(np.ones(3) * 240 / 255)  # 生成画布的大小
    plt.grid()
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
    # plt.title(f"{ENV_NAME}",fontproperties='Times New Roman',fontsize=labelwidth)
    save_figure(f"./photo/exp2/{ENV_NAME}/", f"{ENV_NAME}.pdf")
    plt.show()
plot_result(data_list,figure_number=4)

#