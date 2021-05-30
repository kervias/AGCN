# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pylab as plb # 导入绘图包
import matplotlib.pyplot as plt
from matplotlib import rc
rc('mathtext', default='regular')

# plt.style.use("seaborn-colorblind")

config = {
    "font.family": 'Times New Roman',
    "font.size": 20,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
    'axes.unicode_minus': False
}
matplotlib.rcParams.update(config)
fontcn = {'family': 'SimSun','size': 20} # 1pt = 4/3px
fonten = {'family':'Times New Roman','size': 25}
matplotlib.rcParams.update(config)
xlabels = ['0/Lr=0',0.0001,0.001,0.01,0.1,1]

data_ir = {
    'Amazon': [0.0500,0.0511,0.0507,0.0491,0.0508,0.0361],
    'Movie1M': [0.1685,0.1679,0.1703,0.1679,0.1686,0.1584]
}
data_ai = {
    'Amazon-Platform[MAP]': [0.7106, 0.7722, 0.7752,0.7654,0.7684,0.7577],
    'Amazon-Price[ACC]': [0.1728,0.1980,0.1982,0.1957,0.1840,0.1842],
    'Amazon-Theme[MAP]': [0.6958,0.6966,0.7275,0.7182,0.7168,0.7145],
    'Movie1M-Gender[ACC]': [0.7407,0.7644,0.7701,0.7631,0.3781,0.1549],
    'Movie1M-Age[ACC]': [0.3395,0.3660,0.3892,0.3781,0.3708,0.3623],
    'Movie1M-Occupation[ACC]': [0.1554,0.1236,0.1470,0.1549,0.1569,0.1598]
}

def plot_fig(d1, d2, xlabel, type_, dt, title):
    assert len(d1) == len(d2) == len(xlabel)
    fig, ax1 = plt.subplots()  # 使用subplots()创建窗口
    # print(help(fig))
    fig.set_figwidth(10)
    fig.set_figheight(8)
    ax1.plot(xlabel, d1, 'o-', c='orangered', mfc='w', label='NDCG@10', lw=4,markersize=15)  # 绘制折线图像1,圆形点，标签，线宽
    plt.legend(loc='center',bbox_to_anchor=[0.5,0.1])
    ax1.grid()
    ax1.set_title(title)


    ax2 = ax1.twinx()  # 创建第二个坐标轴
    ax2.plot(xlabel, d2, '^-', c='green',  mfc='w', label=type_, lw=4,markersize=15)  # 同上
    plt.legend(loc='center',bbox_to_anchor=[0.5,0.2])
    ax1.set_xlim([-0.1,5.1])
    ax1.set_xlabel('γ', fontdict=fonten)
    ax1.set_ylabel('NDCG@10', fontdict=fonten)
    ax2.set_ylabel(type_, fontdict=fonten)
    #plt.gcf().autofmt_xdate()  # 自动适应刻度线密度，包括x轴，y轴
    # if 'Amazon' in dt:
    #     ax2.set_ylim([0.035, 0.055])
    # else:
    #     ax2.set_ylim([0.150, 0.190])
    # Swdown = d1
    # Rn = d2
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(xlabel, Swdown, label = 'NDCG@10', marker='o', mfc='w')
    # ax.plot(xlabel, Rn,  label = 'ACC', marker='^', mfc='w')
    # ax2 = ax.twinx()
    # ax.legend(loc=0)
    # ax.grid()
    # ax.set_xlabel("γ")
    # ax.set_ylabel(r"NDCG@10")
    # ax.set_ylim([min(d1)-0.02, max(d1)+0.02])
    # ax2.set_ylabel(r"ACC")
    # ax2.set_ylim([min(d2) - 0.02, max(d2) + 0.02])
    # # ax2.set_ylim(0, 35)
    # # ax.set_ylim(-20,100)
    # ax2.legend(loc=0)


for ir_key, ir_val in data_ir.items():
    for ai_key, ai_val in data_ai.items():
        if ir_key in ai_key:
            d1 = ir_val
            d2 = ai_val
            type_ = ai_key[-4:-1]

            plot_fig(d1,d2,xlabel=xlabels,type_=type_,dt=ir_key,title=None)
            #plt.show()
            plt.savefig("{}.svg".format(ai_key), format='svg')
