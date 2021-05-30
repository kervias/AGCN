import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from collections import OrderedDict
# plt.style.use("seaborn-colorblind")
# 设置matplotlib正常显示中文和负号
config = {
    "font.family": 'Times New Roman',
    "font.size": 12,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
    'axes.unicode_minus': False
}
matplotlib.rcParams.update(config)
fontcn = {'family': 'SimSun','size': 15} # 1pt = 4/3px
fonten = {'family':'Times New Roman','size': 15}


# settings
dataset_id = 2
data = OrderedDict()
dataset_cfg = ['AmazonVideoGames', 'Movielens1M', 'Movielens20M']
label = None
if dataset_id == 0:
    label = label or ['[0,4)', '[4,8)', '[8,16)', '[16,32)', '[32,)']
    data['BPRMF'] = [0.0337,0.0292,0.0344,0.0346,0.03866]
    data['FM'] = [0.0326,0.0337,0.0391,0.0342,0.04504]
    data['NGCF'] = [0.038,0.0401,0.0462,0.0520,0.05503]
    data['AGCN'] = [0.0494,0.0480,0.0565,0.0619,0.06462]
elif dataset_id == 1:
    label = label or ['[0,24)', '[24,48)', '[48,96)', '[96,192)', '[192,)']
    data['BPRMF'] = [0.14448, 0.16736, 0.16374, 0.21949, 0.26999]
    data['FM'] = [0.15327, 0.16577, 0.16058, 0.22122, 0.28338]
    data['NGCF'] = [0.15006, 0.16985, 0.17052, 0.22439, 0.32278]
    data['AGCN'] = [0.16333, 0.17265, 0.16979, 0.22442, 0.33817]
elif dataset_id == 2:
    label = label or ['[0,24)', '[24,48)', '[48,96)', '[96,192)', '[192,)']
    data['BPRMF'] = [0.1512,0.2022,0.2114,0.2888,0.52138]
    data['FM'] = [0.15763,0.20484,0.21405,0.29094,0.5153]#[0.158,0.2057,0.2146,0.2909,0.51636],
    data['NGCF'] = [0.163,0.2089,0.2162,0.2924,0.51831]
    data['AGCN'] = [0.183,0.2239,0.2232,0.3034,0.52831]

assert len(label) == len(list(data.values())[0])

bar_width = 0.2
bar_x = np.arange(len(label)) - bar_width * (len(data) / 2) + bar_width/2

fig = plt.figure(figsize=(6, 5))

ax = fig.add_subplot(111)
#ax.set_title(dataset_cfg[dataset_id])

bars_list = []
for key, value in data.items():
    print(key, value)
    bars_list.append(
        ax.bar(x=bar_x,  # 设置不同的x起始位置
               height=value, width=bar_width)
    )
    bar_x += bar_width

print(len(bars_list))

ax.set_xlabel('用户记录数分组区间', fontdict=fontcn)
ax.set_ylabel('NDCG@10', fontdict=fonten)
#ax.set_title('各组不同性别分数')
ax.set_xticks(range(len(label)))
ax.set_xticklabels(label)
if dataset_id == 2:
    ax.set_ylim([0.1,0.55])

#ax.set_yticklabels(np.arange(0, 81, 10))
ax.legend(bars_list, data.keys())
print(data.keys())

#plt.show()
plt.savefig("{}.svg".format(dataset_cfg[dataset_id]), format='svg')
