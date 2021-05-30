import numpy as np
from collections import defaultdict

# load data
data_u2i = np.load("../../../data/movielens1M/train_U2I.npy", allow_pickle=True).tolist()
metric_filename = 'AGCN-4-0.1703.npy'

# split_group
def split_group_by_assign(U2I: dict, assign: list = [24, 48, 96, 192, np.inf]):
    assign.sort()
    assert assign[-1] == np.inf
    map_u_num = defaultdict(int)
    # map_num_u = defaultdict(int)
    for k, v in U2I.items():
        assert len(v) > 0
        map_u_num[k] = len(v)
        # map_num_u[len(v)] += 1
    group_num = len(assign)
    group = defaultdict(list)

    def group_id(num):
        for ind, bound in enumerate(assign):
            if num < bound:
                return ind

    for k, v in map_u_num.items():
        group[group_id(v)].append(k)
    return group


group = split_group_by_assign(data_u2i)
print({ind: len(item) for ind, item in group.items()})

# statistics
data_metric = np.load(metric_filename, allow_pickle=True).tolist()
print(data_metric[0])
map_metric = {item['uid']: item['perf'] for item in data_metric}
uid_list = []
for item in data_metric:
    uid_list.append(item['uid'])

def is_sorted(a):
    return all([a[i] <= a[i + 1] for i in range(len(a) - 1)])

print(is_sorted(uid_list))

group_result = {}
except_uid_list = []
for group_id, uid_list in group.items():
    all_metric = 0.0
    u_num = len(uid_list)
    for uid in uid_list:
        if uid not in map_metric:
            except_uid_list.append(uid)
            u_num -= 1
            continue
        u_metric = map_metric[uid]
        all_metric += u_metric[3]
    group_result[group_id] = round(all_metric / u_num, 5)

print(group_result)
print("except uid list: {}".format(except_uid_list))

