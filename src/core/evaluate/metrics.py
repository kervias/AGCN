import math


def recall_k(ranked_list, ground_list):
    hits = 0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            hits += 1
    rec = hits / (1.0 * len(ground_list))
    return rec


def hr_k(ranked_list, ground_truth):
    hits = 0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id not in ground_truth:
            continue
        hits += 1
    rec = hits / (1.0 * min(len(ranked_list), len(ground_truth)))
    return rec


def ndcg_k(ranked_list, ground_truth):
    dcg = 0
    idcg = IDCG(min(len(ranked_list), len(ground_truth)))
    for i in range(len(ranked_list)):
        iid = ranked_list[i]
        if iid not in ground_truth:
            continue
        rank = i + 1
        dcg += 1 / math.log(rank + 1, 2)
    return dcg / idcg


def IDCG(n):
    idcg = 0
    for i in range(1, n + 1):
        idcg += 1 / math.log(i + 1, 2)
    return idcg
