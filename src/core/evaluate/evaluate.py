import numpy as np
import torch
from collections import defaultdict
import math

class Evaluate(object):
    is_cuda = True
    def __init__(self):
        pass

    @staticmethod
    def get_idcg(length):
        idcg = 0.0
        for i in range(length):
            idcg += math.log(2) / math.log(i + 2)
        return idcg

    @staticmethod
    def get_hr_ndcg(uid_list: list, pos_predict: dict, neg_predict: dict, top_k: int):
        """
            compute the hr and ndcg
        :param uid_list: user id list , dtype: list<uid>
        :param pos_predict: predict_pos_rating for each user, dtype: dict{uid:list<rating>}
        :param neg_predict: predict_neg_rating for each user, dtype: dict{uid:list<rating>}
        :param top_k: topN value
        :return: hr@top_k and ndcg@top_k
        """
        hr_list = []
        ndcg_list = []
        for uid in uid_list:
            pos_rating_list = pos_predict[uid]
            neg_rating_list = neg_predict[uid]
            target_len = min(top_k, len(pos_rating_list))
            if target_len > 0:
                rating_all = np.asarray(pos_rating_list + neg_rating_list)
                sort_des_index = np.argsort(rating_all)[::-1]  # 从大到小排序后索引
                hit_value = dcg_value = 0.0
                for idx in range(top_k):
                    rank = sort_des_index[idx]
                    if rank < len(pos_rating_list):
                        hit_value += 1
                        dcg_value += math.log(2) / math.log(idx + 2)
                t_hr = round(hit_value / target_len, 4)
                t_idcg = Evaluate.get_idcg(target_len)
                t_ndcg = round(dcg_value / t_idcg, 4)
                hr_list.append(t_hr)
                ndcg_list.append(t_ndcg)
            else:
                pass
                #print("存在target为0")
        mean_hr = round(np.mean(hr_list), 4)
        mean_ndcg = round(np.mean(ndcg_list), 4)
        return mean_hr, mean_ndcg

    @staticmethod
    @torch.no_grad()
    def get_hr_ndcg_from_emb(user_emb: torch.Tensor, item_emb: torch.Tensor, dict_pos_data: dict,
                             dict_neg_data: dict, topk_list: list):
        """
            根据正样本索引和负样本索引计算rating，排序后计算hr和ndcg
        :param user_emb: user_embedding
        :param item_emb: item_embedding
        :param dict_pos_data: dict<uid:list<item_id>>
        :param dict_neg_data: dict<uid:list<item_id>>
        :param topk_list: list<int>
        :return: hr[dict<topk:float>] ndcg[dict<topk:float>]
        """
        dict_hr_list, dict_ndcg_list = defaultdict(list), defaultdict(list)
        dict_hr, dict_ndcg = dict(), dict()
        batch_size = 10000
        user_count = user_emb.shape[0]
        batch_num = math.ceil(user_count / batch_size)
        for batch_ind in range(batch_num):
            start_ind = batch_ind * batch_size
            end_ind = min(start_ind + batch_size, user_count)
            ratings = torch.mm(user_emb[start_ind:end_ind], item_emb.t())
            batch_user_count = ratings.shape[0]
            u_list = list(range(batch_user_count))
            pos_predict,neg_predict = dict(), dict()
            if Evaluate.is_cuda:
                ratings = ratings.cpu()
            for i in range(batch_user_count):
                pos_predict[i] = ratings[i, dict_pos_data[start_ind + i]].detach().numpy().tolist()
                neg_predict[i] = ratings[i, dict_neg_data[start_ind + i]].detach().numpy().tolist()
            for topk in topk_list:
                tmp_hr, tmp_ndcg = Evaluate.get_hr_ndcg(u_list, pos_predict, neg_predict, topk)
                dict_hr_list[topk].append(tmp_hr * batch_user_count)
                dict_ndcg_list[topk].append(tmp_ndcg * batch_user_count)
        for topk in topk_list:
            dict_hr[topk] = round(sum(dict_hr_list[topk]) / user_count, 4)
            dict_ndcg[topk] = round(sum(dict_ndcg_list[topk]) / user_count, 4)
        return dict_hr, dict_ndcg

    @staticmethod
    def comput_ap(true_label: np.ndarray, predict_label: np.ndarray):
        """
            计算AP
        :param true_label: 真实标签
        :param predict_lable: 预测置信度
        :return: Average Precision
        """
        assert len(true_label.shape) == len(predict_label.shape) == 1
        assert len(true_label) > 0
        #sort_pd = np.sort(predict_label)  # [::-1]
        sort_ind = np.argsort(predict_label)  # [::-1]
        sort_gt = true_label[sort_ind]
        curr_max_precision = 0.0
        total_num = len(true_label)
        curr_pos_num = int(np.sum(sort_gt))
        if curr_pos_num == 0:
            return None
        assert curr_pos_num > 0, "ground truth标签全为0"
        max_precision_in_each_recall = np.zeros(curr_pos_num, dtype=np.float64)
        for i in range(total_num):
            # pd = sort_pd[i]
            gt = sort_gt[i]
            assert gt in [0, 1]
            curr_max_precision = max(curr_max_precision, curr_pos_num / (total_num - i))
            if gt == 1:
                max_precision_in_each_recall[curr_pos_num - 1] = curr_max_precision
                curr_pos_num -= 1
        return np.asarray(max_precision_in_each_recall).mean()

    @staticmethod
    def arg_topk_max(ary:np.ndarray, n:int):
        """Returns the n largest indices from a numpy array."""
        flat = ary.flatten()
        indices = np.argpartition(flat, -n)[-n:]
        indices = indices[np.argsort(-flat[indices])]
        return np.unravel_index(indices, ary.shape)


    @staticmethod
    def get_hr_ndcg_for_test(user_emb: torch.Tensor, item_emb: torch.Tensor, dict_pos_data: dict,
                             complete_data: dict, topk_list: list):
        """
            put all neg samples into consideration
        :param user_emb: user_embedding
        :param item_emb: item_embedding
        :param dict_pos_data: dict<uid:list<item_id>>
        :param complete_data: dict<uid:list<item_id>>
        :param topk_list: list<int>
        :return: hr[dict<topk:float>] ndcg[dict<topk:float>]
        """
        dict_hr_list, dict_ndcg_list = defaultdict(list), defaultdict(list)
        dict_hr, dict_ndcg = dict(), dict()
        ratings_all = torch.mm(user_emb, item_emb.t())
        itemset = set(range(item_emb.shape[0]))
        if Evaluate.is_cuda is True:
            ratings_all = ratings_all.cpu()
        ratings_all = ratings_all.detach().numpy()
        max_topk = max(topk_list)
        for uid in range(user_emb.shape[0]):
            pos_index = dict_pos_data[uid]
            neg_index = list(itemset - set(complete_data[uid]))
            # print(uid, max(pos_index),max(neg_index), ratings_all.shape)
            ratings = ratings_all[uid][pos_index + neg_index]
            topk_des_index = list(Evaluate.arg_topk_max(ratings, max_topk)[0])
            for topk in topk_list:
                hit_value, dcg_value = 0.0, 0.0
                for idx in range(topk):
                    rank = topk_des_index[idx]
                    if rank < pos_index.__len__():
                        hit_value += 1
                        dcg_value += np.log(2) / np.log(idx + 2)
                target_len = min(pos_index.__len__(), topk)
                dict_hr_list[topk].append(hit_value / target_len)
                dict_ndcg_list[topk].append(dcg_value / Evaluate.get_idcg(target_len))
        for topk in topk_list:
            dict_hr[topk] = round(sum(dict_hr_list[topk]) / user_emb.shape[0], 4)
            dict_ndcg[topk] = round(sum(dict_ndcg_list[topk]) / user_emb.shape[0], 4)
        return dict_hr, dict_ndcg
