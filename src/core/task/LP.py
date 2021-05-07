import math
import random
import json
from utils import tensor2npy
from core.data.loadUtil import LoadUtil
from collections import defaultdict
import torch
import numpy as np
from core.models.LP import LP
from sklearn.metrics import average_precision_score, accuracy_score
from tqdm import tqdm


class LP_Manager(object):
    """
        Label Propagation
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_name = self.cfg.dataset_name
        self.tmpout_folder_path = self.cfg['tmpout_folder_path']
        self.logger = self.cfg.logger
        self.yml_cfg = self.cfg.yml_cfg
        self.model_cfg = self.yml_cfg['LP']

        self.item_attr_cfg = self.yml_cfg['item_attr']
        self.user_attr_cfg = self.yml_cfg['user_attr']
        self.user_count = self.yml_cfg['user_count']
        self.item_count = self.yml_cfg['item_count']
        self.load_util = LoadUtil(cfg=self.cfg)

    def start(self):
        # 1. 加载数据
        train_U2I = self.load_util.load_train_U2I()

        item_attrs_complete = None
        item_attrs_missing = None
        item_attrs_existing_index_list = None
        item_attrs_missing_index_list = None
        item_gt_list = None
        if self.item_attr_cfg['have'] is True:
            item_attrs_complete = self.load_util.load_item_attrs_complete()
            item_attrs_missing = self.load_util.load_item_attrs_missing()
            item_attrs_existing_index_list = self.load_util.load_item_attrs_existing_index()
            item_attrs_missing_index_list = self.load_util.load_item_attrs_missing_index()
            item_gt_list = self.load_util.load_item_gt_list()

        user_attrs_complete = None
        user_attrs_missing = None
        user_attrs_missing_index_list = None
        user_attrs_existing_index_list = None
        user_gt_list = None
        if self.user_attr_cfg['have'] is True:
            user_attrs_complete = self.load_util.load_user_attrs_complete()
            user_attrs_missing = self.load_util.load_user_attrs_missing()
            user_attrs_existing_index_list = self.load_util.load_user_attrs_existing_index()
            user_attrs_missing_index_list = self.load_util.load_user_attrs_missing_index()
            user_gt_list = self.load_util.load_user_gt_list()

        epoch_num = self.model_cfg['epoch_num']
        loss_threshold = self.model_cfg['loss_threshold']
        knn = self.model_cfg['knn']
        select_count = self.model_cfg['select_count']

        item_max_metric_dict, user_max_metric_dict = dict(), dict()
        if self.user_attr_cfg['have'] is True:
            self.logger.info("start handle user attribute...")
            trans_mat = self.get_transform_mat(self.get_similarity_mat(train_U2I, count=select_count),
                                               count=self.item_count,
                                               k=knn).cuda()
            attr_dim_list = self.user_attr_cfg['attr_dim_list']
            for i, dim_len in enumerate(attr_dim_list):
                attr_type = self.user_attr_cfg['attr_type_list'][i]
                slice_l = sum(attr_dim_list[0:i])
                slice_r = slice_l + attr_dim_list[i]
                model = LP()
                user_attr_input = torch.from_numpy(user_attrs_missing[:, slice_l:slice_r]).cuda()
                user_attr_gt = user_gt_list[i][:, slice_l:slice_r].cuda()
                self.logger.info("start train {}th user attribute".format(i))
                best_metric = -np.inf
                for epoch in range(epoch_num):
                    attr_pd = model.forward(user_attr_input, trans_mat)
                    loss = model.get_loss(attr_pd[user_attrs_existing_index_list[i]], user_attr_gt)
                    self.logger.info("[EPOCH={:03d}]: loss={}".format(epoch, loss.item()))
                    metric = self.evaluate_metric(
                        attr_pd[user_attrs_missing_index_list[i]],
                        user_attrs_complete[user_attrs_missing_index_list[i]][:, slice_l:slice_r], attr_type)
                    self.logger.info("[EPOCH={:03d}]: user_attr<{}, {}>: {}:{:.4f}".format(
                        epoch, i, *('multi', 'map') if attr_type == 1 else ('single', 'acc'), metric
                    ))
                    best_metric = max(best_metric, metric)
                    user_attr_input[user_attrs_missing_index_list[i]] = attr_pd[user_attrs_missing_index_list[i]]
                    if loss.item() < loss_threshold:
                        break
                user_max_metric_dict[i] = best_metric
                self.logger.info("Best metric result of {}th user attribute is {:.4f}".format(i, best_metric))

        if self.item_attr_cfg['have'] is True:
            self.logger.info("start handle item attribute...")
            train_I2U = self.load_util.get_I2U_from_U2I(train_U2I, to_list=False)
            trans_mat = self.get_transform_mat(self.get_similarity_mat(train_I2U, count=select_count),
                                               count=self.item_count,
                                               k=knn).cuda()
            attr_dim_list = self.item_attr_cfg['attr_dim_list']
            for i, dim_len in enumerate(attr_dim_list):
                attr_type = self.item_attr_cfg['attr_type_list'][i]
                slice_l = sum(attr_dim_list[0:i])
                slice_r = slice_l + attr_dim_list[i]
                model = LP()
                item_attr_input = torch.from_numpy(item_attrs_missing[:, slice_l:slice_r]).cuda()
                item_attr_gt = item_gt_list[i][:, slice_l:slice_r].cuda()
                self.logger.info("start train {}th item attribute".format(i))
                best_metric = -np.inf
                for epoch in range(epoch_num):
                    attr_pd = model.forward(item_attr_input, trans_mat)
                    loss = model.get_loss(attr_pd[item_attrs_existing_index_list[i]], item_attr_gt)
                    self.logger.info("[EPOCH={:03d}]: loss={}".format(epoch, loss.item()))
                    metric = self.evaluate_metric(
                        attr_pd[item_attrs_missing_index_list[i]],
                        item_attrs_complete[item_attrs_missing_index_list[i]][:, slice_l:slice_r], attr_type)
                    self.logger.info("[EPOCH={:03d}]: item_attr<{}, {}>: {}:{:.4f}".format(
                        epoch, i, *('multi', 'map') if attr_type == 1 else ('single', 'acc'), metric
                    ))
                    best_metric = max(best_metric, metric)
                    item_attr_input[item_attrs_missing_index_list[i]] = attr_pd[item_attrs_missing_index_list[i]]
                    if loss.item() < loss_threshold:
                        break
                item_max_metric_dict[i] = best_metric
                self.logger.info("Best metric result of {}th item attribute is {:.4f}".format(i, best_metric))

        # ====================
        self.logger.info("训练完毕！")
        item_max_metric_dict, user_max_metric_dict = dict(), dict()
        output_cont = []
        if self.item_attr_cfg['have'] is True:
            attr_dim_list = self.item_attr_cfg['attr_dim_list']
            attr_type_list = self.item_attr_cfg['attr_type_list']
            for i in range(len(attr_dim_list)):
                single_or_multi, metric_name = ('single', 'acc') if attr_type_list[i] == 0 else ('multi', 'map')
                output_cont.append("item_attr<%d, %s>: %s:%.4f" % (
                    i, single_or_multi, metric_name, item_max_metric_dict[i]
                ))
                self.logger.info(output_cont[-1])

        if self.user_attr_cfg['have'] is True:
            attr_dim_list = self.user_attr_cfg['attr_dim_list']
            attr_type_list = self.user_attr_cfg['attr_type_list']
            for i in range(len(attr_dim_list)):
                single_or_multi, metric_name = ('single', 'acc') if attr_type_list[i] == 0 else ('multi', 'map')
                output_cont.append("user_attr<%d, %s>: %s:%.4f" % (
                    i, single_or_multi, metric_name, user_max_metric_dict[i]
                ))
                self.logger.info(output_cont[-1])
        result = {
            "user_attr_max_metric": None if self.user_attr_cfg['have'] is False else user_max_metric_dict,
            "item_attr_max_metric": None if self.item_attr_cfg['have'] is False else item_max_metric_dict,
            "output_cont": output_cont
        }
        with open(self.tmpout_folder_path + "/" + "result.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    def evaluate_metric(self, attr_pd: torch.Tensor, attr_gt: np.ndarray, attr_type: int) -> float:
        """
            evaluate attr metric: acc or map
        :param attr_pd: predict attr
        :param attr_gt: groundtruth attr
        :param attr_type: attr_type=0<single label attr> attr_type=1<multi label attr>
        :return: acc or map
        """
        assert attr_pd.shape[1] == attr_gt.shape[1] and attr_pd.shape[0] == attr_gt.shape[0]
        assert attr_type in [0, 1]
        count = attr_pd.shape[0]
        attr_pd = tensor2npy(attr_pd)

        metric_list = []
        if attr_type == 0:  # single
            for _id in range(count):
                pd, gt = attr_pd[_id], attr_gt[_id]
                if np.sum(gt) == 0:
                    continue
                max_val_ind = np.argmax(gt)
                if pd[max_val_ind] == np.max(pd):
                    metric_list.append(1)
        else:  # multi
            for _id in range(count):
                pd, gt = attr_pd[_id], attr_gt[_id]
                if np.sum(gt) == 0:
                    continue
                metric_list.append(average_precision_score(gt, pd))
        return round(sum(metric_list) / len(metric_list), 6)

    def get_similarity_mat(self, _2_: dict, count: int = 1000) -> dict:
        """
            随机选取count个user或item
            for user: 将两个用户购买的商品重合比例作为相似度
            for item: 将两个商品被购买的用户的重合比例作为相似度
        :param _2_: U2I or I2U
        :param count: sample count
        :return: sim_mat
        """
        sim_mat = defaultdict(defaultdict)
        id_set = set(_2_.keys())
        for id1 in tqdm(_2_.keys()):
            id_list = random.sample(id_set, count)
            for id2 in id_list:
                if id1 == id2:
                    continue
                sim_mat[id1][id2] = len(set(_2_[id1]) & set(_2_[id2])) / math.sqrt(len(_2_[id1]) * len(_2_[id2]))
        return sim_mat

    def get_knn_id_dict(self, sim_mat: dict, k=20) -> dict:
        """
            record the topk similar object id
        :param sim_mat: sim_mat
        :param k: knn value
        :return: knn dict
        """
        knn_id_dict = dict()
        for _id in sim_mat.keys():
            sim_list = sorted(sim_mat[_id].items(), key=lambda item: item[1], reverse=True)[:k]
            knn_id_dict[_id] = [item[0] for item in sim_list]
        return knn_id_dict

    def get_transform_mat(self, sim_mat: dict, count: int, k: int = 20) -> torch.Tensor:
        """
            计算转换矩阵
        :param sim_mat: 相似矩阵
        :param count: user数量或item数量
        :param k: knn value
        :return: sparse Tensor
        """
        knn_id_dict = self.get_knn_id_dict(sim_mat, k=20)
        indices = [[], []]
        values = []
        for _id in knn_id_dict.keys():
            indices[0].extend([_id] * len(knn_id_dict[_id]))
            indices[1].extend(knn_id_dict[_id])
            values.extend([1 / k] * len(knn_id_dict[_id]))
        return torch.sparse_coo_tensor(indices=torch.Tensor(indices), values=values, size=(count, count),
                                       dtype=torch.float)
