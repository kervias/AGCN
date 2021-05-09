import math
import random
import json
from utils import tensor2npy
from core.data.loadUtil import LoadUtil
from core.data.graph import LaplaceGraph
from core.data.dataset import TrainDataset
from collections import defaultdict
import torch
import numpy as np
from core.models.SemiGCN import SemiGCN
from sklearn.metrics import average_precision_score, accuracy_score


class SemiGCN_Manager(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_name = self.cfg.dataset_name
        self.tmpout_folder_path = self.cfg['tmpout_folder_path']
        self.logger = self.cfg.logger
        self.yml_cfg = self.cfg.yml_cfg
        self.model_cfg = self.yml_cfg['Semi-GCN']

        self.item_attr_cfg = self.yml_cfg['item_attr']
        self.user_attr_cfg = self.yml_cfg['user_attr']
        self.user_count = self.yml_cfg['user_count']
        self.item_count = self.yml_cfg['item_count']
        self.load_util = LoadUtil(cfg=self.cfg)

    def start(self):
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

        graph_adjmat = LaplaceGraph(
            n_users=self.user_count,
            n_items=self.item_count,
            train_U2I=self.load_util.load_train_U2I()
        ).generate(add_self_loop=True, norm_type=1)

        model = SemiGCN(cfg=self.cfg)
        model.init_net_data(
            graph_adj_mat=graph_adjmat.cuda()
        )
        model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.model_cfg['learning_rate'])
        # 将加载后的数据转换为Tensor
        item_attrs_input = torch.from_numpy(item_attrs_missing).cuda() if self.item_attr_cfg['have'] else None
        user_attrs_input = torch.from_numpy(user_attrs_missing).cuda() if self.user_attr_cfg['have'] else None
        epoch_num = self.model_cfg['epoch_num']


        # 3.迭代
        item_metrics_dict = defaultdict(list)
        user_metrics_dict = defaultdict(list)
        for epoch in range(epoch_num):
            optimizer.zero_grad()
            model(user_attrs_input=user_attrs_input, item_attrs_input=item_attrs_input)
            loss = model.get_infer_loss(
                user_existing_index_list=user_attrs_existing_index_list, user_gt_list=user_gt_list,
                item_existing_index_list=item_attrs_existing_index_list, item_gt_list=item_gt_list
            )

            loss.backward()
            optimizer.step()
            self.logger.info("[EPOCH={:03d}]: loss={}".format(epoch, loss.item()))

            with torch.no_grad():
                final_user_emb, final_item_emb = model(
                    user_attrs_input=user_attrs_input,
                    item_attrs_input=item_attrs_input
                )
                if self.item_attr_cfg['have'] is True:
                    attr_dim_list = self.item_attr_cfg['attr_dim_list']
                    attr_type_list = self.item_attr_cfg['attr_type_list']
                    for i in range(len(attr_dim_list)):
                        attr_type = attr_type_list[i]
                        slice_l = sum(attr_dim_list[0:i])
                        slice_r = slice_l + attr_dim_list[i]
                        emb = final_item_emb[item_attrs_missing_index_list[i]]
                        attr_pd = model.infer_item_attr(emb)[:, slice_l:slice_r]
                        metric = self.evaluate_metric(
                            attr_pd,
                            item_attrs_complete[item_attrs_missing_index_list[i]][:, slice_l:slice_r], attr_type)
                        item_metrics_dict[i].append(metric)
                        self.logger.info("[EPOCH={:03d}]: item_attr<{}, {}>: {}:{:.4f}".format(
                            epoch, i, *('multi', 'map') if attr_type == 1 else ('single', 'acc'), metric
                        ))

                if self.user_attr_cfg['have'] is True:
                    attr_dim_list = self.user_attr_cfg['attr_dim_list']
                    attr_type_list = self.user_attr_cfg['attr_type_list']
                    for i in range(len(attr_dim_list)):
                        attr_type = attr_type_list[i]
                        slice_l = sum(attr_dim_list[0:i])
                        slice_r = slice_l + attr_dim_list[i]
                        emb = final_user_emb[user_attrs_missing_index_list[i]]
                        attr_pd = model.infer_user_attr(emb)[:, slice_l:slice_r]
                        metric = self.evaluate_metric(
                            attr_pd,
                            user_attrs_complete[user_attrs_missing_index_list[i]][:, slice_l:slice_r], attr_type)
                        user_metrics_dict[i].append(metric)
                        self.logger.info("[EPOCH={:03d}]: user_attr<{}, {}>: {}:{:.4f}".format(
                            epoch, i, *('multi', 'map') if attr_type == 1 else ('single', 'acc'), metric
                        ))

        item_max_metric_dict, user_max_metric_dict = dict(), dict()
        output_cont = []
        if self.item_attr_cfg['have'] is True:
            attr_dim_list = self.item_attr_cfg['attr_dim_list']
            attr_type_list = self.item_attr_cfg['attr_type_list']
            for i in range(len(attr_dim_list)):
                single_or_multi, metric_name = ('single', 'acc') if attr_type_list[i] == 0 else ('multi', 'map')
                item_max_metric_dict[i] = max(item_metrics_dict[i])
                output_cont.append("item_attr<%d, %s>: %s:%.4f" % (
                    i, single_or_multi, metric_name, item_max_metric_dict[i]
                ))
                self.logger.info(output_cont[-1])

        if self.user_attr_cfg['have'] is True:
            attr_dim_list = self.user_attr_cfg['attr_dim_list']
            attr_type_list = self.user_attr_cfg['attr_type_list']
            for i in range(len(attr_dim_list)):
                single_or_multi, metric_name = ('single', 'acc') if attr_type_list[i] == 0 else ('multi', 'map')
                user_max_metric_dict[i] = max(user_metrics_dict[i])
                output_cont.append("user_attr<%d, %s>: %s:%.4f" % (
                    i, single_or_multi, metric_name, user_max_metric_dict[i]
                ))
                self.logger.info(output_cont[-1])
        result = {
            "user_attr_metric": None if self.user_attr_cfg['have'] is False else dict(user_metrics_dict),
            "item_attr_metric": None if self.item_attr_cfg['have'] is False else dict(item_metrics_dict),
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
        metric = None
        if attr_type == 0:  # single
            non_count = 0
            for _id in range(count):
                pd, gt = attr_pd[_id], attr_gt[_id]
                if np.sum(gt) == 0:
                    non_count += 1
                    continue
                max_val_ind = np.argmax(gt)
                if pd[max_val_ind] == np.max(pd):
                    metric_list.append(1)
            metric = sum(metric_list) / (count - non_count) # acc
        else:  # multi
            for _id in range(count):
                pd, gt = attr_pd[_id], attr_gt[_id]
                if np.sum(gt) == 0:
                    continue
                metric_list.append(average_precision_score(gt, pd))
            metric = sum(metric_list) / len(metric_list) # map
        return round(metric, 6)