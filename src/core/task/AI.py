from core.models.AGCN import AGCN
from core.data.loadUtil import LoadUtil
import torch
import numpy as np
import json
from collections import defaultdict
from core.session.aisession import AISession
from sklearn.metrics import average_precision_score, accuracy_score
from torch.utils.data import DataLoader
from core.data.dataset import TrainDataset
from utils import tensor2npy


class AI(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_name = self.cfg.dataset_name
        self.tmpout_folder_path = self.cfg['tmpout_folder_path']
        self.logger = self.cfg.logger
        self.model_cfg = self.cfg.model_cfg
        self.item_attr_cfg = self.model_cfg['item_attr']
        self.user_attr_cfg = self.model_cfg['user_attr']

    def start(self):
        # 1. 加载数据
        loadutil = LoadUtil(settings=self.cfg)
        train_U2I = loadutil.load_train_U2I()
        
        item_attrs_complete = None
        item_attrs_missing = None
        item_attrs_existing_index_list = None
        item_attrs_missing_index_list = None
        item_gt_list = None
        if self.item_attr_cfg['have'] is True:
            item_attrs_complete = loadutil.load_item_attrs_complete()
            item_attrs_missing = loadutil.load_item_attrs_missing()
            item_attrs_existing_index_list = loadutil.load_item_attrs_existing_index()
            item_attrs_missing_index_list = loadutil.load_item_attrs_missing_index()
            item_gt_list = loadutil.load_item_gt_list()

        user_attrs_complete = None
        user_attrs_missing = None
        user_attrs_missing_index_list = None
        user_attrs_existing_index_list = None
        user_gt_list = None
        if self.user_attr_cfg['have'] is True:
            user_attrs_complete = loadutil.load_user_attrs_complete()
            user_attrs_missing = loadutil.load_user_attrs_missing()
            user_attrs_existing_index_list = loadutil.load_user_attrs_existing_index()
            user_attrs_missing_index_list = loadutil.load_user_attrs_missing_index()
            user_gt_list = loadutil.load_user_gt_list()

        graph_adjmat = loadutil.load_graph_adj_mat()

        # 2. 初始化 model
        model = AGCN(settings=self.cfg)
        model.init_net_data(
            graph_adj_mat=graph_adjmat.cuda()
        )
        model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.model_cfg['learning_rate'])
        # 将加载后的数据转换为Tensor
        item_attrs_input = torch.from_numpy(item_attrs_missing).cuda() if self.item_attr_cfg['have'] else None
        user_attrs_input = torch.from_numpy(user_attrs_missing).cuda() if self.user_attr_cfg['have'] else None
        epoch_num = self.model_cfg['epoch_num']
        batch_size = self.model_cfg['batch_size']

        dataloader = DataLoader(
            TrainDataset(TrainDataset.gene_UI_from_U2I(train_U2I, num_neg_item=self.model_cfg['neg_item_num']),
                         train_U2I, self.model_cfg['item_count']),
            batch_size, num_workers=4, shuffle=True
        )
        sess = AISession(self.cfg, model)
        sess.inject_static_data(
            user_attrs_existing_index_list=user_attrs_existing_index_list,
            item_attrs_existing_index_list=item_attrs_existing_index_list,
            user_gt_list=user_gt_list,
            item_gt_list=item_gt_list
        )
        
        # 3.迭代
        item_metrics_dict = defaultdict(list)
        user_metrics_dict = defaultdict(list)
        for epoch_ind in range(epoch_num):
            mean_loss, mean_loss1, mean_loss2, mean_auc = sess.train(
                dataloader, optimizer, user_attrs_input, item_attrs_input
            )
            self.logger.info("[EPOCH=%d]: (auc=%.4f) (loss=%.4f) (loss1=%.4f) (loss2=%.4f)" % (
                epoch_ind, mean_auc, mean_loss, mean_loss1, mean_loss2))

            with torch.no_grad():
                model(user_attrs_input=user_attrs_input, item_attrs_input=item_attrs_input)
                if self.item_attr_cfg['have'] is True:
                    attr_dim_list = self.item_attr_cfg['attr_dim_list']
                    attr_type_list = self.item_attr_cfg['attr_type_list']
                    for i in range(len(attr_dim_list)):
                        slice_l = sum(attr_dim_list[0:i])
                        slice_r = slice_l + attr_dim_list[i]
                        item_attrs_infer = model.item_attr_inference(item_attrs_missing_index_list[i])[:,
                                           slice_l:slice_r]
                        # item_attrs_infer = torch.softmax(item_attrs_infer, 1)
                        # 替换属性
                        item_attrs_input[item_attrs_missing_index_list[i], slice_l:slice_r] = item_attrs_infer
                        # item_attrs_input[item_attrs_missing_index_list[i], slice_l:slice_r] = torch.softmax(item_attrs_infer,1)
                        item_attrs_infer = tensor2npy(item_attrs_infer)
                        if attr_type_list[i] == 1:
                            # multi-label attribute
                            ap_list = []
                            for j, item_id in enumerate(item_attrs_missing_index_list[i]):
                                gt_label = item_attrs_complete[item_id][slice_l:slice_r]
                                if np.sum(gt_label) == 0:
                                    continue
                                pd_label = item_attrs_infer[j]
                                ap = average_precision_score(gt_label,
                                                             pd_label)  # Evaluate.comput_ap(gt_label, pd_label)
                                if ap is not None:
                                    ap_list.append(ap)
                            map = sum(ap_list) / len(ap_list)
                            item_metrics_dict[i].append(map)
                            self.logger.info("[EPOCH=%d]: item_attr<%d, multi>: map:%.4f" % (
                                epoch_ind, i, map
                            ))
                        else:
                            # single-label attribute
                            acc, count = 0, 0
                            for j, item_id in enumerate(item_attrs_missing_index_list[i]):
                                gt_label = item_attrs_complete[item_id][slice_l:slice_r]  # .astype(np.bool_).tolist()
                                if np.sum(gt_label) == 0:
                                    count += 1
                                    continue
                                pd_label = item_attrs_infer[j]
                                max_val_ind = np.argmax(gt_label)
                                if pd_label[max_val_ind] == np.max(pd_label):
                                    acc += 1
                                # gt_label = item_attrs_complete[item_id][slice_l:slice_r].astype(np.bool_).tolist()
                                # pd_label = (item_attrs_infer[item_id] >= np.max(item_attrs_infer[item_id])).tolist()
                                # if gt_label == pd_label:
                                #     acc += 1
                            acc /= (item_attrs_infer.shape[0] - count)
                            item_metrics_dict[i].append(acc)
                            self.logger.info("[EPOCH=%d]: item_attr<%d, single>: acc:%.4f" % (
                                epoch_ind, i, acc
                            ))

                if self.user_attr_cfg['have'] is True:
                    attr_dim_list = self.user_attr_cfg['attr_dim_list']
                    attr_type_list = self.user_attr_cfg['attr_type_list']
                    for i in range(len(attr_dim_list)):
                        slice_l = sum(attr_dim_list[0:i])
                        slice_r = slice_l + attr_dim_list[i]
                        user_attrs_infer = model.user_attr_inference(user_attrs_missing_index_list[i])[:,
                                           slice_l:slice_r]
                        # user_attrs_infer = torch.softmax(user_attrs_infer, 1)
                        user_attrs_input[user_attrs_missing_index_list[i], slice_l:slice_r] = user_attrs_infer
                        # user_attrs_input[user_attrs_missing_index_list[i], slice_l:slice_r] = torch.softmax(user_attrs_infer,1)
                        user_attrs_infer = tensor2npy(user_attrs_infer)
                        if attr_type_list[i] == 1:
                            # multi-label attribute
                            ap_list = []
                            for j, user_id in enumerate(user_attrs_missing_index_list[i]):
                                gt_label = user_attrs_complete[user_id][slice_l:slice_r]
                                if sum(gt_label) == 0:
                                    continue
                                pd_label = user_attrs_infer[j]
                                ap = average_precision_score(gt_label,
                                                             pd_label)  # Evaluate.comput_ap(gt_label, pd_label)
                                # ap_list.append(ap)
                                if ap is not None:
                                    ap_list.append(ap)
                            map = sum(ap_list) / len(ap_list)
                            user_metrics_dict[i].append(map)
                            self.logger.info("[EPOCH=%d]:user_attr<%d, multi>: map:%.4f" % (
                                epoch_ind, i, map
                            ))
                        else:
                            # single-label attribute
                            acc = 0
                            count = 0
                            for j, user_id in enumerate(user_attrs_missing_index_list[i]):
                                gt_label = user_attrs_complete[user_id][slice_l:slice_r]  # .astype(np.bool_).tolist()
                                if np.sum(gt_label) == 0:
                                    count += 1
                                    continue
                                pd_label = user_attrs_infer[j]
                                max_val_ind = np.argmax(gt_label)
                                # pd_label = [False]*len(user_attrs_infer[user_id])
                                # pd_label[max_val_ind] = True
                                if pd_label[max_val_ind] == np.max(pd_label):
                                    acc += 1
                                # pd_label = (user_attrs_infer[user_id] >= np.max(user_attrs_infer[user_id])).tolist()
                                # if gt_label == pd_label:
                                #     acc += 1
                            acc /= (user_attrs_infer.shape[0] - count)
                            user_metrics_dict[i].append(acc)
                            self.logger.info("[EPOCH=%d]: user_attr<%d, single>: acc:%.4f" % (
                                epoch_ind, i, acc
                            ))

            if epoch_ind > 10:  # 收敛条件判断
                bool_item, bool_user = True, True
                if self.item_attr_cfg['have'] is True:
                    for i in range(self.item_attr_cfg['attr_type_num']):
                        bool_item = bool_item and item_metrics_dict[i][-1] <= item_metrics_dict[i][-2] <= \
                                    item_metrics_dict[i][-3]

                if self.user_attr_cfg['have'] is True:
                    for i in range(self.user_attr_cfg['attr_type_num']):
                        bool_user = bool_user and user_metrics_dict[i][-1] <= user_metrics_dict[i][-2] <= \
                                    user_metrics_dict[i][-3]
                if bool_item is True and bool_user is True:
                    break

        self.logger.info("收敛完毕！")
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
