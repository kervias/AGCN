from core.models.AGCN import AGCN
from core.data.loadUtil import LoadUtil
import torch
import numpy as np
import random
from tqdm import tqdm
import math, json
from collections import defaultdict
from sklearn.metrics import average_precision_score, accuracy_score


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
        data_util = LoadUtil(settings=self.cfg)

        # item_attrs_complete = None
        item_attrs_missing = None
        item_attrs_existing_index_list = None
        item_attrs_missing_index_list = None
        item_gt_list = None
        if self.item_attr_cfg['have'] is True:
            # item_attrs_complete = data_util.load_item_attrs_complete()
            item_attrs_missing = data_util.load_item_attrs_missing()
            item_attrs_existing_index_list = data_util.load_item_attrs_existing_index()
            item_attrs_missing_index_list = data_util.load_item_attrs_missing_index()
            item_gt_list = data_util.load_item_gt_list()

        # user_attrs_complete = None
        user_attrs_missing = None
        user_attrs_missing_index_list = None
        user_attrs_existing_index_list = None
        user_gt_list = None
        if self.user_attr_cfg['have'] is True:
            # user_attrs_complete = data_util.load_user_attrs_complete()
            user_attrs_missing = data_util.load_user_attrs_missing()
            user_attrs_existing_index_list = data_util.load_user_attrs_existing_index()
            user_attrs_missing_index_list = data_util.load_user_attrs_missing_index()
            user_gt_list = data_util.load_user_gt_list()

        graph_adjmat = data_util.load_graph_adj_mat()

        # 2. 初始化 model
        model = AGCN(settings=self.cfg).cuda()
        # model.to(self.device)
        model.init_net_data(
            graph_adj_mat=graph_adjmat.cuda()
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.model_cfg['learning_rate'])
        # 将加载后的数据转换为Tensor
        item_attrs_input = item_attrs_missing.cuda() if self.item_attr_cfg['have'] else None
        user_attrs_input = user_attrs_missing.cuda() if self.user_attr_cfg['have'] else None
        epoch_num = self.model_cfg['epoch_num']
        batch_size = self.model_cfg['batch_size']

        # 3.迭代
        item_metrics_dict = defaultdict(list)
        user_metrics_dict = defaultdict(list)
        for epoch_ind in range(epoch_num):
            # 获得 BPR 数据
            triple_list = np.asarray(data_util.get_bpr_data(neg_sample_num=1))
            all_ind = list(range(len(triple_list)))
            random.shuffle(all_ind)

            sum_num, sum_loss, sum_loss1, sum_loss2, sum_auc = 0, 0.0, 0.0, 0.0, 0.0
            for i in tqdm(range(math.ceil(len(triple_list) / model_cfg['batch_size'])),
                          desc="[EPOCH=%d]" % (epoch_ind)):
                model(user_attrs_input=user_attrs_input, item_attrs_input=item_attrs_input)  # forward
                start_ind = i * model_cfg['batch_size']
                end_ind = min((i + 1) * model_cfg['batch_size'], len(triple_list))
                triple_batch = triple_list[all_ind[start_ind:end_ind]].T
                u_list, i_list, j_list = triple_batch[0], triple_batch[1], triple_batch[2]

                loss, loss1, loss2 = model.total_loss(
                    user_index=u_list, item_index_1=i_list, item_index_2=j_list,
                    user_existing_index_list=user_attrs_existing_index_list, user_gt_list=user_gt_list,
                    item_existing_index_list=item_attrs_existing_index_list, item_gt_list=item_gt_list
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # 统计数据
                sum_auc += model.auc.item() * len(u_list)
                sum_loss += loss.item() * len(u_list)
                sum_loss1 += loss1.item() * len(u_list)
                sum_loss2 += loss2.item() * len(u_list)
                sum_num += len(u_list)
                # self.logger.info("[ITER=%d|EPOCH=%d|BATCH=%d]: (auc=%.4f) (loss=%.4f) (loss1=%.4f) (loss2=%.4f)" % (iter_ind, epoch_ind, i, model.auc.item(), loss.item(), loss1.item(), loss2.item()))
            mean_auc = round(sum_auc / sum_num, 4)
            mean_loss = round(sum_loss / sum_num, 4)
            mean_loss1 = round(sum_loss1 / sum_num, 4)
            mean_loss2 = round(sum_loss2 / sum_num, 4)
            self.logger.info("[EPOCH=%d]: (auc=%.4f) (loss=%.4f) (loss1=%.4f) (loss2=%.4f)" % (
                epoch_ind, mean_auc, mean_loss, mean_loss1, mean_loss2))

            with torch.no_grad():
                model(user_attrs_input=user_attrs_input, item_attrs_input=item_attrs_input)
                if model_cfg['item_attr'].get('have', False) is True:
                    attr_dim_list = model_cfg['item_attr']['attr_dim_list']
                    attr_type_list = model_cfg['item_attr']['attr_type_list']
                    for i in range(len(attr_dim_list)):
                        slice_l = sum(attr_dim_list[0:i])
                        slice_r = slice_l + attr_dim_list[i]
                        item_attrs_infer = model.item_attr_inference(item_attrs_missing_index_list[i])[:,
                                           slice_l:slice_r]
                        # item_attrs_infer = torch.softmax(item_attrs_infer, 1)
                        # 替换属性
                        item_attrs_input[item_attrs_missing_index_list[i], slice_l:slice_r] = item_attrs_infer
                        # item_attrs_input[item_attrs_missing_index_list[i], slice_l:slice_r] = torch.softmax(item_attrs_infer,1)
                        if self.settings['DEVICE'] == 'cuda':
                            item_attrs_infer = item_attrs_infer.cpu()
                        item_attrs_infer = item_attrs_infer.detach().numpy()
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

                if model_cfg['user_attr'].get('have', False) is True:
                    attr_dim_list = model_cfg['user_attr']['attr_dim_list']
                    attr_type_list = model_cfg['user_attr']['attr_type_list']
                    for i in range(len(attr_dim_list)):
                        slice_l = sum(attr_dim_list[0:i])
                        slice_r = slice_l + attr_dim_list[i]
                        user_attrs_infer = model.user_attr_inference(user_attrs_missing_index_list[i])[:,
                                           slice_l:slice_r]
                        # user_attrs_infer = torch.softmax(user_attrs_infer, 1)
                        user_attrs_input[user_attrs_missing_index_list[i], slice_l:slice_r] = user_attrs_infer
                        # user_attrs_input[user_attrs_missing_index_list[i], slice_l:slice_r] = torch.softmax(user_attrs_infer,1)
                        if self.settings['DEVICE'] == 'cuda':
                            user_attrs_infer = user_attrs_infer.cpu()
                        user_attrs_infer = user_attrs_infer.detach().numpy()
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
                if model_cfg['item_attr'].get('have', False) is True:
                    for i in range(model_cfg['item_attr']['attr_type_num']):
                        bool_item = bool_item and item_metrics_dict[i][-1] <= item_metrics_dict[i][-2] <= \
                                    item_metrics_dict[i][-3]

                if model_cfg['user_attr'].get('have', False) is True:
                    for i in range(model_cfg['user_attr']['attr_type_num']):
                        bool_user = bool_user and user_metrics_dict[i][-1] <= user_metrics_dict[i][-2] <= \
                                    user_metrics_dict[i][-3]
                if bool_item is True and bool_user is True:
                    break

        self.logger.info("收敛完毕！")
        item_max_metric_dict, user_max_metric_dict = dict(), dict()
        output_cont = []
        if model_cfg['item_attr'].get('have', False) is True:
            attr_dim_list = model_cfg['item_attr']['attr_dim_list']
            attr_type_list = model_cfg['item_attr']['attr_type_list']
            for i in range(len(attr_dim_list)):
                single_or_multi, metric_name = ('single', 'acc') if attr_type_list[i] == 0 else ('multi', 'map')
                item_max_metric_dict[i] = max(item_metrics_dict[i])
                output_cont.append("item_attr<%d, %s>: %s:%.4f" % (
                    i, single_or_multi, metric_name, item_max_metric_dict[i]
                ))
                self.logger.info(output_cont[-1])

        if model_cfg['user_attr'].get('have', False) is True:
            attr_dim_list = model_cfg['user_attr']['attr_dim_list']
            attr_type_list = model_cfg['user_attr']['attr_type_list']
            for i in range(len(attr_dim_list)):
                single_or_multi, metric_name = ('single', 'acc') if attr_type_list[i] == 0 else ('multi', 'map')
                user_max_metric_dict[i] = max(user_metrics_dict[i])
                output_cont.append("user_attr<%d, %s>: %s:%.4f" % (
                    i, single_or_multi, metric_name, user_max_metric_dict[i]
                ))
                self.logger.info(output_cont[-1])
        result = {
            "user_attr_metric": None if model_cfg['user_attr'].get('have', False) is False else dict(user_metrics_dict),
            "item_attr_metric": None if model_cfg['item_attr'].get('have', False) is False else dict(item_metrics_dict),
            "user_attr_max_metric": None if model_cfg['user_attr'].get('have',
                                                                       False) is False else user_max_metric_dict,
            "item_attr_max_metric": None if model_cfg['item_attr'].get('have',
                                                                       False) is False else item_max_metric_dict,
            "output_cont": output_cont
        }
        with open(self.output_folder_path + "/" + "result.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
