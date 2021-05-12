import math
import random
import json
from utils import tensor2npy
from torch.utils.data import DataLoader
from core.data.loadUtil import LoadUtil
from core.data.dataset import TrainDataset
from core.session.fmsession import FM_Session
from core.evaluate.performance import evaluate
from utils import UnionConfig
import torch
import numpy as np
from core.models.FM import FM
from collections import defaultdict


class FM_Manager(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_name = self.cfg.dataset_name
        self.tmpout_folder_path = self.cfg['tmpout_folder_path']
        self.logger = self.cfg.logger
        self.yml_cfg = self.cfg.yml_cfg
        self.model_cfg = self.yml_cfg['FM']

        self.item_attr_cfg = self.yml_cfg['item_attr']
        self.user_attr_cfg = self.yml_cfg['user_attr']
        self.user_count = self.yml_cfg['user_count']
        self.item_count = self.yml_cfg['item_count']
        self.load_util = LoadUtil(cfg=self.cfg)

    def start(self):
        # load data
        train_U2I = self.load_util.load_train_U2I()
        val_U2I = self.load_util.load_val_U2I()
        test_U2I = self.load_util.load_test_U2I()
        train_and_val_U2I = self.load_util.merge_U2I_dict(train_U2I, val_U2I, self.user_count)

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

        # define model
        model = FM(cfg=self.cfg).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.model_cfg['learning_rate'])
        item_attrs_input = torch.from_numpy(item_attrs_missing).cuda() if self.item_attr_cfg['have'] else None
        user_attrs_input = torch.from_numpy(user_attrs_missing).cuda() if self.user_attr_cfg['have'] else None
        epoch_num = self.model_cfg['epoch_num']
        batch_size = self.model_cfg['batch_size']
        stop_epoch = self.model_cfg['stop_epoch']

        dataloader = DataLoader(
            TrainDataset(TrainDataset.gene_UI_from_U2I(train_U2I, num_neg_item=self.model_cfg['neg_item_num']),
                         train_U2I, self.item_count),
            batch_size, num_workers=4, shuffle=True
        )
        sess = FM_Session(self.cfg, model)
        sess.inject_static_data(
            user_attrs_existing_index_list=user_attrs_existing_index_list,
            item_attrs_existing_index_list=item_attrs_existing_index_list,
            user_gt_list=user_gt_list,
            item_gt_list=item_gt_list,
            user_attrs_input=user_attrs_input,
            item_attrs_input=item_attrs_input
        )

        epoch_metric_dict = defaultdict(lambda: defaultdict(dict))
        best_ndcg = -np.inf
        best_output = []
        stop_epoch_count = 0
        for epoch in range(epoch_num):
            mean_loss, mean_loss1, mean_loss2, mean_auc = sess.train(dataloader, optimizer)
            self.logger.info("[EPOCH=%03d]: (auc=%.4f) (loss=%.4f) (loss1=%.4f) (loss2=%.4f)" % (
                epoch, mean_auc, mean_loss, mean_loss1, mean_loss2))

            with torch.no_grad():
                user_emb, item_emb = model(user_attrs_input=user_attrs_input, item_attrs_input=item_attrs_input)
                user_emb = tensor2npy(user_emb)
                item_emb = tensor2npy(item_emb)

                perf_info, all_perf = evaluate(
                    user_emb, item_emb, train_and_val_U2I, test_U2I, args=UnionConfig({
                        'topks': self.model_cfg['test_topks'],
                        'cores': 4
                    }))
                output_cont = []
                for i, topk in enumerate(self.model_cfg['test_topks']):
                    epoch_metric_dict[epoch][topk] = {
                        'ndcg': float(perf_info[i * 3]),
                        'hr': float(perf_info[i * 3 + 1]),
                        'recall': float(perf_info[i * 3 + 2])
                    }
                    output_cont.append("[epoch=%03d]@%d: (ndcg=%.4f) (hr=%.4f) (recall=%.4f)" % (
                        epoch, topk, perf_info[i * 3], perf_info[i * 3 + 1], perf_info[i * 3 + 2]))
                    self.logger.info(output_cont[-1])
                stop_epoch_count += 1
                if best_ndcg <= epoch_metric_dict[epoch][10]['ndcg']:
                    best_ndcg = epoch_metric_dict[epoch][10]['ndcg']
                    best_output = output_cont
                np.save(self.tmpout_folder_path + "/all_metric/all_metrics-{}.npy".format(epoch), all_perf)
                if stop_epoch_count > stop_epoch:
                    break
        self.logger.info("Train and Test complete! The best metric of epochs: \n" + '\n'.join(best_output))
        with open(self.tmpout_folder_path + "/results.json", 'w', encoding='utf-8') as f:
            data = {
                "best_info": best_output,
                "epoch_record": epoch_metric_dict
            }
            json.dump(data, f, indent=4, ensure_ascii=False)
