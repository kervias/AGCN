import json

from core.models.BPRMF import BPRMF
from core.session.bprmfSession import BPRMF_Session
from core.data.loadUtil import LoadUtil
from core.data.dataset import TrainDataset
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils import UnionConfig, tensor2npy
from core.evaluate.performance import evaluate
from collections import defaultdict

class BPRMF_Manager(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_name = self.cfg.dataset_name
        self.tmpout_folder_path = self.cfg['tmpout_folder_path']
        self.logger = self.cfg.logger
        self.yml_cfg = self.cfg.yml_cfg
        self.model_cfg = self.yml_cfg['BPRMF']

        self.item_attr_cfg = self.yml_cfg['item_attr']
        self.user_attr_cfg = self.yml_cfg['user_attr']
        self.user_count = self.yml_cfg['user_count']
        self.item_count = self.yml_cfg['item_count']
        self.load_util = LoadUtil(cfg=self.cfg)

    def start(self):
        train_U2I = self.load_util.load_train_U2I()
        val_U2I = self.load_util.load_val_U2I()
        test_U2I = self.load_util.load_test_U2I()
        train_and_val_U2I = self.load_util.merge_U2I_dict(train_U2I, val_U2I, self.user_count)
        batch_size = self.model_cfg['batch_size']
        epoch_num = self.model_cfg['epoch_num']
        stop_epoch = self.model_cfg['stop_epoch']

        model = BPRMF(cfg=self.cfg).cuda()
        dataloader = DataLoader(
            TrainDataset(TrainDataset.gene_UI_from_U2I(train_U2I, num_neg_item=self.model_cfg['neg_item_num']),
                         train_U2I, self.item_count),
            batch_size, num_workers=4, shuffle=True
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.model_cfg['learning_rate'])
        sess = BPRMF_Session(model)

        epoch_metric_dict = defaultdict(lambda: defaultdict(dict))
        best_ndcg = -np.inf
        best_output = []
        stop_epoch_count = 0
        for epoch in range(epoch_num):
            loss = sess.train(dataloader, optimizer)
            self.logger.info("[epoch:{:03d}]: loss:[{:.6f}] = mf:[{:.6f}] + reg:[{:.6f}]".format(epoch, *loss))
            model.eval()
            with torch.no_grad():
                user_emb, item_emb = model.propagate()
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
                    stop_epoch_count = 0
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
