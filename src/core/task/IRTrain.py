from core.models.AGCN import AGCN
from core.data.loadUtil import LoadUtil
from core.data.dataset import TrainDataset
from core.evaluate.evaluate import Evaluate
from core.data.graph import LaplaceGraph
import torch
from torch.utils.data import DataLoader
import numpy as np
from core.session.irsession import IRSession
from utils import tensor2npy


class IR_Train(object):
    """
        item recommendation train
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.model_cfg = self.cfg.yml_cfg['IR-Train']
        self.yml_cfg = self.cfg.yml_cfg
        self.item_attr_cfg = self.yml_cfg['item_attr']
        self.user_attr_cfg = self.yml_cfg['user_attr']

        self.dataset_name = self.cfg.dataset_name
        self.tmpout_folder_path = self.cfg['tmpout_folder_path']
        self.logger = self.cfg.logger
        self.user_count = self.yml_cfg['user_count']
        self.item_count = self.yml_cfg['item_count']

    def train(self):
        # 1. 加载数据
        loadutil = LoadUtil(cfg=self.cfg)
        train_U2I = loadutil.load_train_U2I()
        dict_val_data = loadutil.load_val_U2I()
        dict_neg_1000_data = loadutil.get_neg_samples(force_regenerate=False, neg_sample_num=1000)
        # item_attrs_complete = None
        item_attrs_missing = None
        item_attrs_existing_index_list = None
        item_attrs_missing_index_list = None
        item_gt_list = None
        if self.item_attr_cfg['have'] is True:
            # item_attrs_complete = loadutil.load_item_attrs_complete()
            item_attrs_missing = loadutil.load_item_attrs_missing()
            item_attrs_existing_index_list = loadutil.load_item_attrs_existing_index()
            item_attrs_missing_index_list = loadutil.load_item_attrs_missing_index()
            item_gt_list = loadutil.load_item_gt_list()

        # user_attrs_complete = None
        user_attrs_missing = None
        user_attrs_missing_index_list = None
        user_attrs_existing_index_list = None
        user_gt_list = None
        if self.user_attr_cfg['have'] is True:
            # user_attrs_complete = loadutil.load_user_attrs_complete()
            user_attrs_missing = loadutil.load_user_attrs_missing()
            user_attrs_existing_index_list = loadutil.load_user_attrs_existing_index()
            user_attrs_missing_index_list = loadutil.load_user_attrs_missing_index()
            user_gt_list = loadutil.load_user_gt_list()

        graph_adjmat = LaplaceGraph(
            n_users=self.user_count,
            n_items=self.item_count,
            train_U2I=train_U2I
        ).generate(add_self_loop=False)
        # 2. 初始化 model
        model = AGCN(cfg=self.cfg, task='IR-Train')
        model.init_net_data(
            graph_adj_mat=graph_adjmat.cuda()
        )
        model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.model_cfg['learning_rate'])
        item_attrs_input = torch.from_numpy(item_attrs_missing).cuda() if self.item_attr_cfg['have'] else None
        user_attrs_input = torch.from_numpy(user_attrs_missing).cuda() if self.user_attr_cfg['have'] else None
        iter_num = self.model_cfg['iter_num']
        epoch_num = self.model_cfg['epoch_num']
        batch_size = self.model_cfg['batch_size']

        dataloader = DataLoader(
            TrainDataset(TrainDataset.gene_UI_from_U2I(train_U2I, num_neg_item=self.model_cfg['neg_item_num']),
                         train_U2I, self.item_count),
            batch_size, num_workers=4, shuffle=True
        )
        sess = IRSession(self.cfg, model)
        sess.inject_static_data(
            user_attrs_existing_index_list=user_attrs_existing_index_list,
            item_attrs_existing_index_list=item_attrs_existing_index_list,
            user_gt_list=user_gt_list,
            item_gt_list=item_gt_list
        )
        # 3.开始三层迭代
        iter_hr_list, iter_ndcg_list = [], []  # 记录每个epoch中的最优值
        all_best_pth = []
        pth_fmt = self.dataset_name + "-{iter:03d}-{epoch:03d}.pth"
        for iter_ind in range(iter_num):
            epoch_hr_list, epoch_ndcg_list = [], []  # 记录Epoch中每次迭代评价指标
            best_ndcg, best_hr, best_epoch = 0, 0, 0
            for epoch_ind in range(epoch_num):
                mean_loss, mean_loss1, mean_loss2, mean_auc = sess.train(
                    dataloader, optimizer, user_attrs_input, item_attrs_input
                )
                self.logger.info("[ITER=%03d|EPOCH=%03d]: (auc=%.4f) (loss=%.4f) (loss1=%.4f) (loss2=%.4f)" % (
                    iter_ind, epoch_ind, mean_auc, mean_loss, mean_loss1, mean_loss2))

                # 计算hr和ndcg，将验证集中的数据与负样本评分对照排序
                dict_hr, dict_ndcg = Evaluate.get_hr_ndcg_for_val(
                    tensor2npy(model.final_user_emb), tensor2npy(model.final_item_emb),
                    dict_val_data, dict_neg_1000_data, [10])
                hr, ndcg = dict_hr[10], dict_ndcg[10]
                self.logger.info(
                    "[ITER=%03d|EPOCH=%03d]: (HR@10=%.4f) (NDCG@10=%.4f)" % (iter_ind, epoch_ind, hr, ndcg))
                epoch_hr_list.append(hr)
                epoch_ndcg_list.append(ndcg)
                best_hr = max(best_hr, hr)
                if best_ndcg <= ndcg:
                    best_ndcg = ndcg
                    best_epoch = epoch_ind
                    pth_file_name = pth_fmt.format(iter=iter_ind, epoch=epoch_ind)
                    torch.save(model.state_dict(), f=self.tmpout_folder_path + "/pths/" + pth_file_name)

                # 跳出epoch条件
                if ((iter_ind == 0 and epoch_ind > 40) or (iter_ind > 0 and epoch_ind > 10)) and epoch_ndcg_list[-1] <= \
                        epoch_ndcg_list[-2] <= epoch_ndcg_list[-3]:
                    iter_hr_list.append(best_hr)
                    iter_ndcg_list.append(best_ndcg)
                    pth_file_name = pth_fmt.format(iter=iter_ind, epoch=best_epoch)
                    all_best_pth.append(pth_file_name)
                    model.load_state_dict(torch.load(self.tmpout_folder_path + "/pths/" + pth_file_name))
                    model(user_attrs_input=user_attrs_input, item_attrs_input=item_attrs_input)  # forward

                    with torch.no_grad():
                        # 更新item推理属性
                        if self.item_attr_cfg['have'] is True:
                            attr_dim_list = self.item_attr_cfg['attr_dim_list']
                            for i in range(len(attr_dim_list)):
                                slice_l = sum(attr_dim_list[0:i])
                                slice_r = slice_l + attr_dim_list[i]
                                item_attrs_infer = model.item_attr_inference(item_attrs_missing_index_list[i])[:,
                                                   slice_l:slice_r]
                                item_attrs_input[item_attrs_missing_index_list[i], slice_l:slice_r] = item_attrs_infer
                        # 更新user推理属性
                        if self.user_attr_cfg['have'] is True:
                            attr_dim_list = self.user_attr_cfg['attr_dim_list']
                            for i in range(len(attr_dim_list)):
                                slice_l = sum(attr_dim_list[0:i])
                                slice_r = slice_l + attr_dim_list[i]
                                user_attrs_infer = model.user_attr_inference(user_attrs_missing_index_list[i])[:,
                                                   slice_l:slice_r]
                                user_attrs_input[user_attrs_missing_index_list[i], slice_l:slice_r] = user_attrs_infer
                    break
            # 跳出iter条件
            if iter_ind > 2 and iter_ndcg_list[-1] <= iter_ndcg_list[-2] <= iter_ndcg_list[-3]:
                break

        np.save(self.tmpout_folder_path + "/all_best_pth.npy", all_best_pth)
