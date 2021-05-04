from core.models.AGCN import AGCN
from core.data.loadUtil import LoadUtil
from core.data.graph import LaplaceGraph
from core.evaluate.evaluate import Evaluate
from core.evaluate.performance import evaluate
import torch
import numpy as np
import json
from utils import tensor2npy


class IR_Test(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_name = self.cfg.dataset_name
        self.tmpout_folder_path = self.cfg['tmpout_folder_path']
        self.logger = self.cfg.logger
        self.model_cfg = self.cfg.model_cfg
        self.item_attr_cfg = self.model_cfg['item_attr']
        self.user_attr_cfg = self.model_cfg['user_attr']
        self.train_folder_path = self.cfg.train_folder_path

    def test(self):
        # 1. 加载数据
        loadutil = LoadUtil(settings=self.cfg)
        dict_test_data = loadutil.load_test_U2I()
        complete_data = loadutil.load_total_U2I()

        item_attrs_missing = None
        item_attrs_missing_index_list = None
        if self.item_attr_cfg['have'] is True:
            item_attrs_missing = loadutil.load_item_attrs_missing()
            item_attrs_missing_index_list = loadutil.load_item_attrs_missing_index()

        user_attrs_missing = None
        user_attrs_missing_index_list = None
        if self.user_attr_cfg['have'] is True:
            user_attrs_missing = loadutil.load_user_attrs_missing()
            user_attrs_missing_index_list = loadutil.load_user_attrs_missing_index()

        graph_adjmat = LaplaceGraph(
            n_users=self.cfg.model_cfg['user_count'],
            n_items=self.cfg.model_cfg['item_count'],
            train_U2I=loadutil.load_train_U2I()
        ).generate(add_self_loop=False, norm_type=2)

        # 2. 初始化 model
        model = AGCN(settings=self.cfg)
        # model.to(self.device)
        model.init_net_data(
            graph_adj_mat=graph_adjmat
        )

        # 将加载后的数据转换为Tensor
        item_attrs_input = torch.from_numpy(item_attrs_missing).cuda() if self.item_attr_cfg['have'] else None
        user_attrs_input = torch.from_numpy(user_attrs_missing).cuda() if self.user_attr_cfg['have'] else None

        # 3.开始测试
        all_best_pth = np.load(self.cfg['train_folder_path'] + "/all_best_pth.npy", allow_pickle=True).tolist()
        output_cont = []
        with torch.no_grad():
            for k, pth_file_name in enumerate(all_best_pth):
                model.load_state_dict(torch.load(self.train_folder_path + "/pths/" + pth_file_name))
                self.logger.info("loaded " + pth_file_name)
                model(user_attrs_input=user_attrs_input, item_attrs_input=item_attrs_input)  # forward

                user_emb = tensor2npy(model.final_user_emb)
                item_emb = tensor2npy(model.final_item_emb)
                # test set
                perf_info, all_perf = evaluate(
                    user_emb, item_emb,
                    loadutil.load_train_U2I(),
                    loadutil.load_test_U2I(), args={'topks': self.cfg.model_cfg['test_topks'], 'cores': 4})

                for i, topk in enumerate(self.cfg.model_cfg['test_topks']):
                    output_cont.append("[%d]@%d: (ndcg=%.4f) (hr=%.4f) (recall=%.4f)" % (
                        k, topk, perf_info[i * 3], perf_info[i * 3 + 1], perf_info[i * 3 + 2])
                    )
                    self.logger.info(output_cont[-1])
                np.save(self.tmpout_folder_path + "/all_metrics-{}.npy".format(k))
                # dict_hr, dict_ndcg = Evaluate.get_hr_ndcg_for_test(
                #     model.final_user_emb, model.final_item_emb,
                #     dict_test_data, complete_data, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
                #
                # for key in dict_hr:
                #     output_cont.append("[%d]@%d: (hr=%.4f) (ndcg=%.4f)" % (k, key, dict_hr[key], dict_ndcg[key]))
                #     self.logger.info(output_cont[-1])

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

        with open(self.tmpout_folder_path + "/result.json", 'w', encoding='utf-8') as f:
            json.dump(output_cont, f, ensure_ascii=False, indent=2)
