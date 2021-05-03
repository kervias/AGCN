from core.models.AGCN import AGCN
from core.data.loadUtil import LoadUtil
from core.data.dataset import TrainDataset
from core.evaluate.evaluate import Evaluate
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
from tqdm import tqdm
import math

class IR_Test(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def test(self):
        model_cfg = self.settings["model_cfg"][self.dataset_name]
        # 1. 加载数据
        data_util = LoadUtil(settings=self.settings)
        dict_test_data = data_util.load_test_data()
        complete_data = data_util.load_complete_data()
        item_attrs_complete = data_util.load_item_attrs_complete() if model_cfg['item_attr']['have'] else None
        item_attrs_missing = data_util.load_item_attrs_missing() if model_cfg['item_attr']['have'] else None
        user_attrs_complete = data_util.load_user_attrs_complete() if model_cfg['user_attr']['have'] else None
        user_attrs_missing = data_util.load_user_attrs_missing() if model_cfg['user_attr']['have'] else None
        item_attrs_missing_index_list = data_util.load_item_attrs_missing_index() if model_cfg['item_attr'][
                                                                                         'have'] is True else []
        user_attrs_missing_index_list = data_util.load_user_attrs_missing_index() if model_cfg['user_attr'][
                                                                                         'have'] is True else []
        # user_gt_list = data_util.load_user_gt_list() if model_cfg['user_attr']['have'] is True else None
        # item_gt_list = data_util.load_item_gt_list() if model_cfg['item_attr']['have'] is True else None
        graph_adjmat = data_util.load_graph_adj_mat()

        # 2. 初始化 model
        model = AGCN(settings=self.settings)
        # model.to(self.device)
        model.init_net_data(
            graph_adj_mat=graph_adjmat
        )

        # 将加载后的数据转换为Tensor
        item_attrs_complete = torch.from_numpy(item_attrs_complete) if model_cfg['item_attr']['have'] else None
        item_attrs_missing = torch.from_numpy(item_attrs_missing) if model_cfg['item_attr']['have'] else None
        user_attrs_complete = torch.from_numpy(user_attrs_complete) if model_cfg['user_attr']['have'] else None
        user_attrs_missing = torch.from_numpy(user_attrs_missing) if model_cfg['user_attr']['have'] else None

        item_attrs_input = item_attrs_missing if model_cfg['item_attr']['have'] else None
        user_attrs_input = user_attrs_missing if model_cfg['user_attr']['have'] else None

        if self.settings['DEVICE'] == 'cuda':
            model.to(self.device)
            item_attrs_input = item_attrs_input.cuda() if item_attrs_input is not None else None
            user_attrs_input = user_attrs_input.cuda() if user_attrs_input is not None else None

        # 3.开始测试
        all_best_pth = np.load(self.settings['train_folder_path'] + "/" + "all_best_pth.npy",
                               allow_pickle=True).tolist()
        output_cont = []
        with torch.no_grad():
            for k, pth_file_name in enumerate(all_best_pth):
                model.load_state_dict(torch.load(self.settings['train_folder_path'] + "/" + pth_file_name))
                self.logger.info("loaded " + pth_file_name)
                model(user_attrs_input=user_attrs_input, item_attrs_input=item_attrs_input)  # forward
                dict_hr, dict_ndcg = Evaluate.get_hr_ndcg_for_test(
                    model.final_user_emb, model.final_item_emb,
                    dict_test_data, complete_data, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

                for key in dict_hr:
                    output_cont.append("[%d]@%d: (hr=%.4f) (ndcg=%.4f)" % (k, key, dict_hr[key], dict_ndcg[key]))
                    self.logger.info(output_cont[-1])

                # 更新item推理属性
                if model_cfg['item_attr'].get('have', False) is True:
                    attr_dim_list = model_cfg['item_attr']['attr_dim_list']
                    for i in range(len(attr_dim_list)):
                        slice_l = sum(attr_dim_list[0:i])
                        slice_r = slice_l + attr_dim_list[i]
                        item_attrs_infer = model.item_attr_inference(item_attrs_missing_index_list[i])[:,
                                           slice_l:slice_r]
                        item_attrs_input[item_attrs_missing_index_list[i], slice_l:slice_r] = item_attrs_infer
                # 更新user推理属性
                if model_cfg['user_attr'].get('have', False) is True:
                    attr_dim_list = model_cfg['user_attr']['attr_dim_list']
                    for i in range(len(attr_dim_list)):
                        slice_l = sum(attr_dim_list[0:i])
                        slice_r = slice_l + attr_dim_list[i]
                        user_attrs_infer = model.user_attr_inference(user_attrs_missing_index_list[i])[:,
                                           slice_l:slice_r]
                        user_attrs_input[user_attrs_missing_index_list[i], slice_l:slice_r] = user_attrs_infer

        with open(self.output_folder_path + "/result.json", 'w', encoding='utf-8') as f:
            json.dump(output_cont, f, ensure_ascii=False, indent=2)