import torch
from torch.nn import Parameter
import numpy as np
from torch.nn import functional as F


class FM(torch.nn.Module):
    def __init__(self, cfg):
        super(FM, self).__init__()
        self.cfg = cfg
        self.yml_cfg = self.cfg.yml_cfg
        self.model_cfg = self.yml_cfg['FM']
        self.init_net_params()
        self.init_net_data()

    def init_net_params(self):
        self.free_emb_dim = self.model_cfg['free_emb_dim']  # user embedding和item embedding的维度
        self.user_count = self.yml_cfg['user_count']
        self.item_count = self.yml_cfg['item_count']
        self.user_attr_cfg = self.yml_cfg['user_attr']
        self.item_attr_cfg = self.yml_cfg['item_attr']
        self.attr_union_dim = self.model_cfg['attr_union_dim']

    def init_net_data(self):
        # 初始化user和item的free embedding
        self.user_emb = torch.nn.init.normal_(torch.empty(self.user_count, self.free_emb_dim), mean=0, std=0.01)
        self.item_emb = torch.nn.init.normal_(torch.empty(self.item_count, self.free_emb_dim), mean=0, std=0.01)
        self.user_emb = Parameter(self.user_emb)
        self.item_emb = Parameter(self.item_emb)

        # user属性初始化

        if self.user_attr_cfg.get('have', False):
            user_attr_dim = sum(self.user_attr_cfg['attr_dim_list'])
            self.user_attrs_trans_w = Parameter(
                torch.nn.init.normal_(torch.empty(user_attr_dim, self.attr_union_dim), mean=0, std=0.01))
            # 属性推理
            self.user_attrs_infer_w = Parameter(
                torch.nn.init.normal_(torch.empty(self.free_emb_dim + self.attr_union_dim, user_attr_dim), mean=0,
                                      std=0.01)
            )
            self.user_attrs_infer_b = Parameter(torch.zeros(1, user_attr_dim))
        else:
            self.user_attrs_emb = torch.nn.init.normal_(torch.empty(self.user_count, self.attr_union_dim), mean=0,
                                                        std=0.01)
            self.user_attrs_emb = Parameter(self.user_attrs_emb)

        # item属性初始化
        if self.item_attr_cfg.get('have', False):
            item_attr_dim = sum(self.item_attr_cfg['attr_dim_list'])
            self.item_attrs_trans_w = Parameter(
                torch.nn.init.normal_(torch.empty(item_attr_dim, self.attr_union_dim), mean=0, std=0.01))
            # 属性推理
            self.item_attrs_infer_w = Parameter(
                torch.nn.init.normal_(torch.empty(self.free_emb_dim + self.attr_union_dim, item_attr_dim), mean=0,
                                      std=0.01)
            )
            self.item_attrs_infer_b = Parameter(torch.zeros(1, item_attr_dim))
        else:
            self.item_attrs_emb = torch.nn.init.normal_(torch.empty(self.item_count, self.attr_union_dim), mean=0,
                                                        std=0.01)
            self.item_attrs_emb = Parameter(self.item_attrs_emb)

    def forward(self, **kwargs):
        if self.user_attr_cfg.get('have', 'False') is True:
            self.user_attrs_input = kwargs['user_attrs_input']
            self.user_attrs_emb = torch.mm(self.user_attrs_input, self.user_attrs_trans_w)

        if self.item_attr_cfg.get('have', 'False') is True:
            self.item_attrs_input = kwargs['item_attrs_input']
            self.item_attrs_emb = torch.mm(self.item_attrs_input, self.item_attrs_trans_w)

        # 拼接 free embedding 和 attr embedding
        self.final_user_emb = torch.cat((self.user_emb, self.user_attrs_emb), 1)
        self.final_item_emb = torch.cat((self.item_emb, self.item_attrs_emb), 1)
        return self.final_user_emb, self.final_item_emb

    def link_predict(self, user_index, item_index):
        latent_user = self.final_user_emb[user_index]
        latent_item = self.final_item_emb[item_index]
        latent_res = torch.mul(latent_user, latent_item)
        return torch.sigmoid(torch.sum(latent_res, 1, keepdim=True))

    def item_attr_inference(self, item_index):
        items_emb = self.final_item_emb[item_index]
        infer = torch.mm(items_emb, self.item_attrs_infer_w) + self.item_attrs_infer_b
        return torch.sigmoid(infer)

    def user_attr_inference(self, user_index):
        users_emb = self.final_user_emb[user_index]
        infer = torch.mm(users_emb, self.user_attrs_infer_w) + self.user_attrs_infer_b
        return torch.sigmoid(infer)

    def item_attr_infer_loss(self, item_existing_index, item_gt, slice_l=None, slice_r=None):
        assert item_gt.shape[1] in self.item_attr_cfg['attr_dim_list']
        if slice_l is None or slice_r is None:
            slice_l = 0
            slice_r = item_gt.shape[1]
        else:
            assert item_gt.shape[1] == slice_r - slice_l and slice_l < slice_r
        item_infer = self.item_attr_inference(item_existing_index)[:, slice_l:slice_r]
        # loss1 = F.cross_entropy(torch.softmax(item_infer, 1), item_gt)
        item_infer = - torch.log(torch.clip(torch.softmax(item_infer, 1), 1e-10, 1.0))
        return torch.mean(torch.sum(torch.mul(item_infer, item_gt), 1))

    def user_attr_infer_loss(self, user_existing_index, user_gt, slice_l=None, slice_r=None):
        assert user_gt.shape[1] in self.user_attr_cfg['attr_dim_list']
        if slice_l is None or slice_r is None:
            slice_l = 0
            slice_r = user_gt.shape[1]
        else:
            assert user_gt.shape[1] == slice_r - slice_l and slice_l < slice_r
        user_infer = self.user_attr_inference(user_existing_index)[:, slice_l:slice_r]
        user_infer = - torch.log(torch.clip(torch.softmax(user_infer, 1), 1e-10, 1.0))
        return torch.mean(torch.sum(torch.mul(user_infer, user_gt), 1))

    def link_predict_loss(self, user_index, item_index_1, item_index_2):
        lambda1 = self.model_cfg['lambda1']
        lambda2 = self.model_cfg['lambda2']
        ua = self.final_user_emb[user_index]
        vi = self.final_item_emb[item_index_1]
        vj = self.final_item_emb[item_index_2]
        Rai = torch.sum(torch.mul(ua, vi), 1)
        Raj = torch.sum(torch.mul(ua, vj), 1)
        with torch.no_grad():
            self.auc = torch.mean((Rai > Raj).float())
        bpr_loss = - torch.mean(torch.log(torch.clip(torch.sigmoid(Rai - Raj), 1e-10, 1.0)))
        regulation = lambda1 * torch.mean(torch.square(ua)) + lambda2 * torch.mean(torch.square(vi) + torch.square(vj))
        return bpr_loss + regulation

    def total_loss(self, user_index, item_index_1, item_index_2, user_existing_index_list=None, user_gt_list=None,
                   item_existing_index_list=None, item_gt_list=None):
        gamma = self.model_cfg['gamma']
        loss1 = self.link_predict_loss(user_index, item_index_1, item_index_2)
        # loss2 = torch.FloatTensor([0.0]).cuda()
        # if self.user_attr_cfg['have']:
        #     assert user_existing_index_list is not None and user_gt_list is not None
        #     assert len(user_existing_index_list) == len(user_gt_list) == len(self.user_attr_cfg['attr_dim_list'])
        #     attr_dim_list = self.user_attr_cfg['attr_dim_list']
        #     for i in range(len(attr_dim_list)):
        #         slice_l = sum(attr_dim_list[0:i])
        #         slice_r = slice_l + attr_dim_list[i]
        #         gt_list = user_gt_list[i]
        #         gt_list = torch.from_numpy(np.asarray(gt_list)).cuda()
        #         loss2 += self.user_attr_infer_loss(user_existing_index_list[i], gt_list, slice_l, slice_r)
        #
        # if self.item_attr_cfg['have']:
        #     assert item_existing_index_list is not None and item_gt_list is not None
        #     assert len(item_existing_index_list) == len(item_gt_list) == len(self.item_attr_cfg['attr_dim_list'])
        #     attr_dim_list = self.item_attr_cfg['attr_dim_list']
        #     for i in range(len(attr_dim_list)):
        #         slice_l = sum(attr_dim_list[0:i])
        #         slice_r = slice_l + attr_dim_list[i]
        #         gt_list = item_gt_list[i]
        #         gt_list = torch.from_numpy(np.asarray(gt_list)).cuda()
        #         loss2 += self.item_attr_infer_loss(item_existing_index_list[i], gt_list, slice_l, slice_r)

        return loss1
