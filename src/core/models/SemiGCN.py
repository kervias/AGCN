import torch
from torch.nn import Parameter
from torch.nn import functional as F
from utils import UnionConfig
import numpy as np

class PropagationLayer(torch.nn.Module):
    """
        AXW
    """

    def __init__(self, in_features: int, out_features: int, bias=True):
        super(PropagationLayer, self).__init__()
        self.linear = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        torch.nn.init.normal_(self.linear.weight, mean=0, std=0.01)

    def forward(self, A: torch.sparse_coo, X: torch.Tensor):
        return self.linear(torch.sparse.mm(A, X))


class FeatTransform(torch.nn.Module):
    """
        Trasnform Feat
    """

    def __init__(self, in_features: int, out_features: int, bias=True):
        super(FeatTransform, self).__init__()
        self.linear = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        torch.nn.init.normal_(self.linear.weight, mean=0, std=0.01)

    def forward(self, mat):
        return self.linear(mat)


class SemiGCN(torch.nn.Module):
    def __init__(self, cfg: UnionConfig):
        super(SemiGCN, self).__init__()
        self.cfg = cfg
        self.yml_cfg = self.cfg.yml_cfg
        self.model_cfg = self.yml_cfg['Semi-GCN']

        self.user_attr_cfg = self.yml_cfg['user_attr']
        self.user_count = self.yml_cfg['user_count']
        self.item_attr_cfg = self.yml_cfg['item_attr']
        self.item_count = self.yml_cfg['item_count']
        self.init_net_params()

    def init_net_params(self):
        self.attr_union_dim = self.model_cfg['attr_union_dim']
        self.gcn_layer = self.model_cfg['gcn_layer']
        self.layer_dim_list = self.model_cfg['layer_dim_list']
        assert self.user_attr_cfg['have'] or self.item_attr_cfg['have']

    def init_net_data(self, **kwargs):
        # user属性初始化
        if self.user_attr_cfg['have'] is True:
            user_attr_dim = sum(self.user_attr_cfg['attr_dim_list'])
            self.user_attrs_trans_w = FeatTransform(user_attr_dim, self.attr_union_dim, bias=False)
            self.user_attrs_infer = FeatTransform(self.attr_union_dim, user_attr_dim, bias=True)
        else:
            self.user_attrs_emb = torch.nn.init.normal_(torch.empty(self.user_count, self.attr_union_dim), mean=0,
                                                        std=0.01)
            self.user_attrs_emb = Parameter(self.user_attrs_emb)

        # item属性初始化
        if self.item_attr_cfg['have'] is True:
            item_attr_dim = sum(self.item_attr_cfg['attr_dim_list'])
            self.item_attrs_trans_w = FeatTransform(item_attr_dim, self.attr_union_dim, bias=False)
            self.item_attrs_infer = FeatTransform(self.attr_union_dim, item_attr_dim, bias=True)
        else:
            #self.item_attrs_emb = torch.nn.Embedding(self.item_count, self.attr_union_dim)
            self.item_attrs_emb = torch.nn.init.normal_(torch.empty(self.item_count, self.attr_union_dim), mean=0,
                                                        std=0.01)
            self.item_attrs_emb = Parameter(self.item_attrs_emb)

        # 初始化传播层
        self.propagation_layers = torch.nn.ModuleList([
            PropagationLayer(self.layer_dim_list[i], self.layer_dim_list[i + 1], bias=False)
            for i in range(self.gcn_layer)
        ])

        # 初始化二分图
        self.graph_adj_mat = kwargs['graph_adj_mat']

    def forward(self, **kwargs):
        if self.user_attr_cfg['have'] is True:
            user_attrs_input = kwargs['user_attrs_input']
            self.user_attrs_emb = self.user_attrs_trans_w(user_attrs_input)

        if self.item_attr_cfg['have'] is True:
            item_attrs_input = kwargs['item_attrs_input']
            self.item_attrs_emb = self.item_attrs_trans_w(item_attrs_input)

        attr_emb = torch.cat((self.user_attrs_emb, self.item_attrs_emb), dim=0)
        for layer_ind in range(self.gcn_layer):
            attr_emb = self.propagation_layers[layer_ind](self.graph_adj_mat, attr_emb)
            if layer_ind != self.gcn_layer - 1:
                attr_emb = F.relu(attr_emb)

        self.final_user_emb, self.final_item_emb = torch.split(attr_emb, [self.user_count, self.item_count], dim=0)
        return self.final_user_emb, self.final_item_emb

    def infer_user_attr(self, user_emb):
        attr_dim_list = self.user_attr_cfg['attr_dim_list']
        emb = self.user_attrs_infer(user_emb)
        attr_pd_list = []
        for i in range(len(attr_dim_list)):
            slice_l = sum(attr_dim_list[0:i])
            slice_r = slice_l + attr_dim_list[i]
            tmp_emb = emb[:, slice_l:slice_r]
            attr_pd_list.append(F.softmax(tmp_emb, dim=1))
        return torch.cat(attr_pd_list, dim=1)

    def infer_item_attr(self, item_emb):
        attr_dim_list = self.item_attr_cfg['attr_dim_list']
        emb = self.item_attrs_infer(item_emb)
        attr_pd_list = []
        for i in range(len(attr_dim_list)):
            slice_l = sum(attr_dim_list[0:i])
            slice_r = slice_l + attr_dim_list[i]
            tmp_emb = emb[:, slice_l:slice_r]
            attr_pd_list.append(F.softmax(tmp_emb, dim=1))
        return torch.cat(attr_pd_list, dim=1)

    def user_attr_infer_loss(self, user_existing_index, user_gt, slice_l=None, slice_r=None):
        assert user_gt.shape[1] in self.user_attr_cfg['attr_dim_list']
        if slice_l is None or slice_r is None:
            slice_l = 0
            slice_r = user_gt.shape[1]
        else:
            assert user_gt.shape[1] == slice_r - slice_l and slice_l < slice_r
        user_infer = self.infer_user_attr(self.final_user_emb[user_existing_index])[:, slice_l:slice_r]
        user_infer = - torch.log(torch.clip(torch.softmax(user_infer, 1), 1e-10, 1.0))
        return torch.mean(torch.sum(torch.mul(user_infer, user_gt), 1))

    def item_attr_infer_loss(self, item_existing_index, item_gt, slice_l=None, slice_r=None):
        assert item_gt.shape[1] in self.item_attr_cfg['attr_dim_list']
        if slice_l is None or slice_r is None:
            slice_l = 0
            slice_r = item_gt.shape[1]
        else:
            assert item_gt.shape[1] == slice_r - slice_l and slice_l < slice_r
        item_infer = self.infer_item_attr(self.final_item_emb[item_existing_index])[:, slice_l:slice_r]
        item_infer = - torch.log(torch.clip(torch.softmax(item_infer, 1), 1e-10, 1.0))
        return torch.mean(torch.sum(torch.mul(item_infer, item_gt), 1))

    def get_infer_loss(self, user_existing_index_list=None, user_gt_list=None,
                   item_existing_index_list=None, item_gt_list=None):

        loss = torch.FloatTensor([0.0]).cuda()
        if self.user_attr_cfg['have']:
            assert user_existing_index_list is not None and user_gt_list is not None
            assert len(user_existing_index_list) == len(user_gt_list) == len(self.user_attr_cfg['attr_dim_list'])
            attr_dim_list = self.user_attr_cfg['attr_dim_list']
            for i in range(len(attr_dim_list)):
                slice_l = sum(attr_dim_list[0:i])
                slice_r = slice_l + attr_dim_list[i]
                gt_list = user_gt_list[i]
                gt_list = torch.from_numpy(np.asarray(gt_list)).cuda()
                loss += self.user_attr_infer_loss(user_existing_index_list[i], gt_list, slice_l, slice_r)

        if self.item_attr_cfg['have']:
            assert item_existing_index_list is not None and item_gt_list is not None
            assert len(item_existing_index_list) == len(item_gt_list) == len(self.item_attr_cfg['attr_dim_list'])
            attr_dim_list = self.item_attr_cfg['attr_dim_list']
            for i in range(len(attr_dim_list)):
                slice_l = sum(attr_dim_list[0:i])
                slice_r = slice_l + attr_dim_list[i]
                gt_list = item_gt_list[i]
                gt_list = torch.from_numpy(np.asarray(gt_list)).cuda()
                loss += self.item_attr_infer_loss(item_existing_index_list[i], gt_list, slice_l, slice_r)
        return loss
