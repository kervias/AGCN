import torch
from utils import UnionConfig


class LP(object):
    def __init__(self):
        pass
    # def __init__(self, cfg: UnionConfig, is_user: bool):
    #     self.cfg = cfg
    #     self.yml_cfg = self.cfg.yml_cfg
    #     self.model_cfg = self.yml_cfg['LP']
    #
    #     self.is_user = is_user
    #     if self.is_user is True:
    #         assert self.user_attr_cfg['have'] is True
    #         self.user_attr_cfg = self.yml_cfg['user_attr']
    #         self.user_count = self.yml_cfg['user_count']
    #     else:
    #         assert self.item_attr_cfg['have'] is True
    #         self.item_attr_cfg = self.yml_cfg['item_attr']
    #         self.item_count = self.yml_cfg['item_count']

    def forward(self, attr_input, transform_mat):
        attr_pd = torch.sparse.mm(transform_mat, attr_input)
        return attr_pd

    def get_loss(self, attr_pd, attr_gt):
        return torch.sum(torch.square(attr_gt - attr_pd))

