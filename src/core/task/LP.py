from core.data.loadUtil import LoadUtil
import torch
import numpy as np


class LP_Manager(object):
    """
        Label Propagation
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_name = self.cfg.dataset_name
        self.tmpout_folder_path = self.cfg['tmpout_folder_path']
        self.logger = self.cfg.logger
        self.model_cfg = self.cfg.model_cfg
        self.lp_cfg = self.model_cfg['LP']
        self.item_attr_cfg = self.model_cfg['item_attr']
        self.user_attr_cfg = self.model_cfg['user_attr']
        self.user_count = self.model_cfg['user_count']
        self.item_count = self.model_cfg['item_count']
        self.load_util = LoadUtil(settings=self.cfg)

    def start(self):
        # 1. 加载数据
        train_U2I = self.load_util.load_train_U2I()

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

