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
        pass
