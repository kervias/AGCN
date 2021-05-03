import numpy as np
from utils import DecoratorTimer

import random
import torch


class LoadUtil(object):
    """
        导入数据
    """
    def __init__(self, settings):
        self.settings = settings
        self.total_U2I = None
        self.train_U2I = None
        self.val_U2I = None
        self.test_U2I = None
        self.tensor_graph_adjmat = None
        self.ndarray_item_attrs_complete = None
        self.ndarray_item_attrs_missing = None
        self.ndarray_user_attrs_complete = None
        self.ndarray_user_attrs_missing = None
        self.list_item_attrs_existing_index = None
        self.list_item_attrs_missing_index = None
        self.list_user_attrs_existing_index = None
        self.list_user_attrs_missing_index = None
        self.user_gt_list = None
        self.item_gt_list = None
        self.dict_neg_samples = dict()

        self.model_cfg = self.settings["model_cfg"]
        self._data_folder = self.settings.data_folder_path

    @DecoratorTimer()
    def load_total_U2I(self, filename="total_U2I.npy"):
        if self.total_U2I is None:
            filepath = self._data_folder + "/" + filename
            self.total_U2I = np.load(filepath, allow_pickle=True).tolist()
        return self.total_U2I

    @DecoratorTimer()
    def load_train_U2I(self, filename="train_U2I.npy"):
        if self.train_U2I is None:
            filepath = self._data_folder + "/" + filename
            self.train_U2I = np.load(filepath, allow_pickle=True).tolist()
        return self.train_U2I

    @DecoratorTimer()
    def load_val_U2I(self, filename="val_U2I.npy"):
        if self.val_U2I is None:
            filepath = self._data_folder + "/" + filename
            self.val_U2I = np.load(filepath, allow_pickle=True).tolist()
        return self.val_U2I

    @DecoratorTimer()
    def load_test_U2I(self, filename="test_U2I.npy"):
        if self.test_U2I is None:
            filepath = self._data_folder + "/" + filename
            self.test_U2I = np.load(filepath, allow_pickle=True).tolist()
        return self.test_U2I

    @DecoratorTimer()
    def load_graph_adj_mat(self, filenameL=['A_indexs.npy', 'A_values.npy']):
        if self.tensor_graph_adjmat is None:
            filepath1 = self._data_folder + "/" + filenameL[0]
            filepath2 = self._data_folder + "/" + filenameL[1]
            list_graph_adjmat_index = np.load(filepath1).tolist()
            list_graph_adjmat_value = np.load(filepath2).tolist()
            total_count = self.model_cfg['user_count'] + self.model_cfg['item_count']
            self.tensor_graph_adjmat = torch.sparse_coo_tensor(torch.tensor(list_graph_adjmat_index).t(),
                                                               list_graph_adjmat_value, (
                                                                   total_count, total_count))
        return self.tensor_graph_adjmat

    @DecoratorTimer()
    def load_item_attrs_complete(self, filename="complete_item_attr.npy"):
        assert self.model_cfg['item_attr'].get('have', False) is True
        if self.ndarray_item_attrs_complete is None:
            filepath = self._data_folder + "/" + filename
            self.ndarray_item_attrs_complete = np.load(filepath, allow_pickle=True).astype(np.float32)
        return self.ndarray_item_attrs_complete

    @DecoratorTimer()
    def load_item_attrs_missing(self, filename="missing_item_attr.npy"):
        assert self.model_cfg['item_attr'].get('have', False) is True
        if self.ndarray_item_attrs_missing is None:
            filepath = self._data_folder + "/" + filename
            self.ndarray_item_attrs_missing = np.load(filepath, allow_pickle=True).astype(np.float32)
        return self.ndarray_item_attrs_missing

    @DecoratorTimer()
    def load_user_attrs_complete(self, filename="complete_user_attr.npy"):
        assert self.model_cfg['user_attr'].get('have', False) is True
        if self.ndarray_user_attrs_complete is None:
            filepath = self._data_folder + "/" + filename
            self.ndarray_user_attrs_complete = np.load(filepath, allow_pickle=True).astype(np.float32)
        return self.ndarray_user_attrs_complete

    @DecoratorTimer()
    def load_user_attrs_missing(self, filename="missing_user_attr.npy"):
        assert self.model_cfg['user_attr'].get('have', False) is True
        if self.ndarray_user_attrs_missing is None:
            filepath = self._data_folder + "/" + filename
            self.ndarray_user_attrs_missing = np.load(filepath, allow_pickle=True).astype(np.float32)
        return self.ndarray_user_attrs_missing

    @DecoratorTimer()
    def load_item_attrs_existing_index(self, filename="existing_item_attr_index.npy"):
        assert self.model_cfg['item_attr'].get('have', False) is True
        if self.list_item_attrs_existing_index is None:
            filepath = self._data_folder + "/" + filename
            dict_item_attrs_existing_index = np.load(filepath, allow_pickle=True).tolist()
            assert dict_item_attrs_existing_index['attr_dim_list'] == self.model_cfg['item_attr']['attr_dim_list']
            self.list_item_attrs_existing_index = dict_item_attrs_existing_index['existing_index_list']
        return self.list_item_attrs_existing_index

    @DecoratorTimer()
    def load_user_attrs_existing_index(self, filename='existing_user_attr_index.npy'):
        assert self.model_cfg['user_attr'].get('have', False) is True
        if self.list_user_attrs_existing_index is None:
            filepath = self._data_folder + "/" + filename
            dict_user_attrs_existing_index = np.load(filepath, allow_pickle=True).tolist()
            assert dict_user_attrs_existing_index['attr_dim_list'] == self.model_cfg['user_attr']['attr_dim_list']
            self.list_user_attrs_existing_index = dict_user_attrs_existing_index['existing_index_list']
        return self.list_user_attrs_existing_index

    @DecoratorTimer()
    def load_item_attrs_missing_index(self):
        assert self.model_cfg['item_attr'].get('have', False) is True
        if self.list_item_attrs_missing_index is None:
            item_set = set(range(self.model_cfg['item_count']))
            list_item_attrs_existing_index = self.load_item_attrs_existing_index()
            self.list_item_attrs_missing_index = []
            for i in range(self.model_cfg['item_attr']['attr_type_num']):
                self.list_item_attrs_missing_index.append(list(item_set - set(list_item_attrs_existing_index[i])))
        return self.list_item_attrs_missing_index

    @DecoratorTimer()
    def load_user_attrs_missing_index(self):
        assert self.model_cfg['user_attr'].get('have', False) is True
        if self.list_user_attrs_missing_index is None:
            user_set = set(range(self.model_cfg['user_count']))
            list_user_attrs_existing_index = self.load_user_attrs_existing_index()
            self.list_user_attrs_missing_index = []
            for i in range(self.model_cfg['user_attr']['attr_type_num']):
                self.list_user_attrs_missing_index.append(list(user_set - set(list_user_attrs_existing_index[i])))
        return self.list_user_attrs_missing_index

    @DecoratorTimer()
    def load_user_gt_list(self):
        assert self.model_cfg['user_attr'].get('have', False) is True
        if self.user_gt_list is None:
            user_attrs_complete = torch.from_numpy(self.load_user_attrs_complete())
            user_attrs_existing_index_list = self.load_user_attrs_existing_index()
            self.user_gt_list = []
            attr_dim_list = self.model_cfg['user_attr']['attr_dim_list']
            for i in range(len(attr_dim_list)):
                slice_l = sum(attr_dim_list[0:i])
                slice_r = slice_l + attr_dim_list[i]
                self.user_gt_list.append(user_attrs_complete[user_attrs_existing_index_list[i]][:, slice_l:slice_r])
        return self.user_gt_list

    @DecoratorTimer()
    def load_item_gt_list(self):
        assert self.model_cfg['item_attr'].get('have', False) is True
        if self.item_gt_list is None:
            item_attrs_complete = torch.from_numpy(self.load_item_attrs_complete())
            item_attrs_existing_index_list = self.load_item_attrs_existing_index()
            self.item_gt_list = []
            attr_dim_list = self.model_cfg['item_attr']['attr_dim_list']
            for i in range(len(attr_dim_list)):
                slice_l = sum(attr_dim_list[0:i])
                slice_r = slice_l + attr_dim_list[i]
                self.item_gt_list.append(item_attrs_complete[item_attrs_existing_index_list[i]][:, slice_l:slice_r])
        return self.item_gt_list

    @DecoratorTimer()
    def get_bpr_data(self, neg_sample_num):
        item_count = self.model_cfg['item_count']
        triple_list = []
        dict_train_data = self.load_train_U2I()
        dict_complete_data = self.load_total_U2I()
        for uid in dict_train_data.keys():
            for i in dict_train_data[uid]:
                # choice_list = list(set(range(0, item_count)) - set(dict_complete_data[uid]))
                # random.shuffle(choice_list)
                # for j in choice_list[0:neg_sample_num]:
                #     triple_list.append([uid, i, j])
                for _ in range(neg_sample_num):
                    j = random.randint(0, item_count - 1)
                    while j in dict_complete_data[uid]:
                        j = random.randint(0, item_count - 1)
                    triple_list.append([uid, i, j])
        return triple_list

    @DecoratorTimer()
    def get_neg_samples(self, force_regenerate=False, neg_sample_num=1000):
        """
            验证集需要，为每个user生成1000个负样本
        :param force_regenerate: 是否强制重新生成数据
        :return: dict<uid:list<item_id>>
        """
        item_count = self.model_cfg['item_count']
        dict_complete_data = self.load_total_U2I()
        all_item_id_set = set(range(item_count))
        if neg_sample_num not in self.dict_neg_samples.keys() or force_regenerate is True:
            self.dict_neg_samples[neg_sample_num] = dict()
            for uid in dict_complete_data.keys():
                self.dict_neg_samples[neg_sample_num][uid] = random.sample(all_item_id_set - set(dict_complete_data[uid]), neg_sample_num)
        return self.dict_neg_samples[neg_sample_num]
