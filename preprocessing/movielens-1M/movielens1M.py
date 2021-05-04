import os
import numpy as np
import random
from collections import defaultdict


class Generate:
    name = "movielens1M"

    def __init__(self, **kwargs):
        self._save_folder = os.path.realpath(kwargs.get('save_folder') or "./handled")
        self._raw_folder = os.path.realpath(kwargs.get('raw_folder') or "./raw")
        if not os.path.exists(self._save_folder):
            os.makedirs(self._save_folder)
        assert os.path.exists(self._raw_folder)

        self.user_count = 6040
        self.item_count = 3952

    def gene_train_val_test_data(self, rawfile='ratings.dat', complete_filename="total_U2I.npy",
                                 train_filename="train_U2I.npy",
                                 val_filename="val_U2I.npy", test_filename="test_U2I.npy", split_ratio=[8, 1, 1],
                                 filter_ratings=5, rating_threshold=5):
        """
            划分数据集为训练集、验证集、测试集 8:1:1
            由于该数据集较为稠密，为了提高稀疏性，只将评分为5的电影作为该用户喜欢的电影, 其总数目作为ratings_num
           rawfile format: UserID::MovieID::Rating::Timestamp
        """
        raw_filepath = self._raw_folder + "/" + rawfile
        complete_filepath = self._save_folder + "/" + complete_filename
        train_filepath = self._save_folder + "/" + train_filename
        val_filepath = self._save_folder + "/" + val_filename
        test_filepath = self._save_folder + "/" + test_filename

        raw_U2I = defaultdict(set)
        with open(raw_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                ele_list = line.split("::")[0:-1]
                ele_list = [int(ele) for ele in ele_list]
                if ele_list[2] < rating_threshold:
                    continue
                ele_list[0] -= 1
                ele_list[1] -= 1
                raw_U2I[ele_list[0]].add(ele_list[1])

        # filter ratings less than filter_num
        total_U2I = {k: list(v) for k, v in raw_U2I.items() if len(v) >= filter_ratings}

        # gene total UI
        total_UI = []
        for uid in total_U2I.keys():
            for iid in total_U2I[uid]:
                total_UI.append([uid, iid])

        # info statistic
        user_count = self.user_count
        item_count = self.item_count
        feedback_count = total_UI.__len__()
        print("user_count: {}\nitem_count: {}\nfeedback_count: {}".format(
            user_count, item_count, feedback_count
        ))

        # split dataset
        train_U2I, val_U2I, test_U2I = dict(), dict(), dict()
        train_UI, val_UI, test_UI = [], [], []
        # 生成训练、验证、测试数据(8:1:1)
        for u, value in total_U2I.items():
            random.shuffle(value)
            count_train = max(1, int(len(value) * (split_ratio[0] / sum(split_ratio))))
            count_val = int((len(value) - count_train) * (split_ratio[1] / sum(split_ratio[1::])))
            train_U2I[u] = value[0:count_train]
            val_U2I[u] = value[count_train:count_train + count_val]
            test_U2I[u] = value[count_train + count_val::]

            for item_id in train_U2I[u]:
                train_UI.append([u, item_id])
            for item_id in test_U2I[u]:
                test_UI.append([u, item_id])
            for item_id in val_U2I[u]:
                val_UI.append([u, item_id])

        np.save(complete_filepath, total_U2I)
        np.save(train_filepath, train_U2I)
        np.save(val_filepath, val_U2I)
        np.save(test_filepath, test_U2I)

    def gene_graph_index_value(self, raw_filename="train_data.npy", index_filename="A_indexs.npy",
                               value_filename="A_values.npy"):
        """
            生成二分图所需的索引和值
        :annotation
        - UserIDs range between 1 and 6040
        - MovieIDs range between 1 and 3952
        - Ratings are made on a 5-star scale (whole-star ratings only)
        - Timestamp is represented in seconds since the epoch as returned by time(2)
        - Each user has at least 20 ratings
        """
        index_filepath = self._save_folder + "/" + index_filename
        value_filepath = self._save_folder + "/" + value_filename
        raw_filepath = self._save_folder + "/" + raw_filename
        train_user_item = np.load(raw_filepath, allow_pickle=True).tolist()
        train_item_user = defaultdict(set)
        # 构造train_item_user
        for uid, item_list in train_user_item.items():
            for item_id in item_list:
                train_item_user[item_id].add(uid)

        # 构造A_indexs和A_values
        A_indexs, A_values = [], []
        for uid, item_list in train_user_item.items():
            len_u = len(item_list)
            for item_id in item_list:
                len_v = len(train_item_user[item_id])
                A_indexs.append([uid, item_id + self.user_count])
                A_values.append(1 / len_u)
                A_indexs.append([item_id + self.user_count, uid])
                A_values.append(1 / len_v)
        np.save(index_filepath, A_indexs)
        np.save(value_filepath, A_values)

    def gene_user_attrs(self, raw_filename="users.dat", complete_user_attr_filename="complete_user_attr.npy",
                        existing_user_attr_index_filename="existing_user_attr_index.npy",
                        missing_user_attr_filename="missing_user_attr.npy", del_percent=0.9
                        ):
        """
            生成用户属性编码矩阵，并删除90%属性（用平均值填充）
            rawfile format: UserID::Gender::Age::Occupation::Zip-code
        :param raw_filename: 原始数据集中的用户属性文件路径
        :param complete_user_attr_filename: 完整的用户属性编码
        :param existing_user_attr_index_filename: 未缺失属性的用户id
        :param missing_user_attr_filename: 删除90%属性后的所有用户属性
        :param del_percent:
        :return:
        """
        complete_user_attr_filepath = self._save_folder + "/" + complete_user_attr_filename
        existing_user_attr_index_filepath = self._save_folder + "/" + existing_user_attr_index_filename
        missing_user_attr_filepath = self._save_folder + "/" + missing_user_attr_filename
        raw_filepath = self._raw_folder + "/" + raw_filename

        # 原始数据统计
        gender_dict = {}
        age_dict = {}
        occupation_dict = {}
        with open(raw_filepath, 'r', encoding='utf-8') as fr:
            for line in fr:
                items_list = line.split('::')
                assert items_list[1].strip().__len__() > 0
                assert items_list[2].strip().__len__() > 0
                assert items_list[3].strip().__len__() > 0
                if items_list[1] not in gender_dict.keys():
                    gender_dict[items_list[1]] = len(gender_dict)
                if items_list[2] not in age_dict.keys():
                    age_dict[items_list[2]] = len(age_dict)
                if items_list[3] not in occupation_dict.keys():
                    occupation_dict[items_list[3]] = len(occupation_dict)

        # 生成完整用户属性
        gender_len, age_len, occupation_len = len(gender_dict), len(age_dict), len(occupation_dict)
        complete_user_attr = np.zeros((self.user_count, gender_len + age_len + occupation_len), dtype=np.float64)
        with open(raw_filepath, 'r', encoding='utf-8') as fr:
            for line in fr:
                items_list = line.split('::')
                uid = int(items_list[0]) - 1
                complete_user_attr[uid][gender_dict[items_list[1]]] = 1
                complete_user_attr[uid][age_dict[items_list[2]] + gender_len] = 1
                complete_user_attr[uid][occupation_dict[items_list[3]] + gender_len + age_len] = 1
        np.save(complete_user_attr_filepath, complete_user_attr)

        # 随机删除90%用户的属性，并以均值填充
        missing_user_attr = complete_user_attr.copy()
        itemset = set(range(self.user_count))

        len_list = [gender_len, age_len, occupation_len]
        existing_user_attr_index_dict = dict()
        existing_user_attr_index_dict['attr_dim_list'] = len_list
        existing_user_attr_index_dict['existing_index_list'] = list()
        for i in range(len_list.__len__()):
            # 针对不同属性，采取不同随机删除处理
            start_ind = sum(len_list[0:i])
            stop_ind = sum(len_list[0:i+1])
            existing_user_attr_index = random.sample(itemset, int(self.user_count * (1 - del_percent)))
            missing_user_attr_index = list(itemset - set(existing_user_attr_index))
            padding_attr_vector = np.mean(complete_user_attr[existing_user_attr_index, start_ind:stop_ind], axis=0)
            missing_user_attr[missing_user_attr_index, start_ind:stop_ind] = np.tile(padding_attr_vector, (len(missing_user_attr_index), 1))
            existing_user_attr_index_dict['existing_index_list'].append(existing_user_attr_index)
        np.save(missing_user_attr_filepath, missing_user_attr)
        np.save(existing_user_attr_index_filepath, existing_user_attr_index_dict)


if __name__ == '__main__':
    gene = Generate()
    gene.gene_train_val_test_data()
    # gene.gene_graph_index_value()
    gene.gene_user_attrs()
    # data = np.load('./handled/existing_user_attr_index.npy', allow_pickle=True).tolist()
    # data2 = np.load('./handled/missing_user_attr.npy', allow_pickle=True).tolist()
    # print(data['attr_dim_list'])
    # ind = set(data['existing_index_list'][2]) & set(data['existing_index_list'][1]) & set(data['existing_index_list'][0])
    # ind = list(ind)[1]
    # print(data2[ind])
