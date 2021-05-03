import numpy as np
import torch


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, UI, U2I, item_count):
        self.UI = UI
        self.U2I = U2I
        self.item_count = item_count

    def __getitem__(self, index):
        user = self.UI[index][0]
        pos_item = self.UI[index][1]
        neg_item = np.random.choice(self.item_count)
        while neg_item in self.U2I[user]:
            neg_item = np.random.choice(self.item_count)
        return np.array([user, pos_item, neg_item])

    def __len__(self):
        return self.UI.shape[0]

    @staticmethod
    def gene_UI_from_U2I(U2I, num_neg_item=1):
        total_num = 0
        for u, items in U2I.items():
            total_num += len(items) * num_neg_item
        UI = np.zeros((total_num, 2), dtype=np.uint64)
        ind = 0
        for u, items in U2I.items():
            temp_u = np.array([u] * len(items) * num_neg_item)
            temp_i = np.array(list(items) * num_neg_item)
            temp = np.vstack((temp_u, temp_i))
            UI[ind:ind + len(items) * num_neg_item, :] = temp.T
            ind += len(items) * num_neg_item
        return UI
