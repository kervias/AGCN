from core.models.AGCN import AGCN
from tqdm import tqdm
from utils import UnionConfig
import torch


class AISession(object):
    def __init__(self, cfg: UnionConfig, model: AGCN):
        self.cfg = cfg
        self.model = model

    def inject_static_data(self, **kwargs):
        self.user_attrs_existing_index_list = kwargs['user_attrs_existing_index_list']
        self.item_attrs_existing_index_list = kwargs['item_attrs_existing_index_list']
        self.user_gt_list = kwargs['user_gt_list']
        self.item_gt_list = kwargs['item_gt_list']

    def train(self, dataloader, optimizer, user_attrs_input, item_attrs_input):
        sum_loss, sum_loss1, sum_loss2, sum_auc = 0.0, 0.0, 0.0, 0.0
        for uij in dataloader:
            u_list = uij[:, 0].type(torch.long).cuda()
            i_list = uij[:, 1].type(torch.long).cuda()
            j_list = uij[:, 2].type(torch.long).cuda()
            optimizer.zero_grad()
            self.model(user_attrs_input=user_attrs_input, item_attrs_input=item_attrs_input)  # forward
            loss, loss1, loss2 = self.model.total_loss(
                user_index=u_list, item_index_1=i_list, item_index_2=j_list,
                user_existing_index_list=self.user_attrs_existing_index_list, user_gt_list=self.user_gt_list,
                item_existing_index_list=self.item_attrs_existing_index_list, item_gt_list=self.item_gt_list
            )
            loss.backward()
            optimizer.step()
            # 统计数据
            sum_auc += self.model.auc.item()
            sum_loss += loss.item()
            sum_loss1 += loss1.item()
            sum_loss2 += loss2.item()
        mean_auc = round(sum_auc / len(dataloader), 4)
        mean_loss = round(sum_loss / len(dataloader), 4)
        mean_loss1 = round(sum_loss1 / len(dataloader), 4)
        mean_loss2 = round(sum_loss2 / len(dataloader), 4)
        return mean_loss, mean_loss1, mean_loss2, mean_auc
