from utils import UnionConfig
from tqdm import tqdm
from core.models.FM import FM
import torch


class FM_Session(object):
    def __init__(self, cfg: UnionConfig, model: FM):
        self.cfg = cfg
        self.model = model

    def inject_static_data(self, **kwargs):
        self.user_attrs_existing_index_list = kwargs['user_attrs_existing_index_list']
        self.item_attrs_existing_index_list = kwargs['item_attrs_existing_index_list']
        self.user_gt_list = kwargs['user_gt_list']
        self.item_gt_list = kwargs['item_gt_list']
        self.user_attrs_input = kwargs['user_attrs_input']
        self.item_attrs_input = kwargs['item_attrs_input']

    def train(self, dataloader, optimizer):
        sum_loss, sum_loss1, sum_loss2, sum_auc = 0.0, 0.0, 0.0, 0.0
        for uij in dataloader:
            u_list = uij[:, 0].type(torch.long).cuda()
            i_list = uij[:, 1].type(torch.long).cuda()
            j_list = uij[:, 2].type(torch.long).cuda()
            optimizer.zero_grad()
            self.model(user_attrs_input=self.user_attrs_input, item_attrs_input=self.item_attrs_input)  # forward
            loss = self.model.total_loss(
                user_index=u_list, item_index_1=i_list, item_index_2=j_list,
                user_existing_index_list=self.user_attrs_existing_index_list, user_gt_list=self.user_gt_list,
                item_existing_index_list=self.item_attrs_existing_index_list, item_gt_list=self.item_gt_list
            )
            loss.backward()
            optimizer.step()
            # 统计数据
            sum_auc += self.model.auc.item()
            sum_loss += loss.item()
        mean_auc = round(sum_auc / len(dataloader), 4)
        mean_loss = round(sum_loss / len(dataloader), 4)
        return mean_loss,  mean_auc
