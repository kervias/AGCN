import torch
from core.models.BPRMF import BPRMF


class BPRMF_Session(object):
    def __init__(self, model: BPRMF):
        self.model = model

    def auc(self, user_emb, pos_emb, neg_emb):
        x = user_emb * (pos_emb - neg_emb)
        x = torch.sum(x, dim=1)
        x = (x > 0).float()
        return torch.mean(x)

    def train(self, train_loader, optimizer):
        self.model.train()
        all_loss, all_mf_loss, all_emb_loss = 0.0, 0.0, 0.0
        for uij in train_loader:
            u = uij[:, 0].type(torch.long).cuda()
            i = uij[:, 1].type(torch.long).cuda()
            j = uij[:, 2].type(torch.long).cuda()

            optimizer.zero_grad()
            user_emb, item_emb = self.model.propagate()
            user_emb, pos_emb, neg_emb = self.model.get_embedding(user_emb, item_emb, u, i, j)
            bpr_loss, mf_loss, reg_loss = self.model.bpr_loss(user_emb, pos_emb, neg_emb)
            bpr_loss.backward()
            optimizer.step()

            all_loss += bpr_loss.item()
            all_mf_loss += mf_loss.item()
            all_emb_loss += reg_loss.item()

        mean_loss = all_loss / len(train_loader)
        mean_mf_loss = all_mf_loss / len(train_loader)
        mean_emb_loss = all_emb_loss / len(train_loader)

        return mean_loss, mean_mf_loss, mean_emb_loss
