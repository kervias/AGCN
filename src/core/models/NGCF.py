import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatTransform(nn.Module):

    def __init__(self, in_features, out_features, mess_dropout):
        super(FeatTransform, self).__init__()

        self.linear_1 = nn.Linear(in_features, out_features)
        self.linear_2 = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(mess_dropout)
        self.act = nn.LeakyReLU(negative_slope=0.2)

        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)

    def forward(self, mat, node_feat):
        agg_feat = torch.sparse.mm(mat, node_feat)

        part_1 = self.act(self.linear_1(agg_feat))
        part_2 = self.act(self.linear_2(agg_feat * node_feat))
        out = part_1 + part_2
        out = self.dropout(out)
        return out


class NGCF(nn.Module):

    def __init__(self, cfg, adj):
        super(NGCF, self).__init__()
        self.cfg = cfg
        self.yml_cfg = self.cfg.yml_cfg
        self.model_cfg = self.yml_cfg['NGCF']

        self.n_users = self.yml_cfg['user_count']
        self.n_items = self.yml_cfg['item_count']
        self.adj = adj  # norm-Laplace matrix

        self.decay = self.model_cfg['decay']
        self.batch_size = self.model_cfg['batch_size']
        self.decay = self.model_cfg['decay']  # l2-norm coefficient
        self.layers = self.model_cfg['layers']
        self.gcn_layer_num = self.model_cfg['gcn_layer_num']
        self.emb_size = self.model_cfg['emb_size']
        self.node_dropout = self.model_cfg['node_dropout']
        self.mess_dropout = self.model_cfg['mess_dropout']

        self.dims = self.layers

        # embedding 层
        self.embeddings = nn.Embedding(self.n_users + self.n_items, self.emb_size)

        # node dropout layer
        self.dropout = nn.Dropout(self.node_dropout)

        # feature transform layer contains the message dropout
        self.layer_stack = nn.ModuleList([
            FeatTransform(self.dims[i], self.dims[i + 1], self.mess_dropout[i])
            for i in range(self.gcn_layer_num)
        ])

        # initial
        nn.init.xavier_uniform_(self.embeddings.weight)

    def dropout_sparse(self, adj):
        i = adj._indices()
        v = adj._values()
        v = self.dropout(v)
        drop_adj = torch.sparse_coo_tensor(i, v, adj.shape).to(adj.device)
        return drop_adj

    def forward(self, adj):
        node_feat = self.embeddings.weight  # [user, item] embedding矩阵
        mat = self.dropout_sparse(adj)  # node dropout 后的拉普拉斯矩阵
        all_embeddings = [node_feat]
        for feat_trans in self.layer_stack:
            node_feat = feat_trans(mat, node_feat)
            all_embeddings += [F.normalize(node_feat)]
        return all_embeddings

    def fusion(self, embeddings):
        embeddings = torch.cat(embeddings, dim=1)
        return embeddings

    def split_emb(self, embeddings):
        user_emb, item_emb = torch.split(embeddings, [self.n_users, self.n_items])
        return user_emb, item_emb

    def propagate(self):
        all_emb = self.forward(self.adj)
        f_emb = self.fusion(all_emb)
        user_emb, item_emb = self.split_emb(f_emb)
        return user_emb, item_emb

    def get_embedding(self, user_emb, item_emb, users, pos_items, neg_items):
        u_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]
        return u_emb, pos_emb, neg_emb

    def bpr_loss(self, user_emb, pos_emb, neg_emb):
        """
        bpr loss function
        """
        pos_scores = torch.sum(user_emb * pos_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_emb, dim=1)

        criterion = nn.LogSigmoid()
        mf_loss = criterion(pos_scores - neg_scores)
        mf_loss = -1.0 * torch.mean(mf_loss)

        regularizer = 0.5 * (torch.norm(user_emb) ** 2 + torch.norm(pos_emb) ** 2 + torch.norm(neg_emb) ** 2)
        emb_loss = self.decay * regularizer / self.batch_size

        loss = mf_loss + emb_loss
        return loss, mf_loss, emb_loss

