import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.utils as utils
import scipy.sparse as sp
from models.Dual_GCN import GCN, Dual_GCN
from utils import accuracy, sparse_mx_to_torch_sparse_tensor
from sklearn.mixture import GaussianMixture
from noisyUtils import plot2distribution

CE = nn.CrossEntropyLoss(reduction='none')
class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

conf_penalty = NegEntropy()

def kl_loss_compute(pred, soft_targets, reduce=True, tempature=1):
    pred = pred / tempature
    soft_targets = soft_targets / tempature
    kl = F.kl_div(F.log_softmax(pred, dim=1), F.softmax(soft_targets, dim=1), reduction='none')
    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)

def gmm_eval(log_pred, train_label, args, epoch ): 

    losses = CE(log_pred,train_label) 
    # if epoch in [20, 50, 100, 150]:
    #     clean_dist = losses[mask].cpu().numpy()
    #     noisy_dist = losses[~mask].cpu().numpy()
    #     plot2distribution(clean_dist, noisy_dist, epoch, num_scales=50, width=1.8)

    losses = (losses-losses.min())/(losses.max()-losses.min())    

    losses = losses.reshape(-1,1)

    # 参数设置默认值
    # tol : float, defaults to 1e-3.
    # The convergence threshold. EM iterations will stop when the lower bound average gain is below this threshold.
    # reg_covar : float, defaults to 0

    gmm = GaussianMixture(n_components=2,max_iter=20,tol=1e-2,reg_covar=5e-4)
    losses = losses.cpu()
    gmm.fit(losses)
    prob = gmm.predict_proba(losses) 
    prob = prob[:,gmm.means_.argmin()] 
    return prob

def remove_tensor_by_prefix(tensor, element_to_remove):
    tensorT = tensor.T
    # 去掉候选边中以prefixes中元素开头和结尾的edge
    # 将前缀列表转换为张量
    # 获取开头元素
    # tensor=torch.tensor([[1,3],[1,4],[2,3],[2,4],[3,1],[4,2]])
    # prefixes=torch.tensor([4,5])
    start_elements = tensorT[:, 0]
    end_elements = tensorT[:, 1]
    # # 检查开头元素是否在前缀列表中
    mask = ~torch.isin(start_elements, element_to_remove) & ~torch.isin(end_elements, element_to_remove)
    
    # # 根据掩码筛选行
    filtered_tensor = tensorT[mask]
    # breakpoint()
    filtered_tensor = filtered_tensor.T
    return filtered_tensor

class LabeledDividedLoss(nn.Module):
    def __init__(self, args, device):
        super(LabeledDividedLoss, self).__init__()
        self.device = device
        self.args = args
        self.epochs = args.epochs
        self.decay_w = args.decay_w

    def forward(self, y_1, y_2, log_pred,log_pred_1, labels, idx_train_clean, idx_train_noisy, co_lambda=0.1, epoch=-1): 
        y_clean = log_pred[idx_train_clean]
        y_clean_1 = log_pred_1[idx_train_clean]
        t_clean = labels[idx_train_clean]
        loss_pick_1 = F.cross_entropy(y_clean, t_clean, reduction='none')
        loss_pick_2 = F.cross_entropy(y_clean_1, t_clean, reduction='none')
        loss_pick = loss_pick_1  + loss_pick_2
       
        # # loss_clean
        # breakpoint()
        loss_clean = torch.sum(loss_pick)/y_1.shape[0]

        # loss_dc
        ind_update_1 = idx_train_noisy
        p_1 = F.softmax(log_pred,dim=-1)
        p_2 = F.softmax(log_pred_1,dim=-1)
        # pred0.max(dim=1)[1] [1, n]:每个样本的类别； pred0.max(dim=1)[1][unfilter_idx]: unfilter_idx样本对应的类别
        filter_condition = ((log_pred.max(dim=1)[1][ind_update_1] != labels[ind_update_1]) &
                            (log_pred.max(dim=1)[1][ind_update_1] == log_pred_1.max(dim=1)[1][ind_update_1]) &
                            (p_1.max(dim=1)[0][ind_update_1] * p_2.max(dim=1)[0][ind_update_1]  > self.args.th**2 ))
        # breakpoint()
        dc_idx = ind_update_1[filter_condition]

        # TODO 添加自适应权重（根据节点标签分布）
        loss_dc = (F.cross_entropy(log_pred[dc_idx],log_pred.max(dim=1)[1][dc_idx], reduction='none')+ \
                                   F.cross_entropy(log_pred_1[dc_idx], log_pred_1.max(dim=1)[1][dc_idx], reduction='none'))
        loss_dc = loss_dc.sum()/y_1.shape[0]

        #  noisy node
        # train - clean - 2个net预测一致且大于阈值的 = 噪声nodes(可能包含干净nodes 因此含有一定的信息) 添加noisy loss
        # loss_noisy = labeled nodes - clean节点 - 预测一致的节点
        idx_noisy  = torch.LongTensor(list(set(idx_train_noisy.tolist()) - set(dc_idx.tolist()))).to(self.device)
        y_noise = log_pred[idx_noisy]
        y_noise_1 = log_pred_1[idx_noisy]
        t_noise = labels[idx_noisy]
        loss_noise_1 = F.cross_entropy(y_noise, t_noise, reduction='none')
        loss_noise_2 = F.cross_entropy(y_noise_1, t_noise, reduction='none')
        loss_noise = loss_noise_1  + loss_noise_2

        loss1 = loss_noise.sum()/y_1.shape[0]

        decay_w = self.decay_w

        # =====ablation study=====
        inter_view_loss = kl_loss_compute(y_1, y_2).mean() +  kl_loss_compute(y_2, y_1).mean()
        # inter_view_loss = 0
        # ========================
        return loss_clean + loss_dc + decay_w*loss1 + co_lambda*inter_view_loss

# =====ablation study w/o GMM====
# class LabeledDividedLoss(nn.Module):
#     def __init__(self, args, device):
#         super(LabeledDividedLoss, self).__init__()
#         self.device = device
#         self.args = args
#         self.epochs = args.epochs
#         self.decay_w = args.decay_w

#     def forward(self, y_1, y_2, log_pred,log_pred_1, labels, idx_train, co_lambda=0.1, epoch=-1): 
#         y_clean = log_pred[idx_train]
#         y_clean_1 = log_pred_1[idx_train]
#         t_clean = labels[idx_train]
#         loss_pick_1 = F.cross_entropy(y_clean, t_clean, reduction='none')
#         loss_pick_2 = F.cross_entropy(y_clean_1, t_clean, reduction='none')
#         loss_pick = loss_pick_1  + loss_pick_2
       
#         # # loss_clean
#         # breakpoint()
#         loss_clean = torch.sum(loss_pick)/y_1.shape[0]

#         # # loss_dc
#         # ind_update_1 = idx_train_noisy
#         # p_1 = F.softmax(log_pred,dim=-1)
#         # p_2 = F.softmax(log_pred_1,dim=-1)
#         # # pred0.max(dim=1)[1] [1, n]:每个样本的类别； pred0.max(dim=1)[1][unfilter_idx]: unfilter_idx样本对应的类别
#         # filter_condition = ((log_pred.max(dim=1)[1][ind_update_1] != labels[ind_update_1]) &
#         #                     (log_pred.max(dim=1)[1][ind_update_1] == log_pred_1.max(dim=1)[1][ind_update_1]) &
#         #                     (p_1.max(dim=1)[0][ind_update_1] * p_2.max(dim=1)[0][ind_update_1]  > self.args.th**2 ))
#         # # breakpoint()
#         # dc_idx = ind_update_1[filter_condition]

#         # # # TODO 添加自适应权重（根据节点标签分布）
#         # loss_dc = (F.cross_entropy(log_pred[dc_idx],log_pred.max(dim=1)[1][dc_idx], reduction='none')+ \
#         #                            F.cross_entropy(log_pred_1[dc_idx], log_pred_1.max(dim=1)[1][dc_idx], reduction='none'))
#         # loss_dc = loss_dc.sum()/y_1.shape[0]

#         #  noisy node
#         # train - clean - 2个net预测一致且大于阈值的 = 噪声nodes(可能包含干净nodes 因此含有一定的信息) 添加noisy loss
#         # loss_noisy = labeled nodes - clean节点 - 预测一致的节点
#         # idx_noisy  = torch.LongTensor(list(set(idx_train_noisy.tolist()) - set(dc_idx.tolist()))).to(self.device)
#         # y_noise = log_pred[idx_noisy]
#         # y_noise_1 = log_pred_1[idx_noisy]
#         # t_noise = labels[idx_noisy]
#         # loss_noise_1 = F.cross_entropy(y_noise, t_noise, reduction='none')
#         # loss_noise_2 = F.cross_entropy(y_noise_1, t_noise, reduction='none')
#         # loss_noise = loss_noise_1  + loss_noise_2

#         # loss1 = loss_noise.sum()/y_1.shape[0]

#         # decay_w = self.decay_w

#         # =====ablation study=====
#         inter_view_loss = kl_loss_compute(y_1, y_2).mean() +  kl_loss_compute(y_2, y_1).mean()
#         # inter_view_loss = 0
#         # ========================
#         # return loss_clean + loss_dc + decay_w*loss1 + co_lambda*inter_view_loss
#         return loss_clean  + co_lambda*inter_view_loss
# =======================
class ORILabeledDividedLoss(nn.Module):
    def __init__(self, args):
        super(LabeledDividedLoss, self).__init__()
        self.args = args
        self.epochs = args.epochs
        self.increment = 0.5/self.epochs
        self.decay_w = args.decay_w

    # TODO RTGNN 在每一个epoch都进行的noise/clean的划分  但是现在我的并不是 加入warm up ？ 增加划分次数？
    def forward(self, y_1, y_2, t,  co_lambda=0.1, epoch=-1): 
        loss_pick_1 = F.cross_entropy(y_1, t, reduction='none')
        loss_pick_2 = F.cross_entropy(y_2, t, reduction='none')
        loss_pick = loss_pick_1  + loss_pick_2

        ind_sorted = torch.argsort(loss_pick)
        loss_sorted = loss_pick[ind_sorted]
        forget_rate = self.increment*epoch
        remember_rate = 1 - forget_rate
        mean_v = loss_sorted.mean()
        idx_small = torch.where(loss_sorted<mean_v)[0]
      
        remember_rate_small = idx_small.shape[0]/t.shape[0]
       
        remember_rate = max(remember_rate,remember_rate_small)
        num_remember = int(remember_rate * len(loss_sorted))
        ind_update = ind_sorted[:num_remember]
    
        loss_clean = torch.sum(loss_pick[ind_update])/y_1.shape[0]
        ind_all = torch.arange(1, t.shape[0]).long()
        ind_update_1 = torch.LongTensor(list(set(ind_all.detach().cpu().numpy())-set(ind_update.detach().cpu().numpy())))
        p_1 = F.softmax(y_1,dim=-1)
        p_2 = F.softmax(y_2,dim=-1)
        
        filter_condition = ((y_1.max(dim=1)[1][ind_update_1] != t[ind_update_1]) &
                            (y_1.max(dim=1)[1][ind_update_1] == y_2.max(dim=1)[1][ind_update_1]) &
                            (p_1.max(dim=1)[0][ind_update_1] * p_2.max(dim=1)[0][ind_update_1] > self.args.th**2 ))
                            # self.args.th**2 (1-(1-min(0.5,1/y_1.shape[0]))*epoch/self.args.epochs)
        dc_idx = ind_update_1[filter_condition]
        
        # adpative_weight = (p_1.max(dim=1)[0][dc_idx]*p_2.max(dim=1)[0][dc_idx])**(0.5-0.5*epoch/self.args.epochs)
        # loss_dc = adpative_weight*(F.cross_entropy(y_1[dc_idx],y_1.max(dim=1)[1][dc_idx], reduce=False)+ \
        #                            F.cross_entropy(y_2[dc_idx], y_1.max(dim=1)[1][dc_idx], reduce=False))

        loss_dc = F.cross_entropy(y_1[dc_idx],y_1.max(dim=1)[1][dc_idx], reduction='none')+ \
                                   F.cross_entropy(y_2[dc_idx], y_1.max(dim=1)[1][dc_idx], reduction='none')
        loss_dc = loss_dc.sum()/y_1.shape[0]
    
        remain_idx = torch.LongTensor(list(set(ind_update_1.detach().cpu().numpy())-set(dc_idx.detach().cpu().numpy())))
        
        loss1 = torch.sum(loss_pick[remain_idx])/y_1.shape[0]
        decay_w = self.decay_w

        inter_view_loss = kl_loss_compute(y_1, y_2).mean() +  kl_loss_compute(y_2, y_1).mean()

        return loss_clean + loss_dc+decay_w*loss1+co_lambda*inter_view_loss


class PseudoLoss(nn.Module):
    def __init__(self):
        super(PseudoLoss, self).__init__()

    def forward(self, y_1, y_2, idx_add,co_lambda=0.1):
        pseudo_label = y_1.max(dim=1)[1]
        loss_pick_1 = F.cross_entropy(y_1[idx_add], pseudo_label[idx_add], reduction='none')
        loss_pick_2 = F.cross_entropy(y_2[idx_add], pseudo_label[idx_add], reduction='none')
        loss_pick = loss_pick_1.mean() + loss_pick_2.mean()
        inter_view_loss = kl_loss_compute(y_1[idx_add], y_2[idx_add]).mean() + kl_loss_compute(y_2[idx_add], y_1[idx_add]).mean()
        loss = torch.mean(loss_pick)+co_lambda*inter_view_loss

        return loss

class IntraviewReg(nn.Module):
    def __init__(self,device):
        super(IntraviewReg, self).__init__()
        self.device = device
    def index_to_mask(self, index, size=None):
        index = index.view(-1)
        size = int(index.max()) + 1 if size is None else size
        mask = index.new_zeros(size, dtype=torch.bool)
        mask[index] = True
        return mask

    def bipartite_subgraph(self,subset, edge_index, max_size):

        subset = (self.index_to_mask(subset[0], size=max_size), self.index_to_mask(subset[1], size=max_size))
        node_mask = subset
        edge_mask = node_mask[0][edge_index[0]] & node_mask[1][edge_index[1]]
        return torch.where(edge_mask == True)[0]

    def neighbor_cons(self,y_1,y_2,edge_index,edge_weight,idx):
        if idx.shape[0]==0:
            return torch.Tensor([0]).to(self.device)
        weighted_adj = utils.to_scipy_sparse_matrix(edge_index, edge_weight.detach())
        colsum = np.array(weighted_adj.sum(0))
        r_inv = np.power(colsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0
        r_mat_inv = sp.diags(r_inv)
        norm_adj = weighted_adj.dot(r_mat_inv)

        norm_idx, norm_weight = utils.from_scipy_sparse_matrix(norm_adj)
        idx_all = torch.arange(0, y_1.shape[0]).to(self.device)

        filter_idx = self.bipartite_subgraph((idx_all,idx), norm_idx.to(self.device),max_size=int(y_1.shape[0]))
        edge_index,edge_weight = norm_idx[:,filter_idx], norm_weight[filter_idx]
        edge_index, edge_weight = edge_index.to(self.device), edge_weight.to(self.device)

        intra_view_loss = (edge_weight*kl_loss_compute(y_1[edge_index[1]], y_1[edge_index[0]].detach())).sum()+ \
                        (edge_weight*kl_loss_compute(y_2[edge_index[1]], y_2[edge_index[0]].detach())).sum()
        intra_view_loss = intra_view_loss/idx.shape[0]
        return intra_view_loss

    def forward(self,y_1,y_2,idx_label,edge_index,edge_weight):
        neighbor_kl_loss = self.neighbor_cons(y_1, y_2, edge_index, edge_weight, idx_label)
        return neighbor_kl_loss


class RTGNN(nn.Module):
    def __init__(self, args, device):
        super(RTGNN, self).__init__()
        self.device = device
        self.args = args
        self.best_acc_pred_val = 0
        self.best_pred = None
        self.best_graph = None
        self.weights = None
        self.estimator = None
        self.model = None
        self.pred_edge_index = None
        self.criterion = LabeledDividedLoss(args, device)
        self.criterion_pse = PseudoLoss()
        self.intra_reg = IntraviewReg(device)



    def fit(self, features, adj, labels, idx_train, idx_val):
        args = self.args
        edge_index, _ = utils.from_scipy_sparse_matrix(adj)
        edge_index = edge_index.to(self.device)

        if sp.issparse(features):
            features = sparse_mx_to_torch_sparse_tensor(features).to_dense().float()
        else:
            features = torch.FloatTensor(np.array(features))
        features = features.to(self.device)
        
        self.edge_index = edge_index
        labels = torch.LongTensor(np.array(labels)).to(self.device)
        self.labels = labels
        self.features = features
        self.idx_unlabel = torch.LongTensor(list(set(range(features.shape[0])) - set(idx_train))).to(self.device)
        self.idx_train = torch.LongTensor(idx_train).to(self.device)

        self.predictor = Dual_GCN(nfeat=features.shape[1],
                                  nhid=self.args.hidden,
                                  nclass=labels.max().item() + 1,
                                  self_loop=True,
                                  dropout=self.args.dropout, device=self.device).to(self.device)

        self.estimator = EstimateAdj(features, features.shape[1], args, device=self.device).to(self.device)

        # self.pred_edge_index = self.KNN(edge_index, features, self.args.K, idx_train).to(self.device)
        self.pred_edge_index = self.get_train_edge(edge_index,features,self.args.K ,idx_train).to(self.device)

        self.optimizer = optim.Adam(list(self.estimator.parameters()) + list(self.predictor.parameters()),
                                    lr=args.lr, weight_decay=args.weight_decay)

        
        for epoch in range(args.epochs):
            # 添加warm up
            if epoch < args.num_warmup:
                self.warmup(epoch, features, edge_index, idx_train, idx_val)

            else:
                # =====ablation study w/o GMM======
                self.idx_train_noisy, self.idx_train_clean = self.devide_labeled_nodes(features, self.idx_train, edge_index, epoch)
                #self.pred_edge_index 中去掉 self.idx_train_noisy
                # 示例数据
                curr_pred_edge_index = remove_tensor_by_prefix(self.pred_edge_index, self.idx_train_noisy)
                self.train(epoch, features, edge_index, idx_train, idx_val, curr_pred_edge_index)
                # ===========================
                # self.train(epoch, features, edge_index, idx_train, idx_val, self.pred_edge_index)


        print("Optimization Finished!")
      
        print("picking the best model according to validation performance")

        self.predictor.load_state_dict(self.predictor_model_weigths)


    def devide_labeled_nodes(self, features,idx_train, edge_index, epoch):

        self.predictor.eval()
        with torch.no_grad():
            log_pred, log_pred_1 = self.predictor(features, self.edge_index)
            # GMM
            prob1 = gmm_eval(log_pred[idx_train],self.labels[idx_train], self.args, epoch)   
            prob2 = gmm_eval(log_pred_1[idx_train],self.labels[idx_train], self.args, epoch) 

        pred1 = (prob1 > self.args.p_threshold)      
        pred2 = (prob2 > self.args.p_threshold) 

        tmp_idx_clean = idx_train[pred1]
        tmp_idx_clean1 = idx_train[pred2]

        # 求两个tensor的交集
        unique_tmp_idx_clean = torch.unique(tmp_idx_clean)
        unique_tmp_idx_clean1 = torch.unique(tmp_idx_clean1)
        # 使用 torch.isin 查找交集
        idx_train_clean = unique_tmp_idx_clean[torch.isin(unique_tmp_idx_clean, unique_tmp_idx_clean1)]

        # 计算 tensor1 中不在 tensor2 中的元素
        idx_train_noisy  = torch.tensor(list(set(self.idx_train.tolist()) - set(idx_train_clean.tolist()))).to(self.device)
        # pred_edge_index = self.KNN(edge_index, features, self.args.K, self.idx_train_clean)
        return idx_train_noisy,idx_train_clean

    def train(self, epoch, features, edge_index, idx_train, idx_val, curr_pred_edge_index):
        args = self.args
        self.predictor.train()
        self.optimizer.zero_grad()
        representations, rec_loss = self.estimator(edge_index, features)

        ###### 对应 TODO-1 的修改######
        predictor_weights = self.estimator.get_estimated_weigths(curr_pred_edge_index,representations)
        pred_edge_index = torch.cat([edge_index,curr_pred_edge_index],dim=1)
        # breakpoint()
        predictor_weights = torch.cat([torch.ones([edge_index.shape[1]],device=self.device),predictor_weights],dim=0)
        ###### over TODO-1 的修改######

        log_pred, log_pred_1 = self.predictor(features, pred_edge_index, predictor_weights)
        acc_pred_train0 = accuracy(log_pred[idx_train], self.labels[idx_train])
        acc_pred_train1 = accuracy(log_pred_1[idx_train], self.labels[idx_train])
        
        # print("=====Train Accuray=====")
        # print("Epoch %d: #1 = %f, #2= %f"%(epoch,acc_pred_train0.item(),acc_pred_train1.item()))

        pred = F.softmax(log_pred, dim=1).detach()
        pred1 = F.softmax(log_pred_1, dim=1).detach()

        # ==== ablation study ====
        self.idx_add= self.get_pseudo_label(pred, pred1)
        # self.idx_add = []
        # ===========================
        if epoch==0:
            loss_pred = F.cross_entropy(log_pred[idx_train],self.labels[idx_train])+F.cross_entropy(log_pred_1[idx_train],self.labels[idx_train])
        else:
            #  ==== ablation study w/o GMM ====
            loss_pred = self.criterion(log_pred[idx_train],log_pred_1[idx_train], log_pred, log_pred_1, self.labels, self.idx_train_clean, self.idx_train_noisy,
                                                    co_lambda=self.args.co_lambda, epoch=epoch)
            # ======================
            # loss_pred = self.criterion(log_pred[idx_train],log_pred_1[idx_train], log_pred, log_pred_1, self.labels, idx_train,
                                                    # co_lambda=self.args.co_lambda, epoch=epoch)

        if len(self.idx_add) != 0:
            loss_add = self.criterion_pse(log_pred, log_pred_1, self.idx_add, co_lambda=self.args.co_lambda)
        else:
            loss_add = torch.Tensor([0]).to(self.device)
        # ====ablation study====
        neighbor_kl_loss = self.intra_reg(log_pred,log_pred_1,self.idx_train, pred_edge_index,predictor_weights)
        # neighbor_kl_loss = 0
        # =====================
        total_loss = loss_pred + self.args.alpha * rec_loss + loss_add + self.args.co_lambda*(neighbor_kl_loss)
        total_loss.backward()
        self.optimizer.step()

        self.predictor.eval()
        output0, output1 = self.predictor(features, pred_edge_index, predictor_weights)
        acc_pred_val0 = accuracy(output0[idx_val], self.labels[idx_val])
        acc_pred_val1 = accuracy(output1[idx_val], self.labels[idx_val])
        acc_pred_val = 0.5*(acc_pred_val0+acc_pred_val1)

        if acc_pred_val >= self.best_acc_pred_val:
            self.best_acc_pred_val = acc_pred_val
            self.best_pred_graph = predictor_weights.detach()
            self.best_edge_idx = pred_edge_index.detach()
            self.best_pred = pred.detach()
            self.predictor_model_weigths = deepcopy(self.predictor.state_dict())
        # print("=====Validation Accuray=====")
        # print("Epoch %d: #1 = %f, #2= %f"%(epoch,acc_pred_val0.item(),acc_pred_val1.item()))
    
    def warmup(self, epoch, features, edge_index, idx_train, idx_val):
        args = self.args
        self.predictor.train()
        self.optimizer.zero_grad()
        representations, rec_loss = self.estimator(edge_index, features)

        predictor_weights = self.estimator.get_estimated_weigths(self.pred_edge_index,representations)
        pred_edge_index = torch.cat([edge_index, self.pred_edge_index],dim=1)
        predictor_weights = torch.cat([torch.ones([edge_index.shape[1]],device=self.device),predictor_weights],dim=0)

        log_pred, log_pred_1 = self.predictor(features, pred_edge_index, predictor_weights)
        # acc_pred_train0 = accuracy(log_pred[idx_train], self.labels[idx_train])
        # acc_pred_train1 = accuracy(log_pred_1[idx_train], self.labels[idx_train])
        
        # print("=====Train Accuray=====")
        # print("Epoch %d: #1 = %f, #2= %f"%(epoch,acc_pred_train0.item(),acc_pred_train1.item()))

        pred = F.softmax(log_pred, dim=1).detach()
        pred1 = F.softmax(log_pred_1, dim=1).detach()

        loss_pred = F.cross_entropy(log_pred[idx_train],self.labels[idx_train])+F.cross_entropy(log_pred_1[idx_train],self.labels[idx_train])
        # if args.noise=='pair':  # penalize confident prediction for asymmetric noise
        #     penalty = conf_penalty(outputs)
        #     L = loss_pred + penalty      
        # elif args.noise=='uniform':   
        #     L = loss_pred
        # penalty = conf_penalty(log_pred) + conf_penalty(log_pred_1)
        # L = loss_pred + penalty 
        L = loss_pred 
        L.backward()  
        self.optimizer.step()

        self.predictor.eval()
        output0, output1 = self.predictor(features, pred_edge_index, predictor_weights)
        acc_pred_val0 = accuracy(output0[idx_val], self.labels[idx_val])
        acc_pred_val1 = accuracy(output1[idx_val], self.labels[idx_val])
        acc_pred_val = 0.5*(acc_pred_val0+acc_pred_val1)

        if acc_pred_val >= self.best_acc_pred_val:
            self.best_acc_pred_val = acc_pred_val
            self.best_pred_graph = predictor_weights.detach()
            self.best_edge_idx = pred_edge_index.detach()
            self.best_pred = pred.detach()
            self.predictor_model_weigths = deepcopy(self.predictor.state_dict())

    def test(self, idx_test):
        features = self.features
        self.predictor.eval()
        estimated_weights = self.best_pred_graph
        pred_edge_index = self.best_edge_idx
        output0, output1 = self.predictor(features, pred_edge_index, estimated_weights)
        acc_pred_test0 = accuracy(output0[idx_test], self.labels[idx_test])
        acc_pred_test1 = accuracy(output1[idx_test], self.labels[idx_test])
        
        acc_mean = (acc_pred_test0+acc_pred_test1)/2 * 100
        
        # print("Test mean Accuray: %f "%(acc_mean))
        # f = open(self.args.dataset+  str(self.args.ptb_rate)+ '_' + str(self.args.noise) + "_result.txt", "a")
        # f.write("avg_result: "+ str(acc_mean)+" acc_pred_test0: "+str(acc_pred_test0.item())+"\t"
        #         +"acc_pred_test1 "+ str(acc_pred_test1.item())+"\n")
        # f.write("seed: "+ str(self.args.seed)+" params--K: "+ str(self.args.K)+" th: "+str(self.args.th)+"\t"
        #         +" alpha: "+ str(self.args.alpha)+ " ptb_rate: "+ str(self.args.ptb_rate) +
        #         " tau: "+ str(self.args.tau)+ " n_neg: "+ str(self.args.n_neg)+" decay_w: "+ str(self.args.decay_w)+
        #         " p_threshold: "+ str(self.args.p_threshold)+ "\n")
        # f.close()
        return (acc_pred_test0+acc_pred_test1)/2


    def get_train_edge(self, edge_index, features, n_p, idx_train):
        '''
        obtain the candidate edge between labeled nodes and unlabeled nodes based on cosine sim
        n_p is the top n_p labeled nodes similar with unlabeled nodes
        '''
      
        if n_p == 0:
            return None

        poten_edges = []
        if n_p > len(idx_train) or n_p < 0:
            for i in range(len(features)):
                indices = set(idx_train)
                indices = indices - set(edge_index[1,edge_index[0]==i])
                for j in indices:
                    pair = [i, j]
                    poten_edges.append(pair)
        else:
            # for i in range(len(features)):
            #     sim = torch.div(torch.matmul(features[i],features[idx_train].T), features[i].norm()*features[idx_train].norm(dim=1))
            #     _,rank = sim.topk(n_p)
            #     if rank.max() < len(features) and rank.min() >= 0:
            #         indices = idx_train[rank.cpu().numpy()]
            #         indices = set(indices)
            #     else:
            #         indices = set()
            #     indices = indices - set(edge_index[1,edge_index[0]==i])
            #     for j in indices:
            #         pair = [i, j]
            #         poten_edges.append(pair)
            # 临时添加 在cora上试试
            # for i in idx_train:
            #     sim = torch.div(torch.matmul(features[i], features[self.idx_unlabel].T),
            #                     features[i].norm() * features[self.idx_unlabel].norm(dim=1))
            #     _, rank = sim.topk(n_p)
            #     indices = self.idx_unlabel[rank.cpu().numpy()]
            #     for j in indices:
            #         pair = [i, j]
            #         poten_edges.append(pair)
            for i in self.idx_unlabel:
                sim = torch.div(torch.matmul(features[i], features[idx_train].T),
                                features[i].norm() * features[idx_train].norm(dim=1))
                _, rank = sim.topk(n_p)
                indices = idx_train[rank.cpu().numpy()]
                indices = set(indices)
               
                for j in indices:
                    pair = [i, j]
                    poten_edges.append(pair)
        
        edge_index = list(edge_index.T)
        poten_edges = set([tuple(t) for t in poten_edges])-set([tuple(t) for t in edge_index])
        poten_edges = [list(s) for s in poten_edges]
        poten_edges = torch.as_tensor(poten_edges).T.to(self.device)
        # breakpoint()

        # 把转为无向图去掉 只添加
        # poten_edges = utils.to_undirected(poten_edges,len(features)).to(self.device)
        return poten_edges

    def get_pseudo_label(self, pred0, pred1):
        
        filter_condition = ((pred0.max(dim=1)[1][self.idx_unlabel] == pred1.max(dim=1)[1][self.idx_unlabel])&
                            (pred0.max(dim=1)[0][self.idx_unlabel]*pred1.max(dim=1)[0][self.idx_unlabel] > self.args.th**2))
        idx_add = self.idx_unlabel[filter_condition]

        return idx_add.detach()



class EstimateAdj(nn.Module):

    def __init__(self, features, nfea, args, device):
        super(EstimateAdj, self).__init__()

        self.estimator = GCN(nfea, args.edge_hidden, args.edge_hidden, dropout=0.0, device=device)
        self.device = device
        self.args = args
        self.representations = 0
        self.sigmoid = nn.Sigmoid()

    def forward(self, edge_index, features):
        representations = self.estimator(features, edge_index, \
                                         torch.ones([edge_index.shape[1]]).to(self.device).float())
        representations =F.normalize(representations,dim=-1)
        rec_loss = self.reconstruct_loss(edge_index, representations)
        return representations, rec_loss


    def get_estimated_weigths_RTGNN(self, edge_index, representations, origin_w=None):
        # 这里的edge_index为拼接候选边后的所有的边
        x0 = representations[edge_index[0]]
        x1 = representations[edge_index[1]]
        output = torch.sum(torch.mul(x0, x1), dim=1)
        # estimated_weights = F.relu(output)
        # estimated_weights = torch.max(output, torch.tensor(0.0, device=output.device))
        # 因为下面有一个self.args.tau>0 所以这里可以不用过滤掉小于0的
        estimated_weights = output

        if estimated_weights.shape[0] != 0:
            estimated_weights[estimated_weights < self.args.tau] = 0
            if origin_w != None:
                estimated_weights = origin_w + estimated_weights*(1-origin_w)

        return estimated_weights,None
    
    def get_estimated_weigths(self, edge_index, representations):
        # arxiv 上分批次计算X0T乘x[1-100]X0T乘x[101-200].... 或者放到cpu上算 x0 = representations[edge_index[0]].cpu()
        x0 = representations[edge_index[0]]
        x1 = representations[edge_index[1]]
        output = torch.sum(torch.mul(x0,x1),dim=1)

        # estimated_weights = F.relu(output)
        # 使用 torch.max 来实现 ReLU 功能
        estimated_weights = torch.max(output, torch.tensor(0.0, device=output.device))
        # breakpoint()
        
        estimated_weights[estimated_weights < self.args.tau] = 0.0

        return estimated_weights

    def reconstruct_loss(self, edge_index, representations):
        num_nodes = representations.shape[0]
        randn = utils.negative_sampling(edge_index, num_nodes=num_nodes,
                                        num_neg_samples=self.args.n_neg * num_nodes)

        randn = randn[:, randn[0] < randn[1]]
        edge_index = edge_index[:, edge_index[0] < edge_index[1]]
        neg0 = representations[randn[0]]
        neg1 = representations[randn[1]]
        neg = torch.sum(torch.mul(neg0, neg1), dim=1)

        pos0 = representations[edge_index[0]]
        pos1 = representations[edge_index[1]]
        pos = torch.sum(torch.mul(pos0, pos1), dim=1)
        rec_loss = (F.mse_loss(neg, torch.zeros_like(neg), reduction='sum') \
                    + F.mse_loss(pos, torch.ones_like(pos), reduction='sum')) \
                   * num_nodes / (randn.shape[1] + edge_index.shape[1])

        return rec_loss

