import os
import argparse
import numpy as np
import torch
from utils import noisify_with_P, random_disassortative_splits
from dataset import Dataset

from models.RTGNN import RTGNN

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--edge_hidden', type=int, default=64,
                    help='Number of hidden units of MLP graph constructor')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default="citeseer", help='dataset')
parser.add_argument('--co_lambda',type=float,default=0.1,
                     help='weight for consistency regularization term')

parser.add_argument('--epochs', type=int,  default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--noise', type=str, default='uniform', choices=['uniform', 'pair'],
                    help='type of noises')
parser.add_argument('--ptb_rate', type=float, default=0.2,
                    help="noise ptb_rate")
parser.add_argument('--split', type=int, default=0,
                    help='num of split ')

parser.add_argument('--cudaId', type=int, default=0,
                    help='cuda ID ')
parser.add_argument("--n_neg", type=int, default=50,
                    help='number of negitive sampling for each node')

parser.add_argument('--tau',type=float, default=0.1,
                    help='threshold of filtering noisy edges')
parser.add_argument("--K", type=int, default=100,
                    help='number of KNN search for each node')

parser.add_argument('--p_threshold', default=0.4, type=float, help='clean probability threshold')
parser.add_argument('--alpha', type=float, default=0.03,
                    help='loss weight of graph reconstruction')
parser.add_argument('--th',type=float, default=0.9,
                    help='threshold of adding pseudo labels')
parser.add_argument('--decay_w', type=float, default=0.1,
                    help='down-weighted factor')
parser.add_argument('--num_warmup', type=int, default=30,
                    help='warm up')


args = parser.parse_args()
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

device = torch.device('cuda:'+str(args.cudaId) if torch.cuda.is_available() else 'cpu')

if args.dataset == "cs":
    from torch_geometric.datasets import Coauthor
    import torch_geometric.utils as utils
    dataset = Coauthor('./data','CS')
    labels = dataset.data.y.numpy()
    adj = utils.to_scipy_sparse_matrix(dataset.data.edge_index)
    features = dataset.data.x.numpy()
    labels = dataset.data.y.numpy()

else:
    data = Dataset(root='./data', name=args.dataset)
    adj, features, labels = data.adj, data.features, data.labels

#  load split
train_val_test_split = torch.load("./data/splits/"+args.dataset+str(args.split)+".pth")
idx_train = train_val_test_split["idx_train"]
idx_val = train_val_test_split["idx_val"] 
idx_test = train_val_test_split["idx_test"] 


ptb = args.ptb_rate
nclass = labels.max() + 1
noise_labels = labels.copy()
train_labels = labels[idx_train]
noise_y, P = noisify_with_P(train_labels,nclass, ptb, 10, args.noise) 
noise_labels[idx_train] = noise_y
mask = noise_labels[idx_train] == train_labels


model = RTGNN(args, device)
model.fit(features, adj, noise_labels, idx_train, idx_val)
acc = model.test(idx_test)
print(str(acc.item()))

