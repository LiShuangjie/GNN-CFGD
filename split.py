#%%
import time
import argparse
import numpy as np
import torch
from utils import random_disassortative_splits

from dataset import Dataset

# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--dataset', type=str, default="citeseer", help='dataset')

for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:

    args = parser.parse_args()
    dataset = args.dataset

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    labeled_rate = 0.02
   
    if args.dataset == "cs":
        from torch_geometric.datasets import Coauthor
        import torch_geometric.utils as utils
        dataset = Coauthor('./data','CS')
        labels = dataset.data.y.numpy()
        num_class = labels.max() + 1
        idx_train, idx_val, idx_test = random_disassortative_splits(labels, num_class, labeled_rate)

    else:
        data = Dataset(root='./data', name=args.dataset)
        adj, features, labels = data.adj, data.features, data.labels
        num_class = labels.max() + 1
        idx_train, idx_val, idx_test = random_disassortative_splits(labels, num_class, labeled_rate)

    # save data split
    train_val_test_split = {
        "idx_train":idx_train,
        "idx_val":idx_val,
        "idx_test":idx_test
    }
    torch.save(train_val_test_split,"./data/splits/lr2/"+args.dataset+str(seed)+".pth")

