import numpy as np
from numpy.testing import assert_array_almost_equal
import torch
from sklearn.metrics.pairwise import cosine_similarity as cos
from scipy.sparse import csc_matrix

def build_uniform_P(size, noise):
    """ The noise matrix flips any class to any other with probability
    noise / (#class - 1).
    """

    assert(noise >= 0.) and (noise <= 1.)

    P = np.float64(noise) / np.float64(size-1) * np.ones((size, size))
    np.fill_diagonal(P, (np.float64(1)-np.float64(noise))*np.ones(size))
    
    diag_idx = np.arange(size)
    P[diag_idx,diag_idx] = P[diag_idx,diag_idx] + 1.0 - P.sum(0)
    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P

def build_pair_p(size, noise):
    assert(noise >= 0.) and (noise <= 1.)
    P = (1.0 - np.float64(noise)) * np.eye(size)
    for i in range(size):
        P[i,i-1] = np.float64(noise)
    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """

    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y

def default_corrupt_large(trainset, split_idx, ratio, seed, t="asymm"):
    """
    Corrupt labels in trainset.

    Args:
        trainset (torch.data.dataset): trainset with clean labels
        split_idx (torch.tensor): index of training set
        ratio (float): corruption ratio
        seed (int): random seed
        t (str): type of corruption

    Returns:
        label (torch.tensor): corrupted labels

    """
    if t == "pair":
        # initialization
        label = []
        y = trainset.y[split_idx].cpu().numpy()
        num_classes = np.max(y) + 1
        np.random.seed(seed)

        # generate the corruption matrix
        C = np.eye(num_classes) * (1 - ratio)
        row_indices = np.arange(num_classes)
        for i in range(num_classes):
            C[i][np.random.choice(row_indices[row_indices != i])] = ratio

        # corrupt the labels and append them into a list
        for label_i in trainset.y[split_idx]:
            data1 = np.random.choice(num_classes, p=C[label_i])
            label.append(data1)
        label = torch.tensor(label).long()
        return label
    elif t == "uniform":
        # initialization
        label = []
        y = trainset.y[split_idx].cpu().numpy()
        num_classes = np.max(y) + 1
        np.random.seed(seed)

        # generate the corruption matrix
        off_diagnal = ratio * \
            np.full((num_classes, num_classes), 1 / (num_classes-1))
        np.fill_diagonal(off_diagnal, 0)
        data = np.eye(num_classes) * (1 - ratio) + off_diagnal

        # corrupt the labels and append them into a list
        for label_i in trainset.y[split_idx]:
            data1 = np.random.choice(num_classes, p=data[label_i])
            label.append(data1)
        label = np.array(label)
        label = torch.from_numpy(label)

        return label

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    sparserow=torch.LongTensor(sparse_mx.row).unsqueeze(1)
    sparsecol=torch.LongTensor(sparse_mx.col).unsqueeze(1)
    sparseconcat=torch.cat((sparserow, sparsecol),1)
    sparsedata=torch.FloatTensor(sparse_mx.data)
    return torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(sparse_mx.shape))

def edge_index_to_sparse_mx(edge_index, num_nodes):
    edge_weight = np.array([1] * len(edge_index[0]))
    adj = csc_matrix((edge_weight, (edge_index[0], edge_index[1])),
                     shape=(num_nodes, num_nodes)).tolil()
    return adj

def sparse_mx_to_torch_sparse_tensor_arxiv(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def noisify(y, p_minus, p_plus=None, random_state=0):
    """ Flip labels with probability p_minus.
    If p_plus is given too, the function flips with asymmetric probability.
    """

    assert np.all(np.abs(y) == 1)

    m = y.shape[0]
    new_y = y.copy()
    coin = np.random.RandomState(random_state)

    if p_plus is None:
        p_plus = p_minus

    # This can be made much faster by tossing all the coins and completely
    # avoiding the loop. Although, it is not simple to write the asymmetric
    # case then.
    for idx in np.arange(m):
        if y[idx] == -1:
            if coin.binomial(n=1, p=p_minus, size=1) == 1:
                new_y[idx] = -new_y[idx]
        else:
            if coin.binomial(n=1, p=p_plus, size=1) == 1:
                new_y[idx] = -new_y[idx]

    return new_y

def noisify_with_P(y_train, nb_classes, noise, random_state=None,  noise_type='uniform'):

    if noise > 0.0:
        if noise_type=='uniform':
            print('Uniform noise')
            P = build_uniform_P(nb_classes, noise)
        elif noise_type == 'pair':
            print('Pair noise')
            P = build_pair_p(nb_classes, noise)
        else:
            print('Noise type have implemented')
        # seed the random numbers with #run
        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    else:
        P = np.eye(nb_classes)

    return y_train, P


def to_onehot(labels):
    class_size = labels.max() + 1
    onehot = np.eye(class_size)
    
    return onehot[labels]

def add_extra_class(labels,idx_train,n_class,pivot_num=15):
    labels_idx = torch.randperm(len(idx_train))[:pivot_num]
    labels[idx_train[labels_idx]] = n_class
    return labels,labels_idx
# %%
import os
def load_emd(path, dataset):

    graph_embedding = np.genfromtxt(
            os.path.join(path,"{}.emb".format(dataset)),
            skip_header=1,
            dtype=float)
    embedding = np.zeros([graph_embedding.shape[0],graph_embedding.shape[1]-1])

    for i in range(graph_embedding.shape[0]):
        embedding[int(graph_embedding[i,0])] = graph_embedding[i,1:]
    
    return embedding
# %%
from sklearn.model_selection import train_test_split
def get_train_val_test(nnodes, val_size=0.1, test_size=0.8, stratify=None, seed=None):
    """This setting follows nettack/mettack, where we split the nodes
    into 10% training, 10% validation and 80% testing data

    Parameters
    ----------
    nnodes : int
        number of nodes in total
    val_size : float
        size of validation set
    test_size : float
        size of test set
    stratify :
        data is expected to split in a stratified fashion. So stratify should be labels.
    seed : int or None
        random seed

    Returns
    -------
    idx_train :
        node training indices
    idx_val :
        node validation indices
    idx_test :
        node test indices
    """

    assert stratify is not None, 'stratify cannot be None!'

    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(nnodes)
    train_size = 1 - val_size - test_size
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=seed,
                                                #    random_state=None,
                                                   train_size=train_size + val_size,
                                                   test_size=test_size,
                                                   stratify=stratify)

    if stratify is not None:
        stratify = stratify[idx_train_and_val]

    idx_train, idx_val = train_test_split(idx_train_and_val,
                                        #   random_state=None,
                                          random_state=seed,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)

    return idx_train, idx_val, idx_test


def accuracy(output, labels):
    """Return accuracy of output compared to labels.

    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels

    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def get_feature_matrix(x):
    fMatrix = x.cpu()
    feature_matrix = cos(fMatrix)
    return feature_matrix

#取出矩阵每列前k个值
def get_top_k_matrix(features, k = 5):
    feature_matrix = get_feature_matrix(features)

    num_nodes = feature_matrix.shape[0]
    row_idx = np.arange(num_nodes) 
    feature_matrix[feature_matrix.argsort(axis=0)[:num_nodes - k], row_idx] = 0. 
    feature_matrix = torch.from_numpy(feature_matrix.T)
    # to sparse matrix
    idx = torch.nonzero(feature_matrix).T  
    data = feature_matrix[idx[0],idx[1]]
    adj_coo = torch.sparse_coo_tensor(idx, data, feature_matrix.shape)  # to COO matrix


    return adj_coo

def random_disassortative_splits(labels, num_classes,labeled_rate):
    labels = torch.LongTensor(np.array(labels))
    labels, num_classes = labels, num_classes
    indices = []
    for i in range(num_classes):
        index = torch.nonzero((labels == i)).view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)
    percls_trn = int(round(labeled_rate * (labels.size()[0] / num_classes)))
    val_lb = int(round((0.2-labeled_rate) * labels.size()[0]))

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)
    rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    # train_mask = index_to_mask(train_index, size=labels.size()[0])
    # val_mask = index_to_mask(rest_index[:val_lb], size=labels.size()[0])
    # test_mask = index_to_mask(rest_index[val_lb:], size=labels.size()[0])
    val_index = rest_index[:val_lb]
    test_index = rest_index[val_lb:]
    # return train_mask.to(device), val_mask.to(device), test_mask.to(device)
    return train_index, val_index, test_index


