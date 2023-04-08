import os
import torch
import time
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing


def compute_distance(query, gallery, reference):
    """The euclidean distance function

    Args:
        query (nparray): training set of target model.
        gallery (nparray): test set of target model.
        reference (nparray): reference samples.

        The input form is feature embedding.
    """
    q, g, r = query.size(0), gallery.size(0), reference.size(0)
    mat1 = np.power(query, 2).sum(dim=1, keepdim=True).expand(q, r)
    mat2 = np.power(gallery, 2).sum(dim=1, keepdim=True).expand(g, r)
    mat3 = np.power(reference, 2).sum(dim=1, keepdim=True).expand(r, q).t()
    mat4 = np.power(reference, 2).sum(dim=1, keepdim=True).expand(r, g).t()
    query_distmat, gallery_distmat = mat1 + mat3, mat2 + mat4
    query_distmat.addmm_(query, reference.t(), beta=1, alpha=-2)
    gallery_distmat.addmm_(gallery, reference.t(), beta=1, alpha=-2)
    return query_distmat, gallery_distmat


class AttackDataset(Dataset):
    """The attack dataset, which is the feature embedding of target model's training and test test

        'data_name' structures as DatasetName_ModelName_QueryOrGallery (eg. duck_xception_gallery)
            gallery -- test set, query -- training set

        return the similarity vector and feature embedding
    """
    def __init__(self, is_train=True, data_path='./data/', reference_num=2000,
                 data_name='duke_xception_gallery'):
        super(AttackDataset, self).__init__()
        data_names = sorted(os.listdir(data_path))
        self.data, self.label = [], []
        self.is_train = is_train
        num = len(data_names)
        print("=> Loading Dataset {name}......".format(name=data_name))
        for i in range(0, num, 2):
            if data_name in str(data_names[i]):
                gallery, query = np.load(os.path.join(data_path, data_names[i])), np.load(os.path.join(data_path, data_names[i + 1]))
                np.random.seed(2)
                tmp = np.concatenate((query, gallery))
                reference = tmp[np.random.permutation(len(tmp))][0:reference_num]
                query_distmat, gallery_distmat = compute_distance(torch.tensor(query), torch.tensor(gallery), torch.tensor(reference))
                np.random.seed(32)
                query_permutation, gallery_permutation = np.random.permutation(len(query)), np.random.permutation(len(gallery))
                query, gallery = torch.from_numpy(query), torch.from_numpy(gallery)
                self.trn_similarity, self.trn_feature, self.trn_label = torch.cat((query_distmat[query_permutation][0:2000], gallery_distmat[gallery_permutation][0:2000])), \
                                                                        torch.cat((query[query_permutation][0:2000], gallery[gallery_permutation][0:2000])),\
                                                                        torch.cat((torch.ones((2000, 1)), torch.zeros((2000, 1))))
                self.test_similarity, self.test_feature, self.test_label = torch.cat((query_distmat[query_permutation][-6000:], gallery_distmat[gallery_permutation][-6000:])), \
                                                                           torch.cat((query[query_permutation][-6000:], gallery[gallery_permutation][-6000:])), \
                                                                           torch.cat((torch.ones((6000, 1)),torch.zeros((6000, 1))))
        # minmnax normalization
        minmax = preprocessing.MinMaxScaler()
        self.trn_similarity = minmax.fit_transform(self.trn_similarity)
        self.test_similarity = minmax.transform(self.test_similarity)
        self.trn_feature = minmax.fit_transform(self.trn_feature)
        self.test_feature = minmax.transform(self.test_feature)
        print("train datasets number: {num1}; test datasets number: {num2}".format(num1=len(self.trn_label), num2=len(self.test_label)))

    def __getitem__(self, idx):
        if self.is_train:
            similarity = self.trn_similarity[idx]
            feature = self.trn_feature[idx]
            label = self.trn_label[idx]
        else:
            similarity = self.test_similarity[idx]
            feature = self.test_feature[idx]
            label = self.test_label[idx]
        return similarity.astype(np.float32), feature.astype(np.float32), label

    def __len__(self):
        if self.is_train:
            length = len(self.trn_label)
        else:
            length = len(self.test_label)
        return length


class Attacker(nn.Module):
    """
        Attack model without anchor selector, implemented as the 4-layer MLP
    """
    def __init__(self, input_dim, hidden_dim):
        super(Attacker, self).__init__()
        self.attacker = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(),
                                       nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                       nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                       nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                       nn.Linear(hidden_dim, 1))


    def forward(self, x):
        out = self.attacker(x)
        return torch.sigmoid(out)


class AnchorSelector(Attacker):
    """
        Attack model with anchor selector
    """
    def __init__(self, input_dim, hidden_dim, feature_dim):
        super().__init__(input_dim, hidden_dim)
        self.anchor_selector = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU(),
                                             nn.Linear(feature_dim, input_dim), nn.ReLU())

    def forward(self, x, f):
        weight = self.anchor_selector(f)
        x = x*weight
        out = self.attacker(x)
        return torch.sigmoid(out)


def train_model(model, method, epoches):
    """A unified pipeline for training and evaluating a model.

           Args:
               model (Attacker): implementation of Attacker.
               method (str): method name.
               epoches (int): maximum epoch.
    """
    loss = F.binary_cross_entropy
    optim = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0., weight_decay=0.)
    lr = torch.optim.lr_scheduler.StepLR(optim, 2000, gamma=0.5)
    print('=> Start training method {method}'.format(method=method))
    end = time.time()
    for epoch in range(1, epoches):
        num_batches = len(train_loader)
        for idx, data in enumerate(train_loader):
            similarity, label = data[0].to(device), data[2].to(device)
            if method == "ASSD":
                feature = data[1].to(device)
                out = model(similarity, feature)
            elif method == "FE":
                feature = data[1].to(device)
                out = model(feature)
            else:
                out = model(similarity)
            l = loss(out, label)
            optim.zero_grad()
            l.backward()
            optim.step()
            out[out > 0.5] = 1
            out[out < 0.5] = 0
            batch_size = len(label)
            correct = torch.sum(out == label.data)
            acc = correct * 100.0 / batch_size
            lr.step()

        if epoch % 100 == 0 or epoch == 1:
            batch_time = time.time() - end
            end = time.time()
            print(
                'epoch: [{0}/{1}][{2}/{3}]\t'
                'time {batch_time:.4f}\t'
                'losses: {losses:.4f}\t'
                'acc: {acc:.4f}\t'
                'lr: {lr:.6f}'.format(
                    epoch,
                    epoches - 1,
                    idx + 1,
                    num_batches,
                    batch_time=batch_time,
                    losses=l,
                    acc=acc,
                    lr=optim.param_groups[-1]['lr']
                )
            )
        if epoch % 1000 == 0:
            print('=> Start Evaluating on Testing set')
            with torch.no_grad():
                for similarity, feature, label in test_loader:
                    similarity, label = similarity.to(device), label.to(device)
                    if method == "ASSD":
                        feature = feature.to(device)
                        out = model(similarity, feature)
                    elif method == "FE":
                        feature = feature.to(device)
                        out = model(feature)
                    else:
                        out = model(similarity)
                    out[out > 0.5] = 1
                    out[out < 0.5] = 0
                    batch_size = len(label)
                    correct = torch.sum(out == label.data)
                    acc = correct * 100.0 / batch_size

                print(
                    "Testing set Accuracy : {acc:.4f}".format(acc=acc))


if __name__ == "__main__":
    train_dataset = AttackDataset()
    test_dataset = copy.deepcopy(train_dataset)
    test_dataset.is_train = False
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    epoches = 10001
    M_sd, M_as_sd, M_fe = Attacker(2000, 512), AnchorSelector(2000, 512, 2048), Attacker(2048, 512)
    M_sd.to(device), M_as_sd.to(device), M_fe.to(device)

    train_model(M_fe, "FE", 10001)
    train_model(M_sd, "SD", 30001)
    train_model(M_as_sd, "ASSD", 5001)
