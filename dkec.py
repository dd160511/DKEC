from __future__ import print_function, division
import argparse
import random
import sqlite3
from multiprocessing import Pool
from time import time
from GNN import GNNLayer
import numpy as np
import sklearn
from scipy.sparse import csgraph
from ranger import Ranger  # this is from ranger.py # this is from ranger.py
from sklearn import cluster
from sklearn.cluster import KMeans, spectral_clustering
#from kmeans_pytorch import kmeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score, pairwise_kernels
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from crsc_ik import crsc_ik, kernel_kmeans
from utils import load_data, load_graph
from evaluation import eva
import math
from collections import Counter
from sklearn.decomposition import TruncatedSVD
import torch
import math
import gpytorch
from gpytorch import kernels, means, models, mlls, settings
from gpytorch import distributions as distr
from compute_kernels import compute_kernels, compute_kernelst, compute_kernelsp, compute_kernelsg


class AE(nn.Module):
    # 初始化
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    # 反向传播
    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z


class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    def __init__(self, num_dim, grid_bounds=(-10., 10.), grid_size=64):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size, batch_shape=torch.Size([num_dim])
        )
        # Our base variational strategy is a GridInterpolationVariationalStrategy,
        # which places variational inducing points on a Grid
        # We wrap it with a IndependentMultitaskVariationalStrategy so that our output is a vector-valued GP
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.GridInterpolationVariationalStrategy(
                self, grid_size=grid_size, grid_bounds=[grid_bounds],
                variational_distribution=variational_distribution,
            ), num_tasks=num_dim,
        )
        super().__init__(variational_strategy)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel=gpytorch.kernels.RBFKernel(ard_num_dims=num_dim,
                                                   lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                                                       math.exp(-1), math.exp(1), sigma=2, transform=torch.exp
                                                   )
                                                   )
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class KAE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, args):
        super(KAE, self).__init__()
        n_input = args.n_input
        n_z = args.data_num
        data_num = args.data_num
        n_clusters = args.n_clusters
        self.args = args
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        # self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))
        self.e1 = Linear(n_input, 500)
        self.e2 = Linear(500, 500)
        self.e3 = Linear(data_num, data_num)
        self.e4 = Linear(data_num, data_num)
        self.gnn_1 = GNNLayer(n_input, data_num)

        self.g1 = Linear(500, n_input)
        self.g2 = Linear(500, 500)
        self.g3 = Linear(500, 500)
        self.g4 = Linear(data_num, n_input)

        self.d1 = Linear(n_input, 64)
        self.d2 = Linear(64, 16)
        self.d3 = Linear(16, 1)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(args.n_clusters, 500))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.gp_layer = GaussianProcessLayer(num_dim=500, grid_bounds=(-50., 50.), grid_size=128)

    def forward(self, x):
        z1 = torch.tanh(self.e1(x))
        # z = z1.transpose(-1, -2).unsqueeze(-1)
        # z = self.gp_layer(z)
        # z = z.mean
        z = torch.tanh(self.e2(z1))

        zbar = torch.tanh(self.g2(z))
        mu = torch.tanh(self.g1(zbar))

        # D
        fake = torch.tanh(self.d1(mu))
        fake = torch.tanh(self.d2(fake))
        fake = (self.d3(fake))

        real = torch.tanh(self.d1(x))
        real = torch.tanh(self.d2(real))
        real = (self.d3(real))

        #  Self-supervised Module
        p = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / (1))
        p = p.pow((1 + 1.0) / 2.0)
        p = (p.t() / torch.sum(p, 1)).t()

        q = 1.0 / (1.0 + torch.sum(torch.pow(zbar.unsqueeze(1) - self.cluster_layer, 2), 2) / (1))
        q = q.pow((1 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return mu, z, p, q, real, fake, zbar, z1


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_sdcn(runtime, dataset, datasetname, device):
    model = KAE(500, 500, 2000, 2000, 500, 500, args=args).to(device)
    # model = KAE(args.data_num, args.data_num, args.data_num, args.data_num, args.data_num, args.data_num, args=args).to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)
    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    # KNN Graph
    # adj = load_graph(args.name, args.k)
    # adj = adj.cuda()

    y = dataset.y
    with torch.no_grad():
        mu, z, p, q, real, fake, zbar, z1 = model(data)
    kmeans1 = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans1.fit_predict(z.detach().cpu().numpy())
    # y_pred, dist = kernel_kmeans(k.detach().cpu().numpy(), args.n_clusters)
    model.cluster_layer.data = torch.tensor(kmeans1.cluster_centers_).to(device)

    eva(y, y_pred, 'pae')

    for epoch in range(200):
        if epoch % 1 == 0:
            # update_interval
            mu, z, p, q, real, fake, zbar, z1 = model(data)
            p = p.data
            s = target_distribution(p)

            res1 = p.cpu().numpy().argmax(1)  # Q
            res2 = s.cpu().numpy().argmax(1)  # S
            # 两种K-means方法，用适合的。
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
            y_pred1 = kmeans.fit_predict(z.data.cpu().numpy())
            # y_pred1,_=kmeans(z, num_clusters=args.n_clusters, tqdm_flag=False,device=device)
            # y_pred1=y_pred1.data.cpu().numpy()

            pdatas = { 'z': z}
            eva(y, y_pred1, pdatas, str(epoch) + 'Q', runtime, datasetname)
        # eva(y, res1, str(epoch) +'Q', runtime, datasetname)

        ## adversarial loss
        real_adv_loss = torch.nn.ReLU()(((1.0 - real)).mean())
        fake_adv_loss = torch.nn.ReLU()(((0 + fake)).mean())

        kl_loss = 1 * F.kl_div(p.log(), s, reduction='batchmean')
        loss = 1 * F.mse_loss(mu, data) + 0.001 * real_adv_loss + 0.001 * fake_adv_loss + 0.1 * F.mse_loss(z,
                                                                                                           zbar) + 0.1 * F.kl_div(
            q.log(), s, reduction='batchmean') + 1 * kl_loss


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



if __name__ == "__main__":
    con = sqlite3.connect("result.db")
    cur = con.cursor()
    cur.execute("delete from sdcn")
    con.commit()
    startdate = time()
    datasets = ['bbcsport']

    for dataset in datasets:
        batch = 1  # 运行轮次
        parser = argparse.ArgumentParser(
            description='train',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--name', type=str, default=dataset)
        parser.add_argument('--lr', type=float, default=5e-3)
        parser.add_argument('--n_clusters', default=3, type=int)
        parser.add_argument('--n_z', default=10, type=int)
        parser.add_argument('--data_num', default=20000, type=int)
        parser.add_argument('--pretrain_path', type=str, default='pkl')
        args = parser.parse_args()
        args.cuda = torch.cuda.is_available()
        print("use cuda: {}".format(args.cuda))
        device = torch.device("cuda" if args.cuda else "cpu")

        args.pretrain_path = 'SDCN-master/data/{}.pkl'.format(args.name)
        dataset = load_data(args.name)

        if args.name == 'usps':
            args.n_clusters = 10
            args.n_input = 256
        if args.name == 'StackOverflow':
            args.n_input = 5236
            args.k = 30
            args.n_clusters = 20
            args.input_dim = 5236
        if args.name == 'abstract':
            args.k = 3
            args.n_clusters = 3
            args.n_input = 10000
            args.data_num = 4306
        if args.name == 'pubmed':
            args.n_clusters = 3
            args.n_input = 500
            args.data_num = 19717
        if args.name == 'face':
            args.k = 3
            args.n_clusters = 20
            args.n_input = 27
            args.data_num = 624

        if args.name == 'reut':
            args.n_clusters = 4
            args.n_input = 2000
            args.data_num = 10000

        if args.name == 'acm':
            args.k = None
            args.n_clusters = 3
            args.n_input = 1870
            args.data_num = 3025
        if args.name == 'wiki':
            args.k = None
            args.n_clusters = 17
            args.n_input = 4973
            args.data_num = 2405
        if args.name == 'dblp':
            args.k = None
            args.n_clusters = 4
            args.n_input = 334
            args.data_num = 4057
        if args.name == 'cora':
            args.k = None
            args.n_clusters = 7
            args.n_input = 1433
            args.data_num = 2708
        if args.name == 'cite':
            args.k = None
            args.n_clusters = 6
            args.n_input = 3703
            args.data_num = 3327
        if args.name == 'bbc':
            args.k = 3
            args.n_clusters = 5
            args.n_input = 9635
            args.data_num = 2225
        if args.name == '20news_train':
            args.n_input = 2000
            args.clusters = 20
        if args.name == 'bbcsport':
            args.n_clusters = 5
            args.n_input = 4613
            args.data_num = 737
        if args.name == '20news_test':
            args.n_clusters = 20
            args.n_input = 2000
            args.data_num = 7505

        if args.name == '20news':
            args.n_clusters = 20
            args.n_input = 1000
            args.data_num = 7025
        if args.name == '20news_test_5':
            args.n_clusters = 5
            args.n_input = 2000
            args.data_num = 1898
        if args.name == 'rcv':
            args.k = None
            args.n_clusters = 4
            args.n_input = 5
            args.data_num = 10000
        print(args)
        for i in range(batch):
            cur.execute("delete from sdcn where datasetname=? and batch=?", [args.name, i])
            con.commit()
            train_sdcn(i, dataset, args.name, device)
            torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    enddate = time()
    print("{} run time:{}".format(dataset, enddate - startdate))
    # _datasets = ['bbc', 'bbcsport']
    _datasets = ['bbc', 'acm', 'abstract', 'bbcsport', 'wiki', 'rcv', 'cite', '20news', 'face', 'StackOverflow']
    for name in _datasets:
        datas = cur.execute(
            "select datasetname,batch,epoch,acc,nmi,ari,f1 from sdcn where datasetname=? order by batch",
            [name]).fetchall()
        for d in datas:
            if d is not None:
                print('dataname:{0},batch:{1},epoch:{2}'.format(d[0], d[1], d[2]), 'acc {:.4f}'.format(d[3]),
                      ', nmi {:.4f}'.format(d[4]), ', ari {:.4f}'.format(d[5]),
                      ', f1 {:.4f}'.format(d[6]))
    for name in _datasets:
        result = cur.execute(
            "select  avg(acc) as acc,avg(nmi) as nmi,avg(ari) as ari,avg(f1) as f1 from ( select acc,nmi,ari,f1 from sdcn where datasetname =? order by nmi desc limit 10)",
            [name]).fetchone()

        if result[0] is not None:
            print('dataname:{0}'.format(name), 'AVG :acc {:.4f}'.format(result[0]),
                  ', nmi {:.4f}'.format(result[1]), ', ari {:.4f}'.format(result[2]),
                  ', f1 {:.4f}'.format(result[3]))
    cur.close()
    con.close()
