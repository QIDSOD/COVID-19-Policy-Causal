import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import reset
from torch_geometric.nn import GCNConv, GATConv
from torch.autograd import Variable, Function
from torch.nn.utils import spectral_norm
import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CovidRNN(nn.Module):
    def __init__(self, x_dim, y_dim, t_dim, y_hist_dim, args, adj_dense_list):
        super(CovidRNN, self).__init__()

        self.x_dim = x_dim  # feature dim
        self.h_dim = args.h_dim
        self.z_dim = args.z_dim
        self.y_dim = y_dim
        self.n_layers_gcn = args.n_layers_gcn
        self.n_out = args.n_out
        self.dropout = args.dropout
        self.P = args.P
        self.t_dim = t_dim
        self.g_dim = args.g_dim
        self.y_hist_dim = y_hist_dim
        self.type_net = args.type_net
        self.history = args.history
        self.encoder_type = args.encoder_type
        self.skip_connect = True

        self.phi_x = nn.Sequential(nn.Linear(x_dim, self.h_dim).to(device), nn.ReLU().to(device))

        # sparse version
        _, edge_weight_list = utils.transfer_adj_to_sparse(adj_dense_list)
        self.gc = [Encoder(self.h_dim, self.g_dim, base_model=self.encoder_type, edge_weight=edge_weight_list[t]).to(device) for t in range(len(adj_dense_list))]

        if self.skip_connect:
            self.fuse = nn.Sequential(nn.Linear(self.h_dim + self.h_dim + self.g_dim, self.z_dim).to(device), nn.ReLU().to(device))  # graph, hist, phi_x
        else:
            self.fuse = nn.Sequential(nn.Linear(self.g_dim + self.h_dim, self.z_dim).to(device), nn.ReLU().to(device))  # graph, hist

        # prediction
        # potential outcome
        self.out_t00 = [nn.Linear(self.z_dim, self.z_dim).to(device) for i in range(self.n_out)]
        self.out_t10 = [nn.Linear(self.z_dim, self.z_dim).to(device) for i in range(self.n_out)]
        self.out_t01 = nn.Linear(self.z_dim, 1).to(device)
        self.out_t11 = nn.Linear(self.z_dim, 1).to(device)

        # propensity score
        self.ps_predictor = nn.Sequential()
        self.ps_predictor.add_module('d_fc1', nn.Linear(self.z_dim, 100).to(device))
        self.ps_predictor.add_module('d_bn1', nn.BatchNorm1d(100).to(device))
        self.ps_predictor.add_module('d_sigmoid1', nn.Sigmoid().to(device))
        self.ps_predictor.add_module('d_fc2', nn.Linear(100, 2).to(device))
        self.ps_predictor.add_module('d_softmax', nn.Softmax(dim=1).to(device))

        # memory unit
        self.rnn = nn.GRUCell(self.z_dim + self.t_dim + self.y_hist_dim, self.h_dim).to(device)  # c_t, z_t, h_{t-1} => h_t


    def forward(self, X_list, edge_index_list, C_list, Y_hist_list, hidden_in=None, edge_weight_list=None):
        '''
        :param X_list:  list of torch.FloatTensor
        :param A_list:  list of torch.sparse.FloatTensor
        :param C_list:  list of torch.FloatTensor
        :param hidden_in:
        :return:
        '''
        all_z_t = []
        all_y1 = []
        all_y0 = []
        all_ps = []

        num_timestep = len(X_list)
        num_node = X_list[-1].size(0)

        if hidden_in is None:
            h = Variable(torch.zeros(num_node, self.h_dim))
        else:
            h = Variable(hidden_in)

        h = h.to(device)

        for t in range(num_timestep):  # time step
            C = C_list[t]
            x = X_list[t]
            edge_index = edge_index_list[t]
            edge_weight = None if edge_weight_list is None else edge_weight_list[t]
            y_hist = Y_hist_list[t]

            phi_x_t = self.phi_x(x)

            # graph
            rep = F.relu(self.gc[t](phi_x_t, edge_index, edge_weight))
            rep = F.dropout(rep, self.dropout, training=self.training)

            if self.skip_connect:
                z_t = self.fuse(torch.cat([h, rep, phi_x_t], 1))
            else:
                z_t = self.fuse(torch.cat([h, rep], 1))

            # C
            C_float = C.type(torch.FloatTensor).view(-1, self.t_dim)
            C_float = C_float.to(device)

            # RNN
            if self.history:
                h = self.rnn(torch.cat([z_t, C_float, y_hist], 1), h)

            for i in range(self.n_out):
                y00 = F.relu(self.out_t00[i](z_t))
                y00 = F.dropout(y00, self.dropout, training=self.training)
                y10 = F.relu(self.out_t10[i](z_t))
                y10 = F.dropout(y10, self.dropout, training=self.training)
            if self.n_out <= 0:
                y00 = z_t
                y10 = z_t

            y0 = self.out_t01(y00).view(-1,1)
            y1 = self.out_t11(y10).view(-1,1)

            # treatment prediction
            ps_hat = self.ps_predictor(z_t)  # [:,1]

            # record
            all_z_t.append(z_t)
            all_y1.append(y1.unsqueeze(0))
            all_y0.append(y0.unsqueeze(0))
            all_ps.append(ps_hat)

        # transform to tensors
        all_y1 = torch.cat(all_y1, dim=0)
        all_y0 = torch.cat(all_y0, dim=0)

        return all_y1, all_y0, all_z_t, all_ps, h


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_model='gcn', edge_weight=None):
        super(Encoder, self).__init__()
        self.base_model = base_model
        self.conv = GCN(in_channels, out_channels, edge_weight)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight=None):
        x = self.conv(x, edge_index, edge_weight)
        return x


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, edge_weight=None):
        super(GCN, self).__init__()
        self.edge_wt = None if edge_weight is None else torch.nn.Parameter(torch.ones_like(edge_weight).to(device))
        self.gc1 = spectral_norm(GCNConv(nfeat, nhid))

    def forward(self, x, edge_index, edge_weight=None):
        x = self.gc1(x, edge_index, self.edge_wt.sigmoid())
        return x

# class GAT(nn.Module):
#     def __init__(self, nfeat, nhid):
#         super(GAT, self).__init__()
#         # self.gc1 = spectral_norm(GATConv(nfeat, nhid))
#         self.gc1 = GATConv(nfeat, nhid)
#
#     def forward(self, x, edge_index, edge_attr=None):  # edge_attr: E x 1
#         x = self.gc1(x, edge_index)  # edge_attr
#         return x