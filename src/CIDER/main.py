'''
COVID-19 Deconfounder
By Jing Ma, 2021-05
'''

import time
import numpy as np
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import math
import argparse
import os
import sys
import scipy.io as scio
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import datetime
import network_preprocess as nets
import utils

import random
from scipy.sparse import csc_matrix
import pandas as pd
from CovidRNN import CovidRNN
import data_preprocessing as dpp

import matplotlib.font_manager
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

#matplotlib.rcParams['text.usetex'] = True  # force to generate Type 1
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


# Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--nocuda', type=int, default=0, help='Disables CUDA training.')
parser.add_argument('--type_y', type=str, default='confirmed', choices=['confirmed', 'death'])

parser.add_argument('--start_time', type=str, default='2020-01-22')  # start time: XXXX-XX-XX
parser.add_argument('--end_time', type=str, default='2020-12-31')  # start time: XXXX-XX-XX
parser.add_argument('--time_interval', type=int, default=15, help='interval between time steps (days)')

parser.add_argument('--type_net', type=str, default='dist', choices=['dist', 'mob', 'no'])
parser.add_argument('--history', type=bool, default=True, help='use historical information or not')

parser.add_argument('--beta', type=float, default= 1, help='weight of treatment prediction loss.')
parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay (L2 loss on parameters).')

parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=350,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=3e-3,
                    help='Initial learning rate.')
parser.add_argument('--h_dim', type=int, default=128,
                    help='dim of hidden units.')
parser.add_argument('--g_dim', type=int, default=10,
                    help='dim of hidden units.')
parser.add_argument('--z_dim', type=int, default=50,
                    help='dim of hidden confounders.')
parser.add_argument('--clip', type=float, default=1.,
                    help='gradient clipping')
parser.add_argument('--normy', type=int, default=1)
parser.add_argument('--normx', type=int, default=1)
parser.add_argument('--n_layers_gcn', type=int, default=1)
parser.add_argument('--n_out', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--P', type=int, default=3)
parser.add_argument('--encoder_type', type=str, default='gcn')  # gcn, gat

parser.add_argument('--wass', type=float, default=1e-4)

args = parser.parse_args()
args.cuda = not args.nocuda and torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")

print('using device: ', device)

# seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)

sys.path.append('../')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def compute_loss_y(y_pred, y_true):
    loss_mse = nn.MSELoss(reduction='mean')
    loss_y_mse = loss_mse(y_pred, y_true)
    return loss_y_mse

def compute_loss_t(t_pred, t_true):
    loss_mse = nn.BCELoss(reduction='mean')
    loss_y_mse = loss_mse(y_pred, y_true)
    return loss_y_mse

def compute_loss(y1_pred_list, y0_pred_list, rep_list, y_true, all_ps, treat_cur, idx_select, beta):
    loss_mse = nn.MSELoss(reduction='mean')
    loss_bse = nn.CrossEntropyLoss()

    T = len(treat_cur)
    y_pred = [torch.where(treat_cur[t] > 0, y1_pred_list[t], y0_pred_list[t]) for t in range(T)]

    # treatment prediction
    ps_pred = [all_ps[t].unsqueeze(0) for t in range(T)]
    ps_pred = torch.cat(ps_pred, dim=0)  # T x N x 2 (2: t= 0/1)
    C_pred = ps_pred.argmax(dim=-1).unsqueeze(-1)  # T x N x 1
    correct = (treat_cur[:, idx_select, :] == C_pred[:, idx_select, :])
    acc_ave_test = correct.sum().float() / float(treat_cur.shape[0] * len(idx_select))

    loss_y = 0.0
    loss_t = 0.0
    loss_b = 0.0
    for t in range(T):
        loss_y += loss_mse(y_pred[t][idx_select], y_true[t][idx_select])
        loss_t += loss_bse(ps_pred[t][idx_select], treat_cur[t][idx_select].reshape(-1).long())

        # balancing
        rep = rep_list[t]
        C_cur = treat_cur[t].reshape(-1)
        idx_treated = (C_cur[idx_select] > 0).nonzero().reshape(-1)
        idx_control = (C_cur[idx_select] < 1).nonzero().reshape(-1)
        if len(idx_treated) == 0 or len(idx_control) == 0:
            dist = 0.0
        else:
            rep_t1, rep_t0 = rep[idx_select][idx_treated], rep[idx_select][idx_control]
            dist, _ = utils.wasserstein(rep_t1, rep_t0, device, cuda=True)
        loss_b += dist

    loss_y /= T
    loss_t /= T
    loss_b /= T
    loss = loss_y + beta * loss_t + args.wass * loss_b
    return loss, acc_ave_test, loss_y, loss_t


def evaluate(y1_pred_list, y0_pred_list, y_true, all_ps, treat_cur, idx_select):
    # treatment prediction
    T = len(treat_cur)
    ps_pred = [all_ps[t].unsqueeze(0) for t in range(T)]  # T x N, pred_treatment
    ps_pred = torch.cat(ps_pred, dim=0)  # T x N x 2 (2: t= 0/1)
    C_pred = ps_pred.argmax(dim=-1).unsqueeze(-1)  # T x N x 1
    correct = (treat_cur[:, idx_select, :] == C_pred[:, idx_select, :])
    acc_ave_test = correct.sum().float() / float(treat_cur.shape[0] * len(idx_select))

    y_pred = [torch.where(treat_cur[t] > 0, y1_pred_list[t], y0_pred_list[t]) for t in range(T)]  # T, N x time window
    loss_y = 0.0

    loss_mse = nn.MSELoss(reduction='mean')
    for t in range(T):
        if args.normy:
            y_pred_t = y_pred[t] * (ys[t]+1) + ym[t]
            y_true_t = y_true[t] * (ys[t]+1) + ym[t]
        else:
            y_pred_t = y_pred[t]
            y_true_t = y_true[t]
        loss_y_t = loss_mse(y_pred_t[idx_select], y_true_t[idx_select])
        loss_y_t = torch.sqrt(loss_y_t)  # MSE -> RMSE
        loss_y += loss_y_t

    loss_y /= T

    return acc_ave_test, loss_y



def train(epochs, x, adj_dense_list, treat_his, treat_cur, y, y_hist, idx_trn, idx_tst, model, optimizer):
    model.train()

    x_train = x[:, idx_trn, :]
    x_test = x[:, idx_tst, :]
    treat_train = treat_his[:, idx_trn, :]
    treat_test = treat_his[:, idx_tst, :]
    treat_cur_train = treat_cur[:, idx_trn, :]
    treat_cur_test = treat_cur[:, idx_tst, :]
    y_train = y[:, idx_trn, :]
    y_test = y[:, idx_tst, :]  # time step x test size x time window
    T = len(x)

    # sparse version
    edge_index_list, edge_weight_list = utils.transfer_adj_to_sparse(adj_dense_list)

    for k in range(epochs):  # epoch
        optimizer.zero_grad()

        # forward
        y1_pred_list, y0_pred_list, all_z_t, all_ps, h = model(x, edge_index_list, treat_cur, y_hist, edge_weight_list=edge_weight_list)

        loss_train, acc_ave_train, loss_y_trn, loss_t_trn = compute_loss(y1_pred_list, y0_pred_list, all_z_t, y, all_ps, treat_cur, idx_trn, args.beta)

        loss_train.backward()
        optimizer.step()

        if k % 100 == 0:
            model.eval()
            y1_pred_list, y0_pred_list, all_z_t, all_ps, h = model(x, edge_index_list, treat_cur, y_hist, edge_weight_list=edge_weight_list)
            y_test_pred = [torch.where(treat_cur_test[t] > 0, y1_pred_list[t][idx_tst], y0_pred_list[t][idx_tst]) for t in range(T)]  # T，N x 1
            y_test_pred = [y_tst.unsqueeze(0) for y_tst in y_test_pred]
            y_test_pred = torch.cat(y_test_pred, dim=0)  # T x N x 1

            loss_test, acc_ave_test, loss_y_tst, loss_t_tst = compute_loss(y1_pred_list, y0_pred_list, all_z_t, y, all_ps, treat_cur, idx_tst, args.beta)
            acc_ave_test_eval, loss_y_test_nonorm = evaluate(y1_pred_list, y0_pred_list, y, all_ps, treat_cur, idx_tst)

            print('Epoch: {:04d}'.format(k + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'loss_test: {:.4f}'.format(loss_test.item())
                  )
            model.train()

    # f = plt.figure()
    # y_test_ave = torch.sum(torch.sum(y_test, dim=-1), dim=-1) / (y_test.shape[1] * y_test.shape[2])  # average, size= time step
    # y_test_pred_ave = torch.sum(torch.sum(y_test_pred, dim=-1), dim=-1) / (y_test_pred.shape[1] * y_test_pred.shape[2])  #
    # y_test_ave = y_test_ave.cpu().detach().numpy()
    # y_test_pred_ave = y_test_pred_ave.cpu().detach().numpy()
    # x_time = range(len(y_test_ave))
    # plt.plot(x_time, y_test_ave, 'k^-', label="true", markersize=6, markevery=1)
    # plt.plot(x_time, y_test_pred_ave, 'gD-', label="pred", markersize=6, markevery=1)
    #
    # plt.xlabel("time step", fontsize=20)
    # plt.ylabel("Y", fontsize=20)
    # plt.legend(loc='lower right', fontsize=18)
    #
    # plt.grid(linestyle=':')
    # plt.grid(axis='x')
    #
    # # font size
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    #
    # plt.show()
    #
    # print('y_test_pred_ave: ', y_test_pred_ave)
    # print('y_test_ave: ', y_test_ave)

    return


def test(x, adj_dense_list, treat_his, treat_cur, y, y_hist, idx_tst, model):
    model.eval()

    x_test = x[:, idx_tst, :]
    treat_test = treat_his[:, idx_tst, :]

    # sparse version
    edge_index_list, edge_weight_list = utils.transfer_adj_to_sparse(adj_dense_list)

    y1_pred_list, y0_pred_list, _, _, _ = model(x, edge_index_list, treat_cur, y_hist,
                                                edge_weight_list=edge_weight_list)  # T x test_size x 1

    y1_pred_list_tst = y1_pred_list[:, idx_tst, :]
    y0_pred_list_tst = y0_pred_list[:, idx_tst, :]

    T = len(treat_cur)
    for t in range(T):
        if args.normy:
            y1_pred_list_tst[t] = y1_pred_list_tst[t] * (ys[t] + 1) + ym[t]
            y0_pred_list_tst[t] = y0_pred_list_tst[t] * (ys[t] + 1) + ym[t]

    ite = y1_pred_list_tst - y0_pred_list_tst

    ave_ite_time = torch.sum(torch.sum(ite, dim=-1), dim=-1) / (ite.shape[1] * ite.shape[2])  # T

    num_pos = torch.sum(ite > 0)
    num_neg = torch.sum(ite < 0)

    # print('num_pos: ', num_pos, ' neg:', num_neg)
    # print('ave_ite_time:', ave_ite_time)

    return ave_ite_time, ite


if __name__ == '__main__':
    type_y = args.type_y

    cate_list = ['MA']  # the categories of policies to assess ['SD', 'RO', 'MA']
    policy_micro = {"SD": [
                           None,
                           "Food and Drink",
                           "Gatherings"
                        ],
                    "RO": [
                           "Entertainment",
                           "Outdoor and Recreation",
                           "Personal Care",
                           "Food and Drink"
                           ],
                    "MA": [
                        "Food and Drink",
                        "Phase 2"
                        ]
                    }

    # ==========
    path_y = '../../dataset/confirm_death_matrix_new.csv'
    path_t = '../../dataset/state_policy_updates_20201206_0721.csv'
    # path_x = '../dataset/county_trend_list.csv'
    path_trend = '../../dataset/GoogleTrend/'
    path_dist = '../../dataset/dis_ad_matrix_v3.csv'
    path_mob = '../../dataset/Population_mobility/'
    path_save = ''
    type_net = args.type_net
    countys = ['Adair,MO', 'Adams,CO', 'Adams,OH', 'Adams,PA', 'Albany,NY', 'Albemarle,VA', 'Allegan,MI',
               'Allegheny,PA', 'Allen,OH', 'Amador,CA', 'Anne Arundel,MD', 'Apache,AZ', 'Arapahoe,CO', 'Ashland,OH',
               'Atlantic,NJ', 'Atoka,OK', 'Auglaize,OH', 'Augusta,VA', 'Baldwin,AL', 'Baldwin,GA', 'Barnstable,MA',
               'Bates,MO', 'Benton,AR', 'Benton,WA', 'Bergen,NJ', 'Berkeley,SC', 'Berkeley,WV', 'Berks,PA',
               'Berrien,MI', 'Boone,MO', 'Bottineau,ND', 'Boulder,CO', 'Box Elder,UT', 'Bristol,MA', 'Broward,FL',
               'Bucks,PA', 'Burleson,TX', 'Burlington,NJ', 'Butler,KS', 'Butler,OH', 'Butler,PA', 'Cache,UT',
               'Calhoun,AL', 'Callaway,MO', 'Calvert,MD', 'Camden,NJ', 'Cameron,TX', 'Camp,TX', 'Campbell,KY',
               'Carbon,PA', 'Carroll,MD', 'Cass,MO', 'Champaign,IL', 'Charles,MD', 'Charlevoix,MI', 'Chatham,GA',
               'Cheboygan,MI', 'Chelan,WA', 'Chittenden,VT', 'Clackamas,OR', 'Clark,NV', 'Clark,WA', 'Clatsop,OR',
               'Clay,MO', 'Clay,WV', 'Clermont,OH', 'Cleveland,OK', 'Clinton,PA', 'Coal,OK', 'Cochise,AZ', 'Cocke,TN',
               'Collin,TX', 'Columbia,NY', 'Columbia,OR', 'Comanche,OK', 'Cook,IL', 'Coweta,GA', 'Crawford,AR',
               'Crockett,TX', 'Cumberland,NC', 'Cumberland,PA', 'Curry,OR', 'Custer,OK', 'Cuyahoga,OH', 'Dale,AL',
               'Dallas,TX', 'Dane,WI', 'Dauphin,PA', 'Davidson,TN', 'Davis,UT', 'Day,SD', 'Decatur,GA', 'Delaware,OH',
               'Delaware,PA', 'Denver,CO', 'Dodge,WI', 'Dougherty,GA', 'Douglas,KS', 'Douglas,NE', 'Dunn,WI',
               'East Baton Rouge,LA', 'Ellis,KS', 'Elmore,AL', 'Erath,TX', 'Erie,OH', 'Essex,MA', 'Essex,VA',
               'Fairfax,VA', 'Fairfield,CT', 'Fauquier,VA', 'Fayette,AL', 'Fayette,PA', 'Fayette,WV',
               'Floyd,VA', 'Franklin,OH', 'Franklin,TN', 'Franklin,VT', 'Frederick,MD', 'Gadsden,FL', 'Garfield,OK',
               'Gaston,NC', 'Geauga,OH', 'Genesee,MI', 'Gloucester,NJ', 'Glynn,GA', 'Grady,OK', 'Grand Isle,VT',
               'Grant,OR', 'Greenbrier,WV', 'Greene,OH', 'Greer,OK', 'Guadalupe,TX', 'Gwinnett,GA', 'Hamilton,IN',
               'Hamilton,OH', 'Hampden,MA', 'Hampshire,MA', 'Hampshire,WV', 'Harrison,KY', 'Harrison,OH', 'Hartford,CT',
               'Hawaii,HI', 'Henderson,NC', 'Hennepin,MN', 'Hernando,FL', 'Hillsborough,NH', 'Honolulu,HI',
               'Hot Spring,AR', 'Howell,MO', 'Hunt,TX', 'Hunterdon,NJ', 'Huron,OH', 'Ingham,MI', 'Iredell,NC',
               'Iron,MO', 'Izard,AR', 'Jackson,GA', 'Jackson,MO', 'Jackson,WI', 'James City,VA', 'Jefferson,AL',
               'Jefferson,CO', 'Jefferson,KY', 'Jefferson,LA', 'Jefferson,TN', 'Jefferson,WI', 'Jefferson,WV',
               'Johnson,IA', 'Johnston,NC', 'Kalamazoo,MI', 'Kanawha,WV', 'Kane,IL', 'Kauai,HI', 'Kenosha,WI',
               'Kent,DE', 'King,WA', 'Kingfisher,OK', 'Kings,NY', 'Kitsap,WA', 'Knox,ME', 'La Crosse,WI',
               'Lackawanna,PA', 'Lafayette,MO', 'Lake,OH', 'Lamoille,VT', 'Lancaster,NE', 'Lane,OR', 'Lapeer,MI',
               'Laramie,WY', 'Lauderdale,AL', 'Lee,FL', 'Lee,MS', 'Lehigh,PA', 'Lexington,SC', 'Liberty,TX',
               'Licking,OH', 'Limestone,AL', 'Limestone,TX', 'Lincoln,OK', 'Livingston,MI', 'Livingston,NY', 'Llano,TX',
               'Lonoke,AR', 'Lorain,OH', 'Los Angeles,CA', 'Loudoun,VA', 'Lucas,OH', 'Lyon,KS', 'Macomb,MI',
               'Madison,AL', 'Madison,IL', 'Madison,TX', 'Manatee,FL', 'Manitowoc,WI', 'Maricopa,AZ', 'Marin,CA',
               'Marion,IN', 'Marion,OR', 'Marion,WV', 'Marshall,KS', 'Marshall,WV', 'Mason,WA', 'Maury,TN',
               'Maverick,TX', 'Mayes,OK', 'Medina,OH', 'Mercer,PA', 'Miami,OH', 'Middlesex,MA', 'Middlesex,NJ',
               'Midland,MI', 'Millard,UT', 'Mingo,WV', 'Minnehaha,SD', 'Mississippi,AR', 'Mobile,AL', 'Mohave,AZ',
               'Monmouth,NJ', 'Monroe,NY', 'Monroe,WV', 'Montcalm,MI', 'Monterey,CA', 'Montgomery,KY', 'Montgomery,MD',
               'Montgomery,PA', 'Morgan,AL', 'Morris,NJ', 'Morton,ND', 'Multnomah,OR', 'Muskingum,OH', 'Nantucket,MA',
               'Nassau,NY', 'New Castle,DE', 'New Hanover,NC', 'New Haven,CT', 'New York,NY', 'Nez Perce,ID',
               'Nicholas,WV', 'Noble,OK', 'Norfolk,MA', 'Northampton,PA', 'Northumberland,PA', 'Nueces,TX',
               'Oakland,MI', 'Ohio,WV', 'Oklahoma,OK', 'Orange,CA', 'Orleans,LA', 'Orleans,VT', 'Osage,KS', 'Ottawa,MI',
               'Ottawa,OH', 'Outagamie,WI', 'Ozaukee,WI', 'Passaic,NJ', 'Pemiscot,MO', 'Perry,KY', 'Pierce,WA',
               'Pike,AL', 'Pike,MO', 'Pima,AZ', 'Pinellas,FL', 'Placer,CA', 'Platte,MO', 'Plymouth,MA', 'Pocahontas,WV',
               'Polk,FL', 'Pottawatomie,KS', 'Pulaski,AR', 'Pulaski,VA', 'Putnam,WV', 'Queens,NY', 'Randolph,WV',
               'Rapides,LA', 'Reno,KS', 'Rensselaer,NY', 'Richland,OH', 'Richland,SC', 'Riley,KS', 'Riverside,CA',
               'Robertson,TN', 'Rock,WI', 'Rockingham,NH', 'Rockland,NY', 'Ross,OH', 'Rowan,NC', 'Rutland,VT',
               'Sacramento,CA', 'Saline,AR', 'Salt Lake,UT', 'San Diego,CA', 'San Mateo,CA', 'San Patricio,TX',
               'Santa Clara,CA', 'Santa Rosa,FL', 'Sarasota,FL', 'Saratoga,NY', 'Sarpy,NE', 'Schuylkill,PA',
               'Scotland,MO', 'Scott,MN', 'Sedgwick,KS', 'Sequatchie,TN', 'Shawano,WI', 'Shelby,AL', 'Shelby,TN',
               'Shiawassee,MI', 'Siskiyou,CA', 'Snohomish,WA', 'Snyder,PA', 'Solano,CA', 'Somerset,NJ', 'Stark,OH',
               'Starr,TX', 'Suffolk,MA', 'Suffolk,NY', 'Summit,OH', 'Sussex,DE', 'Talladega,AL', 'Taos,NM',
               'Tarrant,TX', 'Terrebonne,LA', 'Tioga,NY', 'Tolland,CT', 'Tooele,UT', 'Traill,ND', 'Trigg,KY',
               'Trinity,CA', 'Trumbull,OH', 'Tuscarawas,OH', 'Union,NJ', 'Union,OH', 'Utah,UT', 'Van Buren,AR',
               'Venango,PA', 'Wake,NC', 'Waldo,ME', 'Walker,AL', 'Walworth,WI', 'Warren,IA', 'Wasatch,UT', 'Waseca,MN',
               'Washington,AR', 'Washington,PA', 'Washington,UT', 'Washoe,NV', 'Waukesha,WI', 'Waushara,WI', 'Wayne,MI',
               'Wayne,WV', 'Westchester,NY', 'Westmoreland,PA', 'White,AR', 'Wicomico,MD', 'Will,IL', 'Williamson,TX',
               'Windham,VT', 'Windsor,VT', 'Winnebago,IL', 'Winston,AL', 'Wise,VA', 'Wood,OH', 'Wood,WV', 'Woodford,IL',
               'Worcester,MA', 'Wyoming,WV', 'Yakima,WA', 'Yolo,CA', 'York,PA', 'York,VA', 'Yuma,AZ']

    states_distint = [c.split(',')[-1] for c in countys]
    states_distint = list(set(states_distint))
    print("number of states:", len(states_distint), ' ', states_distint)

    start_time, end_time, time_interval = args.start_time, args.end_time, args.time_interval
    time_step_list = utils.generate_time(start_time, end_time, time_interval)  # start_time, end_time, time_interval (days)

    time_num = len(time_step_list)
    # ====================  treatment: policy of interest ====================
    policy_keywords = {"SD": ["social distance", "social distancing", "gather", "remote", "close"],
                       "RO": ["reopen", "reopening", "Reopen", "Reopening"],
                       "MA": ["mask", "face covering", "Mask", "face", "Face"]}  # policies which contain at least one of the keywords belong to the corresponding category

    policy_index = ["SD", "RO", "MA"]
    mustcontain = None

    # time window: future prediction
    time_window = 1
    P = args.P

    rate_trn = 0.7
    rate_tst = 0.3

    # ====================  outcome ===================
    # stat_outcome(path_y, time_step_list=time_step_list, type_y=type_y)
    y_hist, y, y_orin = dpp.read_outcome(path_y, time_step_list=time_step_list, type_y=type_y, P=P,
                                         time_window=time_window)  # |T| x n x time window

    num_instance = y.shape[1]

    # ==================== features ======================
    feat_trend = dpp.read_features(path_trend, time_step_list, type='trend')
    x = feat_trend[P-1: -1]  # one time step before prediction
    if args.history:
        x = np.concatenate([x, y_hist], axis=2)

    # ==================== network  ======================
    assert type_net == "dist" or type_net == "mob" or type_net == 'no'
    if type_net == "dist":
        adj_dense = nets.read_network_dist(path_dist)
        adj_dense_list = [adj_dense for i in range(time_num)]
    elif type_net == "mob":
        adj_dense_list = nets.read_network_mob(path_mob, time_step_list=time_step_list)
    else:
        adj_dense_list = [np.eye(x.shape[1]) for i in range(time_num)]
    adj_dense_list = adj_dense_list[P - 1: -1]

    # train/test
    idx_all = np.arange(num_instance)
    np.random.shuffle(idx_all)
    idx_trn = idx_all[:int(rate_trn * num_instance)]
    idx_tst = idx_all[int(rate_trn * num_instance):]

    # torch
    y_hist = torch.FloatTensor(y_hist)
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)
    adj_dense_list = [torch.FloatTensor(adj) for adj in adj_dense_list]
    idx_trn = torch.LongTensor(idx_trn)
    idx_tst = torch.LongTensor(idx_tst)

    # normx
    if args.normx:
        xm, xs = torch.mean(x[:, idx_trn, :], dim=1), torch.std(x[:, idx_trn, :], dim=1)  # T x d
        for t in range(x.shape[0]):  # T
            x[t] = (x[t] - xm[t]) / (xs[t] + 1)  # x[t]: N x d

    # norm y: N(0,1)
    if args.normy:
        ym, ys = torch.mean(y[:, idx_trn, :], dim=1), torch.std(y[:, idx_trn, :], dim=1)
        for t in range(y.shape[0]):  # T
            y[t] = (y[t] - ym[t]) / (ys[t] + 1)

    x_dim = x.shape[2]
    y_dim = y.shape[2]
    t_dim = 1
    y_hist_dim = y_hist.shape[2]

    cuda = True
    if cuda:
        y_hist = y_hist.to(device)
        x = x.to(device)
        y = y.to(device)
        adj_dense_list = [adj.to(device) for adj in adj_dense_list]

        idx_trn = idx_trn.to(device)
        idx_tst = idx_tst.to(device)

        if args.normy:
            ys = ys.to(device)
            ym = ym.to(device)

    # ================= treatment =========================
    # treat_orin: category-level treatment,
    # policy_assign_all: policy type-level treatment,
    # policy_names_all: policy type names (top k per category)
    treat_orin, policy_assign_all, policy_names_all = dpp.top_policy(path_t, time_step_list=time_step_list, level='state',
                                                                policy_keywords=policy_keywords,
                                                                policy_index=policy_index, countys=countys,
                                                                mustcontain=mustcontain)

    results_dict = {}
    ite_results_dict = {}
    for policy_cate_interest in cate_list:  # each category
        treat = treat_orin.copy()

        results_dict[policy_cate_interest] = {}
        ite_results_dict[policy_cate_interest] = {}
        for policy_spe_interest in policy_micro[policy_cate_interest]:  # each policy type
            if policy_spe_interest is None:
                policy_select = policy_index.index(policy_cate_interest)
                treat = treat[:, :, policy_select]
            else:
                spe_idx = policy_names_all[policy_cate_interest].index(policy_spe_interest)
                treat = policy_assign_all[policy_cate_interest][spe_idx]
                treat = treat.swapaxes(0, 1)

            treat_his = np.array([treat[:, i:i + P] for i in range(time_num - time_window - P + 1)])
            treat_cur = treat[:, P:]  # N x T
            treat_cur = np.swapaxes(treat_cur, 0, 1)
            treat_cur = treat_cur.reshape(treat_cur.shape[0], treat_cur.shape[1], 1)

            treat_his = torch.FloatTensor(treat_his)
            treat_cur = torch.FloatTensor(treat_cur)

            model = CovidRNN(x_dim, y_dim, t_dim, y_hist_dim, args, adj_dense_list)
            # for param in model.gc[0].parameters():
            #     print(type(param.data), param.size())
            par_g = []
            for t in range(len(model.gc)):
                par_g = par_g + list(model.gc[t].parameters())
            par_m = list(model.parameters())
            optimizer = torch.optim.Adam(par_m + par_g, lr=args.lr, weight_decay=args.weight_decay)

            if cuda:
                treat_his = treat_his.to(device)
                treat_cur = treat_cur.to(device)
                model = model.to(device)

            train(args.epochs, x, adj_dense_list, treat_his, treat_cur, y, y_hist, idx_trn, idx_tst, model, optimizer)
            print('===============  ATE on test data: =================')
            ave_ite_time_tst, ite_time_tst = test(x, adj_dense_list, treat_his, treat_cur, y, y_hist, idx_tst, model)
            print('===============  ATE on all data: =================')
            ave_ite_time_all, ite_time_all = test(x, adj_dense_list, treat_his, treat_cur, y, y_hist, idx_all, model)

            if policy_spe_interest is None:
                print("Average cause effect:", policy_cate_interest + ',\t' + 'all' + ',TEST: ',
                      ave_ite_time_tst, 'ALL: ', ave_ite_time_all)
            else:
                print("Average cause effect:", policy_cate_interest + ',\t' + policy_spe_interest + ',TEST: ', ave_ite_time_tst, 'ALL: ', ave_ite_time_all)

            if policy_spe_interest is None:  # category-level
                results_dict[policy_cate_interest]["all"] = {  # ATE
                    'alldata': ave_ite_time_all.cpu().detach().numpy().copy(),  # all data
                    'test': ave_ite_time_tst.cpu().detach().numpy().copy()  # test data
                }
                ite_results_dict[policy_cate_interest]["all"] = {  # ITE
                    'alldata': ite_time_all.cpu().detach().numpy().copy(),
                    'test': ite_time_tst.cpu().detach().numpy().copy()
                }
            else:  # policy type-level
                results_dict[policy_cate_interest][policy_spe_interest] = {
                    'alldata': ave_ite_time_all.cpu().detach().numpy().copy(),
                    'test': ave_ite_time_tst.cpu().detach().numpy().copy()
                }
                ite_results_dict[policy_cate_interest][policy_spe_interest] = {
                    'alldata': ite_time_all.cpu().detach().numpy().copy(),
                    'test': ite_time_tst.cpu().detach().numpy().copy()
                }

    #np.savez('Effect_'+type_y+".npy", ate_dic=results_dict, ite_dic=ite_results_dict)
        #print('policy_cate_interest')


