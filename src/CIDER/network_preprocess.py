import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import utils
import pickle
from scipy import sparse

def read_network_dist(path):
    csv_data = pd.read_csv(path)
    column_select = range(1, 391)
    column_select = [str(c) for c in column_select]
    csv_weight = csv_data.loc[:, "0": ]
    csv_weight = np.array(csv_weight, dtype=np.float64)

    # stats:
    # sep = 5
    # for i in range(1, 80, sep):
    #     a = ((csv_weight > i) & (csv_weight <= i+sep)).sum()
    #     print("from ", i, " to ", i+sep, ": ", a)

    max_filter = 100  # distance >= max_filter will be filtered out
    tau = 1.0
    csv_weight_filter = csv_weight.copy()
    csv_weight_filter[np.where(csv_weight >= max_filter)] = 0.0
    csv_weight_filter[np.where(csv_weight < max_filter)] = tau / (csv_weight_filter[np.where(csv_weight < max_filter)] + 1)  # + 1
    csv_weight_filter[np.where(np.eye(391) == 1.0)] = 1.0

    return csv_weight_filter

def preprocess_mob(path, time_step_list, max_thresh=0.3, filter_thresh = 0.3, save=True):
    import os
    from scipy.sparse import csr_matrix
    networks = []
    network_cur = []

    # filter out small flow values, if filter_thresh = 0, remove all i!=j; if filter_thresh = 1, keep all entries

    # empty networks
    date_mob_start = datetime.datetime.strptime('2020-02-01', '%Y-%m-%d')
    date_mob_end = datetime.datetime.strptime('2020-10-31', '%Y-%m-%d')
    for ti in range(len(time_step_list)):
        start, end = time_step_list[ti]
        if end < date_mob_start:
            networks.append(np.eye(391))
        else:
            break

    # mobility flow
    for i in range(2, 11):
        path_m = path + str(i) + '/'
        files = os.listdir(path_m)
        files.sort()
        for file in files:
            file_name = str(file)
            if file_name[:5] != 'daily':
                continue
            print('reading file: ', path_m+file_name)
            content = np.load(path_m+file_name)
            indptr, indices, data, shape = content['indptr'], content['indices'], content['data'], content['shape']
            net = csr_matrix((data, indices, indptr), shape=shape).toarray()  # 391 x 391, int
            net[net == -1] = 0.0

            # normalization
            net = np.log(net + 1)  # 0, inf
            np.fill_diagonal(net, 0)
            max_value = np.max(net)  # max value of flow i,j  (i!=j)
            net = net / max_value  # norm by day, norm_flow = log(flow+1) / max log (flow+1)
            net *= max_thresh  # max_thresh of flows (0, max_thresh)
            # filter out
            net_sparse = sparse.csr_matrix(net)
            edge_weight = net_sparse.data
            edge_weight = np.sort(edge_weight)[::-1]  # 41178
            filter_value_thresh = edge_weight[int(filter_thresh * (len(edge_weight)-1))]  # value < filter_value_thresh will be filtered out
            net[net <= filter_value_thresh] = 0.0
            np.fill_diagonal(net, 1)

            # aggregation with time steps
            date_now = '2020-' + file_name[-5-8:-3-8] + '-' + file_name[-2-8:-8]
            date_now = datetime.datetime.strptime(date_now, '%Y-%m-%d')
            network_cur.append(np.expand_dims(net, axis=0))  # 1 x n x n
            if date_now == end or file_name[-5-8:-8] == '10_31':  # it is time to aggregate
                ti += 1
                assert ti < len(time_step_list)
                start, end = time_step_list[ti]

                network_agg = np.concatenate(network_cur, axis=0)
                network_agg = np.mean(network_agg, axis=0)
                networks.append(network_agg)
                network_cur = []

    # end time, empty networks
    while ti < len(time_step_list):
        start, end = time_step_list[ti]
        if start > date_mob_end:
            networks.append(np.eye(391))
        ti += 1

    # write in file
    if save:
        with open(path+'mob_15days.pickle', 'wb') as handle:
            pickle.dump({'adj_list': networks}, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('saved file: ', path+'mob_15days.pickle')
    return networks

def read_network_mob(path, time_step_list=None, type='new'):
    # net_data = sio.loadmat(path + '_15days' + '.mat')
    # A = net_data['A'][0]
    if type == 'load':
        with open(path + 'mob_15days.pickle', 'rb') as handle:
            adj_list = pickle.load(handle)['adj_list']
    else:
        max_thresh = 0.5
        filter_thresh = 0.2
        adj_list = preprocess_mob(path, time_step_list, max_thresh, filter_thresh, save=False)

    return adj_list


if __name__ == '__main__':
    # path = '../dataset/Population mobility across counties/'
    path = '../../dataset/Population_mobility/'

    time_step_list = utils.generate_time("2020-01-22", "2020-12-31", 15)
    preprocess_mob(path, time_step_list)
    # read_network_mob(path)