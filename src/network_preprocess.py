import pandas as pd
import numpy as np
import datetime
from scipy import sparse as sp
import scipy.io as sio
import matplotlib.pyplot as plt

def read_network_dist(path):
    csv_data = pd.read_csv(path)
    column_select = range(1, 391)
    column_select = [str(c) for c in column_select]
    csv_weight = csv_data.loc[:, "0": ]
    csv_weight = np.array(csv_weight, dtype=np.float64)

    # stats:
    sep = 50
    for i in range(1, 500, sep):
        a = ((csv_weight > i) & (csv_weight <= i+sep)).sum()
        print("from ", i, " to ", i+sep, ": ", a)

    max_filter = 1
    tau = 1.0
    csv_weight_filter = csv_weight.copy()
    csv_weight_filter[np.where(csv_weight >= max_filter)] = 0.0
    csv_weight_filter[np.where(csv_weight < max_filter)] = tau / (csv_weight_filter[np.where(csv_weight < max_filter)] + 1)  # + 1
    csv_weight_filter[np.where(np.eye(391) == 1.0)] = 1.0

    return csv_weight_filter

def read_network_mob(path, time_step_list):
    net_data = sio.loadmat(path + '_15days' + '.mat')
    A = net_data['A'][0]

    return A


if __name__ == '__main__':
    path = '../dataset/Population mobility across counties/'
