import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import torch


def read_features(path, time_step_list, type='trend', countys=None):
    features = []  # T x n x d
    if type == 'trend':
        prefix = 'county_topic_ts_'
        postfix = '_final.npy'
        data_all = None
        for i in range(2, 13):
            path_cur = path + prefix + str(i) + postfix
            data = np.load(path_cur)
            if data.shape[0] == 392:
                data = np.delete(data, 108, axis=0)
            data_all = data if data_all is None else np.concatenate([data_all, data], axis=-1)

        n = data_all.shape[0]
        num_words = data_all.shape[1]

        idx_cur = 0
        start_date = '2020-02-01'
        end_date = '2020-12-31'
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        for i in range(len(time_step_list)):
            time_begin = time_step_list[i][0]
            time_end = time_step_list[i][1]

            if time_end < start_date:  # before Feb 1
                data_cur = np.zeros((n, num_words))
                features.append(data_cur)
                continue

            if time_begin < start_date:  #
                interval_days = time_end - start_date
                interval_days = interval_days.days
            else:
                interval_days = time_end - time_begin
                interval_days = interval_days.days

            data_cur = data_all[:, :, idx_cur: idx_cur + interval_days]
            data_cur = np.mean(data_cur, axis=-1)
            idx_cur = idx_cur + interval_days
            features.append(data_cur)

    features = np.array(features)
    return features

def read_outcome(path, time_step_list=None, type_y='death', P=2, time_window=2):
    csv_y_all = pd.read_csv(path)
    column_names = csv_y_all.columns.values

    y = None  # N x time steps

    for i in range(len(time_step_list)):
        time_step_start = time_step_list[i][0]
        time_step_end = time_step_list[i][1]

        start_str = str(time_step_start.month) + '/' + str(time_step_start.day) + '/' + str(time_step_start.year)[-2:]
        end_str = str(time_step_end.month) + '/' + str(time_step_end.day) + '/' + str(time_step_end.year)[-2:] + '.1'

        csv_y = csv_y_all.loc[:, start_str:end_str]
        col_num = csv_y.shape[1]

        if type_y == 'death':
            col_index = [2 * i + 1 for i in range(int(col_num / 2))]
        else:
            col_index = [2 * i for i in range(int(col_num / 2))]
        csv_y = np.array(csv_y, dtype=np.float64)
        csv_y = csv_y[:, col_index]
        csv_y = np.mean(csv_y, axis=1)  # average y

        csv_y = csv_y.reshape((-1,1))

        y = csv_y if y is None else np.concatenate([y, csv_y], axis=1)

    y_orin = y.copy()

    y_norm = y.copy()

    y = y_norm

    time_num = len(time_step_list)
    instance_num = len(y)

    # csv_y_norm
    y_past = np.array([y[:, i:i+P] for i in range(time_num - time_window - P + 1)])
    y_future = np.array([y[:, i+P: i+P+time_window] for i in range(time_num - time_window - P + 1)])

    return y_past, y_future, y_orin

def stat_outcome(path, time_step_list=None, type_y='death'):
    csv_y_all = pd.read_csv(path)
    column_names = csv_y_all.columns.values

    y = None  # N x time steps

    for i in range(len(time_step_list)):
        time_step_start = time_step_list[i][0]
        time_step_end = time_step_list[i][1]

        start_str = str(time_step_start.month) + '/' + str(time_step_start.day) + '/' + str(time_step_start.year)[-2:]
        end_str = str(time_step_end.month) + '/' + str(time_step_end.day) + '/' + str(time_step_end.year)[-2:] + '.1'

        csv_y = csv_y_all.loc[:, start_str:end_str]
        col_num = csv_y.shape[1]

        if type_y == 'death':
            col_index = [2 * i + 1 for i in range(int(col_num / 2))]
        else:
            col_index = [2 * i for i in range(int(col_num / 2))]
        csv_y = np.array(csv_y, dtype=np.float64)
        csv_y = csv_y[:, col_index]
        csv_y = np.mean(csv_y, axis=1)  # average y

        csv_y = csv_y.reshape((-1, 1))

        y = csv_y if y is None else np.concatenate([y, csv_y], axis=1)


    time_num = len(time_step_list)
    instance_num = len(y)

    # draw figure
    # fig 1. x axis: time, y axis: true
    f = plt.figure()

    y_ave = np.average(y, axis=0)
    x_time = range(len(y_ave))
    plt.plot(x_time, y_ave, 'k^-', label=type_y, markersize=6, markevery=1)

    plt.xlabel("time step", fontsize=20)
    plt.ylabel("Y", fontsize=20)
    plt.legend(loc='lower right', fontsize=18)

    # tick_params(which='both', direction='in')
    plt.grid(linestyle=':')
    plt.grid(axis='x')

    # font size
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # plt.title("")
    # plt.savefig('5methods_600.png')
    plt.show()


def stats_policy(treat_assign, policy_index, y):
    '''
    treat_assign: N x T x Category of polices
    policy_index: ['SD', 'RO', ..]
    y: N x T
    :return: statistics of policies
    '''
    # statistics: for each category, show the averaged treated/control outcomes over time
    T = treat_assign.shape[1]
    for pi in range(len(policy_index)):
        y_treated = []  # T
        y_control = []
        for i in range(T):
            idx_treated = np.where(treat_assign[:, i, pi] == 1)
            idx_control = np.where(treat_assign[:, i, pi] == 0)

            y_treated_i = y[idx_treated, i].reshape(1, -1)  # 1 x |treat|
            y_control_i = y[idx_control, i].reshape(1, -1)  # 1 x |control|

            y_treated_i = np.average(y_treated_i, axis=-1)  # 1
            y_treated.append(y_treated_i)
            y_control_i = np.average(y_control_i, axis=-1)  # 1
            y_control.append(y_control_i)

        f = plt.figure()

        x_time = range(T)
        plt.plot(x_time, y_treated, 'g^-', label='treated', markersize=6, markevery=1)
        plt.plot(x_time, y_control, 'k^-', label='control', markersize=6, markevery=1)

        plt.xlabel("time step", fontsize=20)
        plt.ylabel("Y", fontsize=20)
        plt.legend(loc='lower right', fontsize=18)

        # tick_params(which='both', direction='in')
        plt.grid(linestyle=':')
        plt.grid(axis='x')
        plt.title(policy_index[pi])

        # font size
        plt.xticks(range(T), fontsize=14)
        plt.yticks(fontsize=20)

        # plt.title("")
        # plt.savefig('5methods_600.png')
        plt.show()

        ate = [float(y_treated[i] - y_control[i]) for i in range(len(y_treated))]
        print(policy_index[pi], ": ", ate)
    return

def stats_DID(treat, outcomes):
    '''
    :param policy:  # |T| x n x time window
    :param outcomes: # |T| x n x time window
    :return:
    '''
    y11_list = []  # controlled, time=0
    y12_list = []  # controlled, time=1
    y21_list = []  # treated, time=0
    y22_list = []  # treated, time=1

    effect_all = []

    for t in range(len(outcomes)-1):
        idx_treated = np.where(treat[t].reshape(-1) == 1)
        idx_control = np.where(treat[t].reshape(-1) == 0)

        y_treated_begin = outcomes[t, idx_treated].reshape(1, -1)  # 1 x |treat|
        y_treated_end = outcomes[t+1, idx_treated].reshape(1, -1)  # 1 x |treat|
        y_control_begin = outcomes[t, idx_control].reshape(1, -1)  # 1 x |control|
        y_control_end = outcomes[t+1, idx_control].reshape(1, -1)

        # y_treated_begin = np.average(y_treated_begin, axis=-1)  # 1
        # y_treated_end = np.average(y_treated_end, axis=-1)  # 1
        # y_control_begin = np.average(y_control_begin, axis=-1)  # 1
        # y_control_end = np.average(y_control_end, axis=-1)  # 1

        y_treated_begin = torch.mean(y_treated_begin, dim=-1)  # 1
        y_treated_end = torch.mean(y_treated_end, dim=-1)  # 1
        y_control_begin = torch.mean(y_control_begin, dim=-1)  # 1
        y_control_end = torch.mean(y_control_end, dim=-1)  # 1

        effect_t = (y_treated_end - y_treated_begin) - (y_control_end - y_control_begin)
        effect_all.append(float(effect_t))


    # f = plt.figure()
    #
    # x_time = range(len(outcomes)-1)
    # plt.plot(x_time, effect_all, 'g^-', label='DID', markersize=6, markevery=1)
    #
    # plt.xlabel("time step", fontsize=20)
    # plt.ylabel("Causal effect", fontsize=20)
    # plt.legend(loc='lower right', fontsize=18)
    #
    # # tick_params(which='both', direction='in')
    # plt.grid(linestyle=':')
    # plt.grid(axis='x')
    # #plt.title(policy_index[pi])
    #
    # # font size
    # plt.xticks(range(len(outcomes)-1), fontsize=14)
    # plt.yticks(fontsize=20)
    #
    # # plt.title("")
    # # plt.savefig('5methods_600.png')
    # plt.show()

    print("DID: ", effect_all)
    return


def find_time_step(date, time_step_list):

    for i in range(len(time_step_list)):
        t_begin = time_step_list[i][0]
        t_end = time_step_list[i][1]
        if date >= t_begin and date < t_end:
            return i

    if date <= time_step_list[0][0]:
        return 0
    if date > time_step_list[-1][1]:
        return -1
    return -1


def top_policy(path, time_step_list=None, level='state', policy_keywords=None, policy_index=None, countys=None, mustcontain=None):
    # from each category, select the top k policies, and output their names and treatment assignment
    # read data
    # NOTICE: policy file should be sorted by date, with proper date form 'xxxx-xx-xx'
    T = len(time_step_list)
    N = len(countys)

    key2idx = {}
    for i in range(len(countys)):
        key2idx[countys[i]] = i
        state = countys[i][-2:]
        if state not in key2idx:
            key2idx[state] = [i]
        else:
            key2idx[state].append(i)

    csv_t_all = pd.read_csv(path)

    select_columns = ["state_id", "county", "fips_code", "policy_level", "date", "policy_type", "start_stop",
                      "comments"]
    csv_t = csv_t_all[select_columns]

    print('datatype of column date is: ' + str(csv_t_all['date'].dtypes))

    # read the file
    #time_policy = {}  # time_policy['county']['policy'] = (date_start, date_end)

    time_last = '2020-12-31'
    time_last = datetime.datetime.strptime(time_last, '%Y-%m-%d')

    policy_details = {cate: {} for cate in policy_index}  # {'SO': {'food and drink': {assign: T x N assignment}}}

    # start reading
    for i, row in csv_t.iterrows():
        state_id = row["state_id"]
        county = row["county"]
        policy_level = row["policy_level"]
        date = row["date"]
        date = datetime.datetime.strptime(date, '%m/%d/%y')
        if date > time_last:
            print("break at ", i)
            break
        current_time_step = find_time_step(date, time_step_list)
        policy_type = row["policy_type"]
        start_stop = row["start_stop"]
        comments = row["comments"]

        if mustcontain and len(mustcontain) > 0:  # mustcontain is not empty,
            contain_must = False  # contain the must-contain key words?
            for must_word in mustcontain:
                if must_word in policy_type or must_word in comments:
                    contain_must = True
                    break
            if not contain_must:  #  the policy does not contain
                continue

        assert start_stop == 'start' or start_stop == 'stop'

        if level == 'state' and policy_level != 'state':  # if it is in state-level, ignore county-level policies
            continue

        if state_id not in key2idx:
            continue
        if level == 'state':
            county_index = key2idx[state_id]
        elif level == 'county':
            county_index = key2idx[county]

        policy_list = []  # the list of categories that the current policy belongs to
        for policy_cate in policy_keywords:
            for keyword in policy_keywords[policy_cate]:
                if keyword in policy_type or keyword in comments:  # current policy is in policy_cate
                    policy_list.append(policy_cate)
                    break

        for cate in policy_list:
            if cate not in policy_details:
                policy_details[cate] = {}
            if policy_type not in policy_details[cate]:
                policy_details[cate][policy_type] = {'assign': np.zeros((T, N))}  # T x N

            if start_stop == "start":
                #policy_details[cate][policy_type]['status'] = 'start'
                policy_details[cate][policy_type]['assign'][current_time_step:, county_index] = 1
            elif start_stop == "stop":
                #policy_details[cate][policy_type]['status'] = 'stop'
                policy_details[cate][policy_type]['assign'][current_time_step:, county_index] = 0

    # aggregate in each category
    treat_assign = []  # T x N x |Category|
    small_policy_names = {}
    for cate in policy_index:
        small_names_cate = [small_policy for small_policy in policy_details[cate]]
        small_policy_names[cate] = small_names_cate

        policy_assign_cate = [np.expand_dims(policy_details[cate][small_policy]['assign'], axis=0) for small_policy in small_names_cate]
        policy_assign_cate = np.concatenate(policy_assign_cate, axis=0)  # |Cluster size| x T x N
        policy_assign_cate = np.sum(policy_assign_cate, axis=0)  # T x N
        policy_assign_cate[np.nonzero(policy_assign_cate)] = 1
        treat_assign.append(np.expand_dims(policy_assign_cate, axis=-1))

    treat_assign = np.concatenate(treat_assign, axis=-1)  # T x N x |Category|

    # statistics
    for i in range(len(time_step_list)):
        print('============ timestep: ', i)
        for pi in range(len(policy_index)):
            treat_num = (treat_assign[i, :, pi] == 1).sum()  # size of treated group
            treat_rate = float(treat_num) / N
            print('             policy: ', policy_index[pi], ' treated num: ', treat_num, ' total num: ', N, ' treated rate: ', treat_rate)

    # for each category, select top k policies with highest number of applied counties
    topk = 100
    policy_assign_all = {}
    policy_names_all = {}
    for cate in policy_index:
        policy_assign_cate = [np.expand_dims(policy_details[cate][small_policy]['assign'], axis=0) for small_policy in
                              small_policy_names[cate]]
        policy_assign_cate = np.concatenate(policy_assign_cate, axis=0)  # |category| x T x N
        policy_num_cate = np.sum(policy_assign_cate, axis=-1)  # |category| x T, num of counties applied
        sort_idx_cate = np.argsort(policy_num_cate[:, -1])[::-1]  # |category|, sort by the assigned number in the last time step in descending order

        policy_assign_all[cate] = policy_assign_cate[sort_idx_cate]  # |Cluster size| x T x N, ranked
        policy_names_all[cate] = [small_policy_names[cate][i] for i in sort_idx_cate] # |Cluster size|


        # print("============  policy in category ", cate, "==================")
        # for i in range(topk):
        #     if i >= len(sort_idx_cate):
        #         break
        #     print("policy: ", small_policy_names[cate][sort_idx_cate[i]], ": ", policy_num_cate[sort_idx_cate[i]])

    treat_assign = treat_assign.swapaxes(0,1)  # T x N x |Category| -> N x T x |Category|

    return treat_assign, policy_assign_all, policy_names_all



def read_policy(path, time_step_list=None, level='state', policy_keywords=None, policy_index=None, instance_list=None):
    '''
    :param path:
    :param treatments: {'mask',}
    :param time_step_list: [(date_start_1, date_end_1), (), ...]
    :return: {"mask", |T| x county x }
    '''
    # read data
    csv_t_all = pd.read_csv(path)

    select_columns = ["state_id", "county", "fips_code", "policy_level", "date", "policy_type", "start_stop", "comments"]
    csv_t = csv_t_all[select_columns]

    print('datatype of column date is: ' + str(csv_t_all['date'].dtypes))

    # read the file
    time_policy = {}

    #time_last = '2020-08-01'
    time_last = '2020-12-31'
    time_last = datetime.datetime.strptime(time_last, '%Y-%m-%d')

    for i, row in csv_t.iterrows():
        state_id = row["state_id"]
        county = row["county"]
        policy_level = row["policy_level"]
        date = row["date"]
        policy_type = row["policy_type"]
        start_stop = row["start_stop"]
        comments = row["comments"]
        assert start_stop == 'start' or start_stop == 'stop'

        if level == 'state' and policy_level != 'state':  # county-level
            continue

        date = datetime.datetime.strptime(date, '%m/%d/%y')
        if date > time_last:
            print("break at ", i)
            break

        policy_list = []
        # for keyword in policy_keywords:
        #     if keyword in policy_type or keyword in comments:
        #         policy_list.append(keyword)
        for policy_cate in policy_keywords:
            for keyword in policy_keywords[policy_cate]:
                if keyword in policy_type or keyword in comments:
                    policy_list.append(policy_cate)
                    break

        if state_id in time_policy:
            for policy in policy_list:
                if policy in time_policy[state_id]:
                    if start_stop not in time_policy[state_id][policy]:
                        time_policy[state_id][policy][start_stop] = date
                    else:
                        if start_stop == "start" and date < time_policy[state_id][policy]["start"]:
                            time_policy[state_id][policy][start_stop] = date
                        elif start_stop == "stop" and date > time_policy[state_id][policy]["stop"]:
                            time_policy[state_id][policy][start_stop] = date
                else:
                    time_policy[state_id][policy] = {}
                    time_policy[state_id][policy][start_stop] = date
        else:
            time_policy[state_id] = {}
            for policy in policy_list:
                time_policy[state_id][policy] = {}
                time_policy[state_id][policy][start_stop] = date

    #
    num_instance = len(instance_list)
    treat_assign = None  # |T| x N x |policy|
    for i in range(len(time_step_list)):
        treat_assign_i = np.zeros([1, num_instance, len(policy_keywords)])  # T x N x |keywords|

        time_step_start = time_step_list[i][0]
        time_step_end = time_step_list[i][1]

        for j in range(len(instance_list)):
            ct = instance_list[j]
            state = ct[-2:]
            for pi in range(len(policy_index)):
                policy = policy_index[pi]  # "SD"
                if state in time_policy and policy in time_policy[state]:
                    if ("start" in time_policy[state][policy] and time_policy[state][policy][
                        "start"] < time_step_end) and (
                            "stop" not in time_policy[state][policy] or time_policy[state][policy][
                        "stop"] > time_step_start):
                        treat_assign_i[0][j][pi] = 1

        treat_assign = treat_assign_i if treat_assign is None else np.concatenate([treat_assign, treat_assign_i], axis=0)

        # statistics
        # print('============ timestep: ', i)
        # for pi in range(len(policy_index)):
        #     treat_num = (treat_assign[i, :, pi] == 1).sum()  # size of treated group
        #     treat_rate = float(treat_num) / num_instance
        #     print('             policy: ', policy_index[pi], ' treated num: ', treat_num, ' total num: ', num_instance, ' treated rate: ', treat_rate)


    treat_assign = treat_assign.swapaxes(0,1)  # T x N -> N x T

    return treat_assign

