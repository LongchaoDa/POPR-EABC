import decimal
import sys

from ABC.add.constants import env_list
from ABC.metric import evaluatenDCG, SRCC, RegretK_MSE
from ABC.metric import SRCC as SR


# Add the ABC/ path here: .../stb3_zoo/rl_baselines3_zoo/
sys.path.append("../stb3_zoo/rl_baselines3_zoo/")

# Add the ABC/ path here: .../ABC/
sys.path.append("../ABC/")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tqdm
from plot.utils import smooth_y_curve
from plotly.figure_factory._distplot import scipy
from stable_baselines3 import A2C, PPO, DQN, SAC
import rl_zoo3.import_envs  # noqa: F401 pylint: disable=unused-import
from stb3_zoo.rl_baselines3_zoo.rl_zoo3.utils import ALGOS, create_test_env, get_saved_hyperparams
from stb3_zoo.rl_baselines3_zoo.rl_zoo3.exp_manager import ExperimentManager
from stb3_zoo.rl_baselines3_zoo.rl_zoo3.load_from_hub import download_from_hub
from stb3_zoo.rl_baselines3_zoo.rl_zoo3.utils import StoreDict, get_model_path
import argparse
import importlib
import re
import os
import sys
import torch as th
import yaml
from huggingface_sb3 import EnvironmentName
from stable_baselines3.common.callbacks import tqdm
from stable_baselines3.common.utils import set_random_seed
import torch
import os
import json
from scipy.stats import zscore
from ABC.add import online_agent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_param(data, path):
    with open(path, mode="w") as new_f:
        json.dump(data, new_f)
    print("params saved successfully!")


def get_part_data(data, low=0, high=0.25):
    collect_new = []
    for d in data:
        leng_d = len(d)
        new_d = d[int(leng_d * low): int(leng_d * high)]
        collect_new.append(new_d)
    return collect_new


def get_cases_rank(collect_all=None, policy_list='', range_all=None):
    ordered_list = []

    for list_n in collect_all:
        ordered_list.append(sorted(list_n))  # increasing order

    case_all = []
    range_all = range_all
    for i in range(3):
        # range_all 【high->low】 [[0.95, 1], [0.475, 0.525], [0, 0.05]]
        new_data = get_part_data(data=ordered_list, low=range_all[i][0], high=range_all[i][1])
        case_all.append(new_data)

    best_case_dict = {policy_list[i]: case_all[0][i] for i in range(len(policy_list))}
    medium_case_dict = {policy_list[i]: case_all[1][i] for i in range(len(policy_list))}
    worst_case_dict = {policy_list[i]: case_all[2][i] for i in range(len(policy_list))}

    # highest->lowest
    sorted_best_case_dict = dict(sorted(best_case_dict.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True))
    sorted_medium_case_dict = dict(sorted(medium_case_dict.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True))
    sorted_worst_case_dict = dict(sorted(worst_case_dict.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True))
    return sorted_best_case_dict, sorted_medium_case_dict, sorted_worst_case_dict


def replace_non_numeric(str):
    new_str = ""
    for char in str:
        if char.isnumeric():
            new_str += char
        else:
            new_str += " "
    return new_str


def read_order_results(order_folder=""):
    for root, dirs, files in os.walk(order_folder):
        for name in files:
            if name == "ranking":
                the_path = os.path.join(root, name)
                # record_list.append(the_path)
                with open(the_path, mode="r") as rf:
                    content = rf.readline().replace(" ", "").replace("\n", "").replace("'", "").split(",")

    return content


def plot_relation(name_ranking, name=""):
    table_dict = {value: i for i, value in enumerate(name_ranking[0])}
    print(table_dict)
    construct = []
    for i in range(len(name_ranking)):
        arr1 = [table_dict[value] for value in name_ranking[i]]
        construct.append(arr1)

    # corr_matrix = np.corrcoef([construct[0], construct[1], construct[2]])

    fig, ax = plt.subplots(figsize=(8, 6))
    # ax.scatter(construct[0], construct[0], c='red', label='best case')
    # ax.scatter(construct[0], construct[1], c='blue', label='medium case')
    # ax.scatter(construct[0], construct[2], c='green', label='worst case')

    ax.scatter(construct[0], construct[0], c='red', label='round1')
    ax.scatter(construct[0], construct[1], c='blue', label='round2')
    ax.scatter(construct[0], construct[2], c='green', label='round3')

    p1 = np.polyfit(construct[0], construct[0], 1)
    p2 = np.polyfit(construct[0], construct[1], 1)
    p3 = np.polyfit(construct[0], construct[2], 1)

    x = construct[0]
    plt.plot(x, np.polyval(p1, x), 'r--', label='y1 regression line')
    plt.plot(x, np.polyval(p2, x), 'b--', label='y2 regression line')
    plt.plot(x, np.polyval(p3, x), 'g--', label='y2 regression line')
    # plt.imshow(corr_matrix, cmap='hot', interpolation='nearest')
    plt.legend(loc='upper left')
    plt.title(name + ':Linear Relationship Between Three Cases')
    plt.show()
    print(name_ranking)
    print(construct)


def read_theta_per_policy(theta_folder="", stored_name="", low_b=0, high_b=1, mean_action=True, breaker = "-", remove_stuck = False):
    """
    stored_name: only one is enough, because it contains all information
    :param theta_folder:
    :param stored_name:
    :return:
    """
    record_list = []
    for root, dirs, files in os.walk(theta_folder):
        for name in files:
            if name == stored_name:
                the_path = os.path.join(root, name)
                # record_list.append(the_path)
                with open(the_path, mode="r") as rf:
                    content = rf.read()
                    # content = replace_non_numeric(content).replace("  ", " ")
                    string = re.sub(r'\s+', ' ', content)
                    record_list.append(string)
    # print(record_list)
    # length = len(stored_name)
    all_files = []
    for content in record_list:
        alear_signal = False
        # now they are on the name list [[name1, name2...], [name1, name2...], [name1, name2...]]
        break_down = np.array(content.split(breaker)[0:-1])
        update = []

        # counter = 0
        for info in break_down:

            up_info = list(filter(lambda x: x != "", info.split(" ")))
            for i in range(len(up_info)):
                up_info[i] = float(up_info[i])
            length_u = len(up_info)
            # remove the same nums, cause this means the sampling from distribution is stucked somewhere
            if remove_stuck:
                lst = [x for i, x in enumerate(up_info) if i == 0 or up_info[i - 1] != x]
                up_info = list(filter(lambda x: lst.count(x) == 1, lst))
            low_bar = int(low_b * length_u) if int(low_b * length_u) > 0 else 0
            high_bar = int(high_b * length_u) if int(high_b * length_u) < length_u else length_u

            chosen = up_info[low_bar: high_bar]
            if len(chosen) == 0:
                alear_signal = True
                break

            update.append(chosen)
        if alear_signal:
            continue
        else:
            min_len = min(map(len, update))
            new_update = [[] for _ in range(len(update))]  #
            for i, sublist in enumerate(update):
                for j in range(min_len):
                    new_update[i].append(sublist[j])

            all_files.append(new_update)
    # all_files = np.array(all_files)  # (3, 6,v250)
    # all_files_trans = np.transpose(all_files, (1, 0, 2))
    # if mean_action:
    #     out_data = np.mean(all_files_trans, axis=1)
    # else:
    #     out_data = np.concatenate(all_files_trans, axis=1) # (policy_len, num_theta), (6, 250)

    return all_files

    # for name in dirs:
    #     # 处理文件夹
    #     print(os.path.join(root, name))


def case_compare_by_load(theta_file="", sorted_name="A", policy_name="", output_file="", save=True, visual=True,
                         low_b=0.75, high_b=1, mean_action=False, breaker="-", remove_stuck = False):
    """
    default: doing the best case analysis
    :param theta_file:
    :param sorted_name:
    :param policy_name:
    :param output_file:
    :param save:
    :param visual:
    :param low_b:
    :param high_b:
    :param mean_action:
    :return:
    """
    nDCG_list = []
    SRCC_val_list = []
    if theta_file == "" or policy_name == "":
        print("error param input!")
    else:
        store_theta_per_policy = read_theta_per_policy(theta_folder=theta_file, stored_name=sorted_name, low_b=low_b,
                                                       high_b=high_b, mean_action=mean_action, breaker=breaker, remove_stuck = remove_stuck)
        counter = 0
        for epoc in store_theta_per_policy:
            real_out = output_file + str(counter) + "/"
            check_dir_and_mk(real_out)
            nDCG, SRCC_val = final_process(policy_name, epoc, real_out, save=save, visual=visual,
                                           estimation_result=None)
            nDCG_list.append(nDCG)
            SRCC_val_list.append(SRCC_val)
            counter += 1

    return nDCG_list, SRCC_val_list


# def choose_priori(priori_type="beta")
#     if priori_type == "beta":
#         priors = [BetaPrior(0.5, 0.5)] * test_policy_num
#     elif priori_type == "norm":
#         priors = [NormPriori(miu, sigma)] * test_policy_num
# 
#     return priors


def calculate_z_score(input_sample):
    """
    Calculate the z_score when running the experiment
    :param input_sample: input_sample value
    :return: z_score value
    """
    return zscore(input_sample)


def check_dir_and_mk(dir_path):
    """
    check if a dir exist, if not, make dir
    :param dir_path:
    :return:
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        pass


def get_order_list(name_list, trace_list):
    """
    generate a ranked namelist based on the trace_theta
    we use the middle part of the data (strictly middle[0.25-0.75], not ranked)
    :param name_list:
    :param trace_list:
    :return:
    """
    value_list = []
    for i in range(len(name_list)):
        amount = len(trace_list[i])
        if amount < 10:
            value_list.append(np.mean(trace_list[i]))
        else:
            upper = int(0.25 * amount)
            lower = int(0.75 * amount)
            value_list.append(np.mean(trace_list[i][upper: lower]))
    raw_order_dict = dict(zip(name_list, value_list))
    sorted_dict = sorted(raw_order_dict.items(), key=lambda kv: kv[1], reverse=True)
    updated_name_list = []
    for k in range(len(name_list)):
        updated_name_list.append(sorted_dict[k][0])

    return updated_name_list


def save_policy_eva_logs(name_list, trace_list, dir_path, evaluated_order, nDCG, SRCC, regretK=None, MSE=None):
    """
    to save the evaluated policy result to files, including metrics: NDCG, SRCC, (MSE, regretK)
    :param name_list:
    :param trace_list:
    :param dir_path:
    :param evaluated_order:
    :param nDCG:
    :param SRCC:
    :param regretK:
    :param MSE:
    :return:
    """
    name_len = len(name_list)
    if name_len != len(trace_list):
        raise Exception("the length of input: name_list not equals to trace_list!")

    with open(file=os.path.join(dir_path, "ranking"), encoding="utf-8", mode="a+") as rk:
        rk.writelines(str(evaluated_order).replace("[", "").replace("]", "") + "\n")
        rk.writelines("nDCG: " + str(nDCG) + "\n")
        rk.writelines("SRCC: " + str(SRCC) + "\n")
        if regretK != None:
            rk.writelines("regretK: " + str(regretK) + "\n")
        if MSE != None:
            rk.writelines("MSE: " + str(MSE) + "\n")

    for i in range(name_len):
        with open(file=os.path.join(dir_path, name_list[i]), encoding="utf-8", mode="a+") as wf:
            for line in trace_list:
                # print(str(line).replace("[", "").replace("]", "") + "\n")
                wf.writelines(str(line).replace("[", "").replace("]", "") + "\n")
                wf.writelines('-' + "\n")


def generate_all_bootstrapped_indexes(total_Episodes, output_num, bootstrap_M):
    '''
    generate_all_bootstrapped_indexes, only  return the index_lists
    :param total_Episodes:
    :param output_num:
    :param bootstrap_M:
    :return:
    '''
    index_list = []
    for item in range(output_num):
        temp = np.random.randint(low=0, high=total_Episodes - 1, size=bootstrap_M)
        index_list.append(temp)

    return index_list


def generate_all_bootstraps(expert_d, index_list, expert_rew=False):
    """
    generate actual trajs based on the input index and expert_dic
    :param expert_d:
    :param index_list:
    :return:
    """
    # below contains 5 for each items(10000, 5, x)
    boostrap_item_state_list = []
    boostrap_item_action_list = []
    boostrap_item_reward_list = []

    all_states = expert_d["states"]
    all_actions = expert_d["actions"]
    all_reward = expert_d["rewards"]

    for index in index_list:
        temp = []
        temp_action = []
        temp_reward = []
        for k in index:
            temp.append(all_states[k])
            temp_action.append(all_actions[k])
            temp_reward.append(all_reward[k])
        boostrap_item_state_list.append(temp)
        boostrap_item_action_list.append(temp_action)
        boostrap_item_reward_list.append(temp_reward)
    if expert_rew:
        return boostrap_item_state_list, boostrap_item_action_list, boostrap_item_reward_list
    else:
        return boostrap_item_state_list, boostrap_item_action_list


def visualize_each_policy_histograms(name_list, theta_store_list):
    """
    demonstrate the theta trace according to the name list
    :param name_list:
    :param theta_store_list:
    :return:
    """
    import matplotlib.pyplot as plt

    labels = name_list

    data_sets = []
    min_len_data = theta_store_list[0].__len__()
    for data in theta_store_list:
        record = len(data)
        if record < min_len_data:
            min_len_data = record
        new_data = np.array(data)
        data_sets.append(new_data)
    data_cut = []
    for data in data_sets:
        data_cut.append(data[:min_len_data])

    data_sets = data_cut

    number_of_bins = 20
    hist_range = (np.min(data_sets), np.max(data_sets))
    binned_data_sets = [
        np.histogram(d, range=hist_range, bins=number_of_bins)[0]
        for d in data_sets
    ]
    binned_maximums = np.max(binned_data_sets, axis=1)
    x_locations = np.arange(0, sum(binned_maximums), np.max(binned_maximums))

    # The bin_edges are the same for all of the histograms
    bin_edges = np.linspace(hist_range[0], hist_range[1], number_of_bins + 1)
    heights = np.diff(bin_edges)
    centers = bin_edges[:-1] + heights / 2

    # Cycle through and plot each histogram
    fig, ax = plt.subplots()
    for x_loc, binned_data in zip(x_locations, binned_data_sets):
        lefts = x_loc - 0.5 * binned_data
        ax.barh(centers, binned_data, height=heights, left=lefts)

    ax.set_xticks(x_locations, labels)

    ax.set_ylabel("Data values")
    ax.set_xlabel("Data sets")

    plt.show()


def visualize_each_policy_distribution(name_list, theta_store_list, fname=None):
    """
    Violin plot and Box plot based on the thata trace list
    :param name_list:
    :param theta_store_list:
    :param fname:
    :return:
    """
    import matplotlib.pyplot as plt

    labels = name_list

    data_sets = []
    min_len_data = theta_store_list[0].__len__()
    for data in theta_store_list:
        record = len(data)
        if record < min_len_data:
            min_len_data = record
        new_data = np.array(data)
        data_sets.append(new_data)
    data_cut = []
    for data in data_sets:
        data_cut.append(data[:min_len_data])

    data_sets = data_cut

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

    axs[0].violinplot(data_sets,
                      showmeans=False,
                      showmedians=True)
    axs[0].set_title('Violin plot')

    # plot box plot
    axs[1].boxplot(data_sets)
    axs[1].set_title('Box plot')

    # labels = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']

    length = len(data_sets)
    labels = []
    for i in range(length):
        labels.append("x" + str(i + 1))

    # adding horizontal grid lines
    for ax in axs:
        ax.yaxis.grid(True)
        ax.set_xticks([y + 1 for y in range(len(data_sets))],
                      labels)
        ax.set_xlabel('Separate policies')
        ax.set_ylabel('Theta values')

    if fname:
        plt.savefig(fname, dpi=500)
        # plt.clf()
        # plt.close()

    plt.show()


def viz_policy_kde_plots(name_list, theta_store_list, fname=None):
    """
    a kde distribution plot
    :param name_list:
    :param theta_store_list:
    :param fname:
    :return:
    """
    print("name_list")
    print(name_list)
    # name_list = name_list[1:3] + name_list[4:]
    # theta_store_list = theta_store_list[1:3] + theta_store_list[4:]
    # print(name_list)
    df = pd.DataFrame(np.transpose(np.array(theta_store_list)), columns=name_list)
    # input = np.transpose(np.array(theta_store_list)).reshape(1, -1)
    # df = pd.DataFrame(input, columns=name_list)
    for data in theta_store_list:
        sns.kdeplot(data=df)
    if fname:
        plt.savefig(fname)
        plt.clf()
        plt.close()
    else:
        plt.show()


def viz_trace_plots(name_list, theta_store_list, fname=None):
    num_policies = len(theta_store_list)
    fig, axs = plt.subplots(
        nrows=num_policies,
        ncols=1,
        sharex=True,
        sharey=True,
        figsize=(16, 8)
    )

    for idx in range(num_policies):

        data = theta_store_list[idx]
        total = np.arange(len(data))
        smooth_result = smooth_y_curve(data)
        y_std_low = []
        y_std_up = []
        for j in range(total[-1] + 1):
            if data[j] > smooth_result[j]:
                y_std_up.append(data[j])
                y_std_low.append(smooth_result[j])
            else:
                y_std_up.append(smooth_result[j])
                y_std_low.append(data[j])

        axs[idx].fill_between(total, y_std_low, y_std_up, alpha=.3, lw=0)
        # print(str(temp_prob[i]))
        print(str(name_list[idx]) + "mean value: ".format(np.mean(smooth_result)))
        axs[idx].plot(total, smooth_result, label=name_list[idx])
        axs[idx].set_xlabel('Policy: ' + name_list[idx], fontsize=6)
        fig.suptitle("energyABC")
    if fname:
        plt.savefig(fname)
        plt.clf()
        plt.close()
    else:
        plt.show()


def viz_prob_mtx(name_list, store_theta_per_policy, fname):
    """
    the prob_metrix of each policy against others
    :param name_list:
    :param store_theta_per_policy:
    :param fname:
    :return:
    """
    n = len(store_theta_per_policy)
    mtx = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            theta_i = store_theta_per_policy[i]
            theta_j = store_theta_per_policy[j]

            p = np.around(np.mean((theta_i > theta_j)), decimals=2)
            mtx[i, j] = p
    figure = plt.figure(figsize=(10, 10))
    sns.heatmap(
        mtx,
        vmin=0, vmax=1,
        # square=True,
        cmap="YlGnBu",
        yticklabels=name_list,
        xticklabels=name_list,
        linewidths=.5,
        annot=True,
        fmt='.2g',
    )
    plt.title("P(policy Y > policy X)")
    if fname:
        plt.savefig(fname, dpi=500)
        plt.clf()
        plt.close()
    else:
        plt.show()

    return mtx


def normalize_fun(x, clip=None, method='maxmin'):
    '''
    the input should be a list
    :param x:
    :param clip:
    :param method:
    :return: a list of normalized values, which specific normalization ways
    '''
    if method == 'maxmin':

        x_update = []
        x = torch.Tensor(x)

        for i in range(len(x)):
            max_x_val = max(x[i])
            min_x_val = min(x[i])
            x_update.append((x[i] - min_x_val) / (max_x_val - min_x_val))

        x = x_update

    if method == 'zscore':
        avg_x = np.mean(x)
        x_update = []

        re_avg_x = np.std(x)
        for i in range(len(x)):
            x_update.append((x[i] - avg_x) / re_avg_x)

        x = x_update

    if method == 'clipping':
        '''
        here for clipping, we use a param, to signify the partition of retain 
        '''
        min_thresh = min(clip)
        max_thresh = max(clip)
        x_update = []
        for i in range(len(x)):
            x_update.append(min_thresh if x[i] < min_thresh else (max_thresh if x[i] > max_thresh else x[i]))

        x = x_update

    print("x_after:{}, with fun:{}".format(x, method))

    return x


def get_candidate_traj_by_policy_and_obs(candidate_agent_path, input_env, expert_obs, env_name='', algo='PPO', state_dim=None, action_dim = None):
    '''
    we should input the expert_states
    then use the current policy, to generate the related actions
    then organize the corresponding actions give feed backs

    we need: expert_states here(used for input of policy);
    param candidate_agent_path:
    :param input_env:
    :return:
    '''

    global rewards
    total_steps = len(expert_obs)
    side_policy = ''
    if candidate_agent_path.__contains__("zero"):
        # print("---using zero policies---")
        # side_policy = 'zero'
        zero_traj = np.zeros((total_steps,), dtype='float64')
        rewards = 0
        return zero_traj, rewards

        # logger.info("")
    if candidate_agent_path.__contains__("random"):

        if env_name.__contains__("Bipedal"):
            """
            exits some issues, need to check befire apply
            """
            # Box(-1.0, 1.0, (4,), float32)
            # (b - a) * random_sample() + a， a = -1, b = 1，
            random_traj = 2 * np.random.random((total_steps * 4)) - 1
            rewards = 0
            random_traj = random_traj.reshape(())
            print("finished random sampling for actions!")
            return random_traj, rewards
            # pass

        # print("---using random policies---")
        # side_policy = 'random'
        random_traj = np.random.randint(0, 3, size=total_steps)
        rewards = 0
        return random_traj, rewards

    else:
        if algo == 'PPO':
            model = PPO.load(candidate_agent_path)
        elif algo == 'SAC':
            model = SAC.load(candidate_agent_path)
        elif algo == 'onlineSAC':
            expert_structure = online_agent.OnlineAgent(state_dim=state_dim, action_dim=action_dim, algo=algo,
                                                        device=device)
            expert_structure.load(candidate_agent_path)  # load to self.model

        # obs = input_env.reset()

        counter = 0
        # flag = True
        reward_acc = 0
        action_log = []

        # for each observation_state, we get an action
        for index in range(total_steps):
            action = expert_structure.take_action(expert_obs[index])
            obs, rewards, dones, _, info = input_env.step(action)
            reward_acc += rewards
            counter += 1
            action_log.append(action)

            # if dones:
            #     flag = False
        action_log = np.array(action_log, dtype='float64').flatten()

        return action_log, rewards


def get_expert_obs_stb_zoo(algo='dqn', env_name="MountainCar-v0", folder='logs/', customized_path=None, base_path=None):
    '''
    we will generate expert one traj for each time calling this function
    :param algo:
    :param env_name:
    :param folder:
    :param timesteps:
    :return:
    '''
    global sub_path
    generate_expert = True
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", help="environment ID", type=EnvironmentName, default=env_name)
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False,
                        choices=list(ALGOS.keys()))
    # parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=timesteps, type=int)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=1, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--no-render", action="store_true", default=True, help="Do not render the environment (useful for tests)"
    )
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument("--device", help="PyTorch device to be use (ex: cpu, cuda...)", default="auto", type=str)
    parser.add_argument(
        "--load-best", action="store_true", default=False, help="Load best model instead of last model if available"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=int,
        help="Load checkpoint instead of last model if available, "
             "you must pass the number of timesteps corresponding to it",
    )
    parser.add_argument(
        "--load-last-checkpoint",
        action="store_true",
        default=False,
        help="Load last checkpoint instead of last model if available",
    )
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    parser.add_argument(
        "--norm-reward", action="store_true", default=False,
        help="Normalize reward if applicable (trained with VecNormalize)"
    )
    random_seed = np.random.randint(0, 10000)
    parser.add_argument("--seed", help="Random generator seed", type=int, default=random_seed)
    parser.add_argument("--reward-log", help="Where to log reward", default="", type=str)
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environment package modules to import (e.g. gym_minigrid)",
    )
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict,
        help="Optional keyword argument to pass to the env constructor"
    )
    parser.add_argument(
        "--custom-objects", action="store_true", default=False, help="Use custom objects to solve loading issues"
    )
    parser.add_argument(
        "-P",
        "--progress",
        action="store_true",
        default=False,
        help="if toggled, display a progress bar using tqdm and rich",
    )
    args = parser.parse_args()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_name: EnvironmentName = args.env
    algo = algo
    if base_path == None:
        record = sys.path[0]
        last = len(record.split("/")[-1])
        new_record = record[:-last - 4]
        base_path = new_record + "stb3_zoo/rl_baselines3_zoo/"
    else:
        base_path = base_path
    folder = base_path + folder

    try:
        _, model_path, log_path = get_model_path(
            args.exp_id,
            folder,
            algo,
            env_name,
            args.load_best,
            args.load_checkpoint,
            args.load_last_checkpoint,
        )
    except (AssertionError, ValueError) as e:
        # Special case for rl-trained agents
        # auto-download from the hub
        if "rl-trained-agents" not in folder:
            raise e
        else:
            print(
                "Pretrained model not found, trying to download it from sb3 Huggingface hub: https://huggingface.co/sb3")
            # Auto-download
            download_from_hub(
                algo=algo,
                env_name=env_name,
                exp_id=args.exp_id,
                folder=folder,
                organization="sb3",
                repo_name=None,
                force=False,
            )
            # Try again
            _, model_path, log_path = get_model_path(
                args.exp_id,
                folder,
                algo,
                env_name,
                args.load_best,
                args.load_checkpoint,
                args.load_last_checkpoint,
            )

    print(f"Loading {model_path}")

    # Off-policy algorithm only support one env for now
    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    if algo in off_policy_algos:
        args.n_envs = 1

    set_random_seed(args.seed)

    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    is_atari = ExperimentManager.is_atari(env_name.gym_id)

    stats_path = os.path.join(log_path, env_name)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_name, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    # overwrite with command line arguments
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    log_dir = args.reward_log if args.reward_log != "" else None

    env = create_test_env(
        env_name.gym_id,
        n_envs=args.n_envs,
        stats_path=stats_path,
        seed=args.seed,
        log_dir=log_dir,
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    kwargs = dict(seed=args.seed)
    if algo in off_policy_algos:
        # Dummy buffer size as we don't need memory to enjoy the trained agent
        kwargs.update(dict(buffer_size=1))
        # Hack due to breaking change in v1.6
        # handle_timeout_termination cannot be at the same time
        # with optimize_memory_usage
        if "optimize_memory_usage" in hyperparams:
            kwargs.update(optimize_memory_usage=False)

    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version or args.custom_objects:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }
    if env_name.__contains__("MountainCar-v0"):
        sub_path = "/MountainCar-v0_1/MountainCar-v0.zip"
    elif env_name.__contains__("Acrobot-v1"):
        sub_path = "/Acrobot-v1_1/Acrobot-v1.zip"
    elif env_name.__contains__("GridWorld"):
        sub_path = "/GridWorld-v0_1/GridWorld-v0.zip"

    # if algo == 'ppo':
    #     model = PPO.load(
    #         path=base_path + "logs/" + algo + sub_path, env=env, custom_objects=custom_objects, device=args.device,
    #         **kwargs)
    # elif algo == 'dqn':
    #     model = DQN.load(
    #         path=base_path + "logs/" + algo + sub_path, env=env, custom_objects=custom_objects, device=args.device,
    #         **kwargs)
    # # elif algo == 'qrdqn':
    # #     model = QRDQN.load(
    # #         path=base_path + "logs/" + algo + sub_path, env=env, custom_objects=custom_objects, device=args.device,
    # #         **kwargs)
    # else:
    # model = ALGOS[algo].load(path=base_path + "logs/" + algo + sub_path, env=env, custom_objects=custom_objects, device=args.device, **kwargs)
    if customized_path != None:
        model = ALGOS[algo].load(path=customized_path, env=env, custom_objects=custom_objects, device=args.device,
                                 **kwargs)
    else:
        model = ALGOS[algo].load(path=model_path, env=env, custom_objects=custom_objects, device=args.device, **kwargs)

    # return model
    obs = env.reset()

    # Deterministic by default except for atari games
    stochastic = args.stochastic or is_atari and not args.deterministic
    deterministic = stochastic

    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0
    # For HER, monitor success rate
    successes = []
    lstm_states = None
    episode_start = np.ones((env.num_envs,), dtype=bool)

    counter = 0
    flag = True
    reward_acc = []
    action_log = []
    obs_log = []

    try:
        while flag:

            # store the observation which feed into the model
            obs_log.append(obs)
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_start,
                # deterministic=deterministic,
            )
            # store the action which generated from the fed obs
            action_log.append(action)

            # then based on the action, update the observation
            obs, reward, done, infos = env.step(action)
            reward_acc += reward
            counter += 1

            episode_start = done

            if not args.no_render:
                env.render("human")

            episode_reward += reward[0]
            ep_len += 1

            if done:
                flag = False

            if args.n_envs == 1:
                # For atari the return reward is not the atari score
                # so we have to get it from the infos dict
                if is_atari and infos is not None and args.verbose >= 1:
                    episode_infos = infos[0].get("episode")
                    if episode_infos is not None:
                        print(f"Atari Episode Score: {episode_infos['r']:.2f}")
                        print("Atari Episode Length", episode_infos["l"])

                if done and not is_atari and args.verbose > 0:
                    # NOTE: for env using VecNormalize, the mean reward
                    # is a normalized reward when `--norm_reward` flag is passed
                    print(f"Episode Reward: {episode_reward:.2f}")
                    print("Episode Length", ep_len)
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(ep_len)
                    episode_reward = 0.0
                    ep_len = 0

                # Reset also when the goal is achieved when using HER
                if done and infos[0].get("is_success") is not None:
                    if args.verbose > 1:
                        print("Success?", infos[0].get("is_success", False))

                    if infos[0].get("is_success") is not None:
                        successes.append(infos[0].get("is_success", False))
                        episode_reward, ep_len = 0.0, 0

    except KeyboardInterrupt:
        pass

    if args.verbose > 0 and len(successes) > 0:
        print(f"Success rate: {100 * np.mean(successes):.2f}%")

    if args.verbose > 0 and len(episode_rewards) > 0:
        print(f"{len(episode_rewards)} Episodes")
        print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")

    if args.verbose > 0 and len(episode_lengths) > 0:
        print(f"Mean episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")

    env.close()

    action_log = np.array(action_log, dtype='float64').flatten()

    return action_log, reward_acc, obs_log


def get_candidate_traj_from_expert_stb_zoo(expert_obs, algo='dqn', env_name="MountainCar-v0", folder='logs/',
                                           timesteps=5000, customized_path=None, base_path=None):
    '''
    we will generate expert one traj for each time calling this function
    :param algo:
    :param env_name:
    :param folder:
    :param timesteps:
    :return:
    '''
    # generate_expert = True
    global sub_path
    total_steps = len(expert_obs)
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", help="environment ID", type=EnvironmentName, default=env_name)
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False,
                        choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=timesteps, type=int)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--no-render", action="store_true", default=True, help="Do not render the environment (useful for tests)"
    )
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument("--device", help="PyTorch device to be use (ex: cpu, cuda...)", default="auto", type=str)
    parser.add_argument(
        "--load-best", action="store_true", default=False, help="Load best model instead of last model if available"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=int,
        help="Load checkpoint instead of last model if available, "
             "you must pass the number of timesteps corresponding to it",
    )
    parser.add_argument(
        "--load-last-checkpoint",
        action="store_true",
        default=False,
        help="Load last checkpoint instead of last model if available",
    )
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    parser.add_argument(
        "--norm-reward", action="store_true", default=False,
        help="Normalize reward if applicable (trained with VecNormalize)"
    )
    random_seed = np.random.randint(0, 100000)
    parser.add_argument("--seed", help="Random generator seed", type=int, default=random_seed)
    parser.add_argument("--reward-log", help="Where to log reward", default="", type=str)
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environment package modules to import (e.g. gym_minigrid)",
    )
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict,
        help="Optional keyword argument to pass to the env constructor"
    )
    parser.add_argument(
        "--custom-objects", action="store_true", default=False, help="Use custom objects to solve loading issues"
    )
    parser.add_argument(
        "-P",
        "--progress",
        action="store_true",
        default=False,
        help="if toggled, display a progress bar using tqdm and rich",
    )
    args = parser.parse_args()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_name: EnvironmentName = args.env
    algo = algo
    if base_path == None:
        base_path = "/Users/danielsmith/Documents/1-RL/NJIT/research/5_offline_eva_A/my_accumulation/experiments/porter/offline_eval/stb3_zoo/rl_baselines3_zoo/"
    folder = base_path + folder

    try:
        _, model_path, log_path = get_model_path(
            args.exp_id,
            folder,
            algo,
            env_name,
            args.load_best,
            args.load_checkpoint,
            args.load_last_checkpoint,
        )
    except (AssertionError, ValueError) as e:
        # Special case for rl-trained agents
        # auto-download from the hub
        if "rl-trained-agents" not in folder:
            raise e
        else:
            print(
                "Pretrained model not found, trying to download it from sb3 Huggingface hub: https://huggingface.co/sb3")
            # Auto-download
            download_from_hub(
                algo=algo,
                env_name=env_name,
                exp_id=args.exp_id,
                folder=folder,
                organization="sb3",
                repo_name=None,
                force=False,
            )
            # Try again
            _, model_path, log_path = get_model_path(
                args.exp_id,
                folder,
                algo,
                env_name,
                args.load_best,
                args.load_checkpoint,
                args.load_last_checkpoint,
            )

    # print(f"Loading {model_path}")

    # Off-policy algorithm only support one env for now
    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    if algo in off_policy_algos:
        args.n_envs = 1

    set_random_seed(args.seed)

    if args.num_threads > 0:
        if args.verbose > 1:
            pass
            # print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    is_atari = ExperimentManager.is_atari(env_name.gym_id)

    stats_path = os.path.join(log_path, env_name)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_name, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    # overwrite with command line arguments
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    log_dir = args.reward_log if args.reward_log != "" else None

    env = create_test_env(
        env_name.gym_id,
        n_envs=args.n_envs,
        stats_path=stats_path,
        seed=args.seed,
        log_dir=log_dir,
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    kwargs = dict(seed=args.seed)
    if algo in off_policy_algos:
        # Dummy buffer size as we don't need memory to enjoy the trained agent
        kwargs.update(dict(buffer_size=1))
        # Hack due to breaking change in v1.6
        # handle_timeout_termination cannot be at the same time
        # with optimize_memory_usage
        if "optimize_memory_usage" in hyperparams:
            kwargs.update(optimize_memory_usage=False)

    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version or args.custom_objects:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    if env_name.__contains__("MountainCar-v0"):
        sub_path = "/MountainCar-v0_1/MountainCar-v0.zip"
    elif env_name.__contains__("Acrobot"):
        sub_path = "/Acrobot-v1_1/Acrobot-v1.zip"

    # print("Current name: {}".format(env_name))
    # print(env_name.__contains__("Acrobot"))
    # print("Current candidate using2: {}".format(sub_path))

    # if algo == 'ppo':
    #     model = PPO.load(
    #         "/Users/danielsmith/Documents/1-RL/NJIT/research/5_offline_eva_A/my_accumulation/experiments/porter/offline_eval/stb3_zoo/rl_baselines3_zoo/logs/" + algo + sub_path)
    # elif algo == 'dqn':
    #     model = DQN.load(
    #         "/Users/danielsmith/Documents/1-RL/NJIT/research/5_offline_eva_A/my_accumulation/experiments/porter/offline_eval/stb3_zoo/rl_baselines3_zoo/logs/" + algo + sub_path)
    # else:
    if customized_path != None:
        model = ALGOS[algo].load(path=customized_path, env=env, custom_objects=custom_objects, device=args.device,
                                 **kwargs)
    else:
        model = ALGOS[algo].load(path=model_path, env=env, custom_objects=custom_objects, device=args.device, **kwargs)

    # return model
    obs = env.reset()

    # Deterministic by default except for atari games
    stochastic = args.stochastic or is_atari and not args.deterministic
    deterministic = stochastic

    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0
    # For HER, monitor success rate
    successes = []
    lstm_states = None
    episode_start = np.ones((env.num_envs,), dtype=bool)

    generator = range(args.n_timesteps)
    # print(generator)
    if args.progress:
        if tqdm is None:
            raise ImportError("Please install tqdm and rich to use the progress bar")
        generator = tqdm(generator)

    # if generate_expert:
    #     # only go once
    #     generator = tqdm(generator)

    counter = 0
    flag = True
    reward_acc = []
    action_log = []
    obs_log = []

    try:
        for index in range(total_steps):
            action, lstm_states = model.predict(
                expert_obs[index],
                state=lstm_states,
                episode_start=episode_start,
                # deterministic=deterministic,
            )
            obs, reward, done, infos = env.step(action)
            reward_acc += reward
            counter += 1
            action_log.append(action)
            obs_log.append(obs)

            episode_start = done

            if not args.no_render:
                env.render("human")

            episode_reward += reward[0]
            ep_len += 1

            if done:
                flag = False

    #         if args.n_envs == 1:
    #             # For atari the return reward is not the atari score
    #             # so we have to get it from the infos dict
    #             if is_atari and infos is not None and args.verbose >= 1:
    #                 episode_infos = infos[0].get("episode")
    #                 if episode_infos is not None:
    #                     print(f"Atari Episode Score: {episode_infos['r']:.2f}")
    #                     print("Atari Episode Length", episode_infos["l"])
    #
    #             if done and not is_atari and args.verbose > 0:
    #                 # NOTE: for env using VecNormalize, the mean reward
    #                 # is a normalized reward when `--norm_reward` flag is passed
    #                 print(f"Episode Reward: {episode_reward:.2f}")
    #                 print("Episode Length", ep_len)
    #                 episode_rewards.append(episode_reward)
    #                 episode_lengths.append(ep_len)
    #                 episode_reward = 0.0
    #                 ep_len = 0
    #
    #             # Reset also when the goal is achieved when using HER
    #             if done and infos[0].get("is_success") is not None:
    #                 if args.verbose > 1:
    #                     print("Success?", infos[0].get("is_success", False))
    #
    #                 if infos[0].get("is_success") is not None:
    #                     successes.append(infos[0].get("is_success", False))
    #                     episode_reward, ep_len = 0.0, 0
    #
    except KeyboardInterrupt:
        pass
    #
    # if args.verbose > 0 and len(successes) > 0:
    #     print(f"Success rate: {100 * np.mean(successes):.2f}%")
    #
    # if args.verbose > 0 and len(episode_rewards) > 0:
    #     print(f"{len(episode_rewards)} Episodes")
    #     print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    #
    # if args.verbose > 0 and len(episode_lengths) > 0:
    #     print(f"Mean episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")

    env.close()

    action_log = np.array(action_log, dtype='float64').flatten()

    return action_log, reward_acc


def normalize_fun(x, y, clip=None, method='zscore'):
    if method == 'squeeze':
        x_update = np.array(x) / 2
        y_update = np.array(y) / 2

        x = x_update
        y = y_update

    if method == 'maxmin':

        max_x_val = max(x)
        min_x_val = min(x)
        x_update = []
        y_update = []
        for i in range(len(x)):
            x_update.append((x[i] - min_x_val) / (max_x_val - min_x_val))

        max_y_val = max(y)
        min_y_val = min(y)
        for i in range(len(y)):
            y_update.append((y[i] - min_y_val) / (max_y_val - min_y_val))

        x = x_update
        y = y_update

    if method == 'zscore':
        avg_x = np.mean(x)
        x_update = []
        y_update = []

        re_avg_x = np.std(x)
        for i in range(len(x)):
            x_update.append((x[i] - avg_x) / re_avg_x)

        avg_y = np.mean(y)

        re_avg_y = np.std(y)
        for i in range(len(y)):
            y_update.append((y[i] - avg_y) / re_avg_y)

        x = x_update
        y = y_update

    if method == 'clipping':
        '''
        here for clipping, we use a param, to signify the partition of retain 
        '''
        min_thresh = min(clip)
        max_thresh = max(clip)
        x_update = []
        y_update = []
        for i in range(len(x)):
            x_update.append(min_thresh if x[i] < min_thresh else (max_thresh if x[i] > max_thresh else x[i]))

        for j in range(len(y)):
            y_update.append(min_thresh if y[j] < min_thresh else (max_thresh if y[j] > max_thresh else y[j]))

        x = x_update
        y = y_update

    # print("x_after:{}, with fun:{}".format(x, method))
    # print("y_after:{}, with fun:{}".format(y, method))

    return x, y


def calculate_KL(x, y, normalize=None):
    if normalize != None:
        x, y = normalize_fun(x, y, normalize)
    x = np.array(x)
    PX = x / np.sum(x)
    y = np.array(y)
    PY = y / np.sum(y)

    result = np.sum(PX * np.log(PX / PY))

    scipy_result = scipy.stats.entropy(PX, PY)
    print('scipy_result:{}'.format(scipy_result))
    print('my result:{}'.format(result))
    return result


def calculate_JS2(x, y, normalize, clip=None):
    import scipy.stats as ss
    if clip is None:
        clip = [0, 2]
    if normalize != None:
        x, y = normalize_fun(x, y, clip=clip, method=normalize)
    x = np.array(x)
    y = np.array(y)
    PX = x / np.sum(x)
    PY = y / np.sum(y)

    M = (PX + PY) / 2
    return 0.5 * ss.entropy(PX, M, base=2) + 0.5 * ss.entropy(PY, M, base=2)


def evaluate_energy(method, sample1, sample2, kernel):
    global result

    # update with variable as input
    if method == 'JS':
        # JS2 means the modified version of JS
        # options for normalization: maxmin, z-score, None
        result = calculate_JS2(sample1, sample2, normalize=None)
        result = 1 - result
    if method == 'KL':
        result = calculate_KL(sample1, sample2, normalize=None)

    if method == 'MMD':
        # sample1, sample2 = normalize_fun(sample1, sample1, clip=None, method='zscore')
        sample1 = torch.Tensor([sample1])
        sample2 = torch.Tensor([sample2])

        result = MMD(sample1, sample2, kernel=kernel)

    return result


def MMD(x, y, kernel='multiscale'):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

       the default kernel is set as multi-scale

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    # x:(20, 2), x.t:(2, 20) ==> torch.mm(x, x.t) = (20, 20)
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY)


def load_expert_SOPR(expert_path, input_env, determ=False, length=None, algo='onlineSAC', device=torch.device('cpu'), state_dim = None, action_dim = None):
    batch_size = 256
    use_gpu = (device == torch.device(device))
    expert_structure = online_agent.OnlineAgent(state_dim=state_dim, action_dim=action_dim, algo=algo, device=device)
    expert_structure.load(expert_path) # load to self.model
    obs = input_env.reset()

    global rewards
    counter = 0
    flag = True
    reward_acc = 0
    action_log = []
    obs_log = []

    length_control = False
    if length != None:
        length_control = True

    if length_control == False:
        while flag:
            if len(obs) == 2:
                obs = obs[0]
            action = expert_structure.take_action(obs)
            # action, _states = model.predict(obs)
            # get obs before the action updates the env
            obs_log.append(obs)

            obs, rewards, dones, _, info = input_env.step(action)
            reward_acc += rewards
            counter += 1
            action_log.append(action)

            if dones:
                flag = False
    else:

        # for gridworld: if the total steps are too short, we control by expected length
        for iter in range(length):
            if len(obs) == 2:
                obs = obs[0]
            action = expert_structure.take_action(obs)
            obs_log.append(obs)

            counter += 1
            action_log.append(action)
            obs, rewards, dones, _,  info = input_env.step(action)
            reward_acc += rewards

            if dones:
                obs = input_env.reset()

    action_log = np.array(action_log, dtype='float64').flatten()

    return action_log, rewards, obs_log


def load_expert_traj(expert_path, input_env, determ=True, length=None, algo='PPO'):
    '''
    Current used version

    input the expert_path and input_env, we will load the model and play
    around the specific length, the record the traj(state, action, reward)

    we set an option, when deter: the load from files
    when deter is False, then load from exploration(online)

    :param expert_path:
    :param input_env:
    :param determ:
    :return:
    '''
    if determ:
        print("-----generating expert logs fixedly-----")
        obs = input_env.reset()
        with open('../mountain_car/expert/actions.txt') as ac_f:
            ac_result = ac_f.read().splitlines()
            ac_result = list(map(float, ac_result))
            # print(ac_result)

        with open('../mountain_car/expert/obs.txt') as ob_f:
            a_list = ['[', ']']
            obs_log = []
            ob_result = ob_f.read().replace(a_list[0], "").replace(a_list[1], "").splitlines()
            for item in ob_result:
                temp_list = item.replace("   ", ",").replace("  ", ",").replace(" ", ",").split(",")[:2]
                temp_list = list(map(float, temp_list))
                obs_log.append(np.array(temp_list))

        act_log = np.array(ac_result)
        obs_log = list(np.array(obs_log).reshape((112, 1, 2)))
        ac_reward = []

        return act_log, ac_reward, obs_log

    else:
        print("-----generating expert logs online-----")
        global rewards
        if algo == 'PPO':
            model = PPO.load(expert_path)
        elif algo == 'SAC':
            model = SAC.load(expert_path)
        obs = input_env.reset()

        counter = 0
        flag = True
        reward_acc = 0
        action_log = []
        obs_log = []

        length_control = False
        if length != None:
            length_control = True

        if length_control == False:
            while flag:
                action, _states = model.predict(obs)
                # get obs before the action updates the env
                obs_log.append(obs)

                obs, rewards, dones, info = input_env.step(action)
                reward_acc += rewards
                counter += 1
                action_log.append(action)

                if dones:
                    flag = False
        else:

            # for gridworld: if the total steps are too short, we control by expected length
            for iter in range(length):

                action, _states = model.predict(obs)
                obs_log.append(obs)

                counter += 1
                action_log.append(action)
                obs, rewards, dones, info = input_env.step(action)
                reward_acc += rewards

                if dones:
                    obs = input_env.reset()

        action_log = np.array(action_log, dtype='float64').flatten()

        return action_log, rewards, obs_log
    # each obs is the input index
    # each action is the follow up action index


def load_expert_traj2(expert_path, input_env, determ=True):
    """
    Previous version
    :param expert_path:
    :param input_env:
    :param determ:
    :return:
    """
    if determ:
        with open('expert/actions.txt') as ac_f:
            ac_result = ac_f.read().splitlines()
            ac_result = list(map(float, ac_result))
            # print(ac_result)

        with open('expert/obs.txt') as ob_f:
            a_list = ['[', ']']
            obs_log = []
            ob_result = ob_f.read().replace(a_list[0], "").replace(a_list[1], "").splitlines()
            for item in ob_result:
                temp_list = item.replace("   ", ",").replace("  ", ",").replace(" ", ",").split(",")[:2]
                temp_list = list(map(float, temp_list))
                obs_log.append(np.array(temp_list))
            # return_list = list(map(float, temp_list))
            # print(obs_log)
        act_log = np.array(ac_result)
        obs_log = np.array(obs_log).reshape((112, 1, 2))
        ac_reward = []
        # print(act_log)
        # print(obs_log)

        # return act_log, ac_reward, obs_log

        # else:
        global rewards
        model = PPO.load(expert_path)
        obs = input_env.reset()

        counter = 0
        flag = True
        reward_acc = 0
        action_log = []
        obs_log = []

        while flag:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = input_env.step(action)
            reward_acc += rewards
            counter += 1
            action_log.append(action)
            obs_log.append(obs)

            if dones:
                flag = False
        action_log = np.array(action_log, dtype='float64').flatten()
        # print(action_log)
        # print(obs_log)
        return action_log, rewards, obs_log


def makeup_expert_traj(expert_path, input_env, length):
    '''
    makeup_expert_traj with given expert_path and input env,
    generally speaking, this function is used for generating expert trajs

    :param expert_path:
    :param input_env:
    :param length:
    :return:
    '''
    global rewards
    model = PPO.load(expert_path)
    obs = input_env.reset()

    counter = 0
    flag = True
    reward_acc = 0
    action_log = []
    while flag:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = input_env.step(action)
        reward_acc += rewards
        counter += 1
        action_log.append(action)

        if dones:
            flag = False
    action_log = np.array(action_log, dtype='float64').flatten()
    log_length = len(action_log)
    # print(action_log)

    if log_length < length:
        sub = np.ndarray(length - log_length)
        action_log = np.concatenate((action_log, sub), axis=0)

    return action_log, rewards


def get_candidate_traj(candidate_agent_path, input_env, expert_obs, env_name='', algo='PPO'):
    '''get_candidate_traj

    :param candidate_agent_path:
    :param length:
    :param input_env:
    :return:
    '''
    global rewards
    total_steps = len(expert_obs)

    if candidate_agent_path.__contains__("random"):
        # print("---using random policies---")
        # side_policy = 'random'
        random_traj = np.random.randint(0, 3, size=total_steps)
        rewards = 0
        return random_traj, rewards
    if algo == 'PPO':
        model = PPO.load(candidate_agent_path)
    elif algo == 'SAC':
        model = SAC.load(candidate_agent_path)
    # model = PPO.load(candidate_agent_path)
    counter = 0
    reward_acc = 0
    rewards = 0
    action_log = []

    for i in range(total_steps):
        obs = expert_obs[i]
        action, _states = model.predict(obs)
        obs, rewards, dones, info = input_env.step(action)
        reward_acc += rewards
        counter += 1
        action_log.append(action)

    action_log = np.array(action_log, dtype='float64').flatten()

    return action_log, rewards


def get_origin_candidate_traj(candidate_agent_path, input_env):
    '''
    we should input the expert_stets
    then use the current policy, to generate the related actions
    then organize the corresponding actions give feed backs

    we need: expert_states here(used for input of policy);

    param candidate_agent_path:
    :param input_env:
    :return:
    '''
    global rewards
    model = PPO.load(candidate_agent_path)
    obs = input_env.reset()

    counter = 0
    flag = True
    reward_acc = 0
    action_log = []
    while flag:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = input_env.step(action)
        reward_acc += rewards
        counter += 1
        action_log.append(action)

        if dones:
            flag = False
    action_log = np.array(action_log, dtype='float64').flatten()
    return action_log, rewards


def write_into_file(write_path, detail_name, source_data):
    '''
    mainly used to write into a given file path
    :param write_path:
    :param detail_name:
    :param source_data:
    :return:
    '''
    # if os.path.exists(write_path) == False:
    #     os.makedirs(write_path)
    with open(write_path + detail_name, 'w', encoding='utf-8') as wr:
        for i in source_data:
            wr.write(str(i) + "\n")

    print("finished writing into file! location: " + write_path)


def etimate_reward(reward_mean, theta_per_policy, name_list, candidate_acc_true_reward):
    """
    :param reward_mean: the mean value of the expert data
    :param theta_per_policy: the theta_list per policy
    :return: len(mean, mid) of estimation
    """
    estimate_dict = {}
    estimate_per_policy = [[] for i in range(len(theta_per_policy))]
    for i in range(len(theta_per_policy)):
        theta_list = theta_per_policy[i]
        estimate_reward_mean = reward_mean * np.mean(theta_list)
        estimate_reward_mid = reward_mean * np.median(theta_list)
        estimate_per_policy[i].append(estimate_reward_mean)
        estimate_per_policy[i].append(estimate_reward_mid)
        key_is = str(name_list[i])
        estimate_dict[key_is] = [estimate_reward_mean, candidate_acc_true_reward[i]]

    return estimate_dict


def construct_reward_state(store_slevel_reward_per_policy, name_list, candidate_acc_true_reward):
    """
    mean_expert, store_slevel_reward_per_policy, name_list
    :param reward_mean: the mean value of the expert data
    :param theta_per_policy: the theta_list per policy
    :return: len(mean, mid) of estimation
    """
    estimate_dict = {}
    for i in range(len(store_slevel_reward_per_policy)):
        key_is = str(name_list[i])
        estimate_dict[key_is] = [store_slevel_reward_per_policy[i], candidate_acc_true_reward[i]]

    return estimate_dict


def final_process(policy_name, store_theta_per_policy, exp_dir, save=True, visual=True, estimation_result=None):
    evaluated_order = get_order_list(policy_name, store_theta_per_policy)

    nDCG = evaluatenDCG(policy_name, evaluated_order)
    SRCC_val = SRCC(policy_name, evaluated_order)
    RegretK_val, MSE_val = None, None
    if estimation_result != None:
        RegretK_val, MSE_val = RegretK_MSE(k=3, estimation_result=estimation_result, ideal_order=policy_name,
                                           rank_order=evaluated_order)
    if save:
        save_policy_eva_logs(policy_name, store_theta_per_policy, exp_dir, evaluated_order, nDCG, SRCC_val,
                             regretK=RegretK_val, MSE=MSE_val)
    if visual:
        # visualize_each_policy_histograms(name_list, store_theta_per_policy)
        visualize_each_policy_distribution(policy_name, store_theta_per_policy,
                                           fname=os.path.join(exp_dir, "distribution.png"))
        viz_trace_plots(policy_name, store_theta_per_policy, fname=os.path.join(exp_dir, "trace.png"))
        viz_policy_kde_plots(policy_name, store_theta_per_policy, fname=os.path.join(exp_dir, "kde.png"))
        mtx = viz_prob_mtx(policy_name, store_theta_per_policy, fname=os.path.join(exp_dir, "policy_matrix.png"))
        print(mtx)
    if estimation_result != None:
        return nDCG, SRCC_val, RegretK_val, MSE_val
    else:
        return nDCG, SRCC_val

# def draw(list_nDCG, var_1, list_SRCC, var_2):


# algo_list = ['trpo', 'ars', 'qrdqn', 'ppo', 'a2c', 'dqn']
# result = get_expert_obs_stb_zoo(algo=algo_list[5], env_name='MountainCar-v0', folder='logs/',
#                                              timesteps=5000)
# print(result)
