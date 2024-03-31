import numpy as np
import json
import os

from tqdm import tqdm
from datetime import datetime
from ABC.utils import generate_all_bootstrapped_indexes, generate_all_bootstraps, final_process, etimate_reward, check_dir_and_mk, calculate_z_score, save_param
from ABC.priors import BetaPrior, NormPriori
from ABC.proposals import BetaProposal
from ABC.energy_abc import EnergyABCSampler
from ABC.likelihoods import BetaLikelihood
from ToyEnv import ToyExperiment

"""
This is the ToyEnv script for toy experiment
used the proposed Beta with two param vars
"""

"""
params
"""
path_a = "./ABC/a_params/common_params.json"
with open(path_a, 'r') as f:
    common_params = json.load(f)

total_Episodes = common_params[
    'total_Episodes']  # controling the boostrapping (take 5 from 20), meaning how many traj collected,
bootstrap_M = common_params['bootstrap_M']
# total sampling numbers
burnin = common_params['burnin']
thin = common_params['thin']
total_N = common_params['total_N']  # how many indexes we need for each time of optimizing
epsilon = common_params['epsilon']
Running_times = common_params['Running_times']
expert_len = common_params['expert_len']
miu = common_params['miu']
sigma = common_params['sigma']
fix_alpha = common_params['fix_alpha']
priori_type = common_params['priori_type']  # norm or beta, decide the priori

determistic_actions = False
reward_estimate = True

# used for the Norm distribution proposal

"""
preparing files and envs
"""

# root_dir = "../runs/paper/bigtable_dis_10/ToyEnv/js/test"
# root_dir = "../reward_estimation/toy/"
# root_dir = "../runs/nips/ABC_beta_2param/toyenv/run20_5_norm_priori/"
root_dir = "../runs/"
parm_dir = root_dir + "params.json"

check_dir_and_mk(root_dir)

save_param(data=common_params, path=parm_dir)

nDCG_list = []
SRCC_list = []

for _ in range(Running_times):

    run_dir = root_dir
    run_name = str(datetime.now())[:-6] + os.path.basename(__file__) + "_" + "miu_" + str(miu) + "_sigma_" + str(
        sigma) + "_" + str(total_Episodes) + "_" + str(
        bootstrap_M) + "_" + str(burnin) + "_" + str(total_N) + "_" + str(thin)
    exp_dir = os.path.join(run_dir, run_name)

    # setup experiment directories
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    expert_policy = [-4.0, 4.0, 4.0, -4.0]
    temp_prob = np.array([[-4, 4, 4, -4], [-3, 3, 3, -3], [-2, 2, 2, -2], [-1, 1, 1, -1], [0, 0, 0, 0], [3, -3, -3, 3]])
    idiot_policy = [-b for b in expert_policy]
    # firstly, we store the states, and actions for different episodes

    te = ToyExperiment(determinism=1, expert_betas=expert_policy, argmax_actions=determistic_actions)
    expert_d = {}
    expert_accu_reward = []
    episode_expert_states = []
    episode_expert_actions = []
    episode_expert_rewards = []
    candidate_true_reward = []
    candidate_acc_true_reward = []
    for po in temp_prob:
        te_temp = ToyExperiment(determinism=1, expert_betas=po, argmax_actions=determistic_actions)
        expert = te_temp.ToyExperiment(expert_len)
        candidate_true_reward.append(expert[2])
    # we could visualize the data here, to show the values of 20 are different:

    print("Generating data:")
    for i in tqdm(range(total_Episodes)):
        expert = te.ToyExperiment(expert_len)
        episode_expert_states.append(expert[0])
        episode_expert_actions.append(expert[1])
        episode_expert_rewards.append(expert[2])

    expert_d["states"] = episode_expert_states
    expert_d["actions"] = episode_expert_actions
    expert_d["rewards"] = episode_expert_rewards

    for candi in candidate_true_reward:
        # for i in range(len(candi)):
        #     flatten = np.array(candi[i]).flatten()
        culmulative = np.sum(candi)
        candidate_acc_true_reward.append(culmulative)
    # print(candidate_acc_true_reward)
    for i in range(len(episode_expert_rewards)):
        flatten = np.array(episode_expert_rewards[i]).flatten()
        culmulative = np.sum(flatten)
        expert_accu_reward.append(culmulative)

    # set the probabilities of each candidate policy
    policies = np.array(
        [
            temp_prob[0],
            temp_prob[1],
            temp_prob[2],
            temp_prob[3],
            temp_prob[4],
            temp_prob[5],
        ]
    )

    test_policy_num = len(policies)
    # setup prior and proposal distributions
    # priors = [BetaPrior(0.5, 0.5)] * test_policy_num
    if priori_type == "beta":
        priors = [BetaPrior(0.5, 0.5)] * test_policy_num
    elif priori_type == "norm":
        priors = [NormPriori(miu, sigma)] * test_policy_num
    proposals = [BetaProposal(alpha=fix_alpha, eps=epsilon)] * test_policy_num
    likelihoods = [BetaLikelihood(alpha=fix_alpha, eps=epsilon, method="JS")] * test_policy_num

    name_list = []
    record_acc = 0

    index_list = generate_all_bootstrapped_indexes(
        total_Episodes=total_Episodes,
        output_num=total_N,
        bootstrap_M=bootstrap_M
    )

    boostrap_item_state_list, boostrap_item_action_list = generate_all_bootstraps(expert_d, index_list)
    store_theta_per_policy = []

    for num in range(test_policy_num):
        energy_obj = EnergyABCSampler(
            burn_in=burnin,
            total_N=total_N,
            policy_name=str(policies[num]),
            env=te,
            expert_d=expert_d,
            thin=thin,
            prior=priors[num],
            proposal=proposals[num],
            likelihood=likelihoods[num]
        )
        energy_return = energy_obj.main(bootstrap_M=bootstrap_M, candidate_policy=policies[num])
        name_list.append(str(policies[num]))

        result = energy_return[0]
        signal = energy_return[1]
        z_scores = calculate_z_score(result)
        z_signal = np.mean(calculate_z_score(result))
        while np.isnan(z_signal):
            print("-----isnan == true, re-running for currentr sample----")
            energy_return = energy_obj.main(bootstrap_M=bootstrap_M, candidate_policy=policies[num])

            result = energy_return[0]
            signal = energy_return[1]
            z_scores = calculate_z_score(result)
            z_signal = np.mean(calculate_z_score(result))

        store_theta_per_policy.append(result)

        record_acc += energy_return[1]
        print("acc_amount:" + str(signal))

    mean_expert = np.mean(expert_accu_reward)
    estimation_result = etimate_reward(mean_expert, store_theta_per_policy, name_list, candidate_acc_true_reward)
    print("estimation_result:")
    print(estimation_result)

    # final process: evaluated_order, NDCG, visualization, saving...
    nDCG_val, SRCC_val, _, _ = final_process(name_list, store_theta_per_policy, exp_dir,
                                                             estimation_result=estimation_result)

    nDCG_list.append(nDCG_val)
    SRCC_list.append(SRCC_val)

result_nDCG = "nDCG: " + str(np.mean(nDCG_list))[:6] + "±" + str(np.std(nDCG_list))[:4]
result_SRCC = "SRCC: " + str(np.mean(SRCC_list))[:6] + "±" + str(np.std(SRCC_list))[:4]
print(result_nDCG)
print(result_SRCC)

with open(file=os.path.join(root_dir, "results"), encoding="utf-8", mode="w+") as rk:
    rk.writelines(result_nDCG + "\n")
    rk.writelines(result_SRCC + "\n")
