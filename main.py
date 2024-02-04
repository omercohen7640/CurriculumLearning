import numpy as np
import argparse
import datetime
import pickle
import json
import utils
from config import config
from utils import Setup, calc_precision_score, calc_recall_score
import matplotlib.pyplot as plt

from tqdm import tqdm
import os
import seaborn as sb
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 15,
    "lines.linewidth": 2
})
CLOSE = 0
MEDIUM = 1
FAR = 2

OUTPUT_PATH = f"/Users/omercohen/PycharmProjects/CurriculumLearning/output/{config['exp_type']}"
from Algorithms import Xu_Tewari, naive_algo, weak_oracle, strong_oracle, task_elimination_algo
algorithms = ["weak_oracle","ours","Xu","naive","strong_oracle"]
colors = {"ours": 'b', "Xu": 'g', "naive": 'r', "weak_oracle": 'c', "strong_oracle": 'm', "close": 'b', "middle": 'g',
          "far": 'r'}
rng = np.random.default_rng(seed=42)

def save_exp(figs=None, data=None):
    path = open_dir()
    if figs is None:
        fig_path = os.path.join(path, 'fig.pdf')
        plt.savefig(fname=fig_path, bbox_inches='tight')
    else:
        for i,fig in enumerate(figs):
            fig_path = os.path.join(path, f'fig{i}.pdf')
            fig.savefig(fname=fig_path, bbox_inches='tight')
    cfg_path = os.path.join(path, 'cfg.json')
    with open(cfg_path, 'w') as fp:
        json.dump(config, fp, indent=4, default=lambda o: '<not serializable>')
    if not data is None:
        cfg_path = os.path.join(path, 'data')
        file = open(cfg_path, 'wb')
        pickle.dump(data, file)
    print(f"config:\n{json.dumps(config, indent=4, default=lambda o: '<not serializable>')}")

def open_dir():
    path = os.path.join(OUTPUT_PATH, datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S"))
    os.makedirs(path, exist_ok=True)
    # path = os.path.join('output', str(datetime.datetime.now())+'.png')
    print("path:\n"+str(path))
    return path


def run_algorithm(algo_name, setup, get_tau_alg=False):
    tau_alg = []
    if algo_name == "ours":
        err, best_task, tau_alg = task_elimination_algo(setup, config['delta'], config['beta_bar'], get_tau_alg=get_tau_alg)
    elif algo_name == "Xu":
        err, best_task  = Xu_Tewari(setup)
    elif algo_name == "naive":
        err, best_task = naive_algo(setup)
    elif algo_name == "weak_oracle":
        err, best_task = weak_oracle(setup)
    elif algo_name == "strong_oracle":
        err, best_task = strong_oracle(setup)
    else:
        raise Exception('unfamiliar algorithm')
    if not get_tau_alg:
        return err, best_task
    else:
        return err, best_task, tau_alg


def n_growing(ax=None, rng=None):
    if rng == None:
        rng = np.random.default_rng(seed=42)
    seeds = rng.integers(10000, size=20)
    errors = {}
    for algo_name in algorithms:
        errors[algo_name] = []
    n_vec = np.linspace(config["range_for_n_growing"][0], config["range_for_n_growing"][1], 100)
    for n in tqdm(n_vec):
        for algo_name in algorithms:
            errors_for_n = []
            for seed in seeds:
                rng = np.random.default_rng(seed=seed)
                s = Setup(t=100, n=int(n), rng=rng)
                curr_error, curr_task = run_algorithm(algo_name, s)
                errors_for_n.append(curr_error)
            errors[algo_name].append(errors_for_n)
    if ax is None:
        ax = plt.subplot()
        plot = True
    else:
        plot = False
    for algo_name in algorithms:
        np_array = np.log(np.array(errors[algo_name]))
        mean_array = np_array.mean(axis=1)
        std_array = np_array.std(axis=1)
        ax.fill_between(n_vec, mean_array-std_array, mean_array+std_array, color=colors[algo_name], alpha=0.2)
        ax.plot(n_vec, np_array.mean(axis=1), label=algo_name, color=colors[algo_name])
    ax.legend(fontsize=10)
    ax.set_xlabel('N')
    ax.set_ylabel('$\log(\|\hat{\\theta}-\\theta\|^2)$')
    if plot:
        path = open_dir()
        plt.show()
        plt.savefig(fname=path)


def t_growing():
    rng = np.random.default_rng(seed=42)
    seeds = rng.integers(10000, size=200)
    errors = {}
    for algo_name in algorithms:
        errors[algo_name] = []
    t_vec = np.linspace(100, 10000, 100)
    for t in tqdm(t_vec):
        for algo_name in algorithms:
            errors_for_n = []
            for seed in seeds:
                rng = np.random.default_rng(seed=seed)
                s = Setup(t=int(t),n=int(10e4), rng=rng)
                curr_error, curr_task = run_algorithm(algo_name, s)
                errors_for_n.append(curr_error)
            errors[algo_name].append(errors_for_n)
    ax = plt.subplot(2,1)
    for algo_name in algorithms:
        np_array = np.log(np.array(errors[algo_name]))
        mean_array = np_array.mean(axis=1)
        std_array = np_array.std(axis=1)
        ax[1].fill_between(t_vec, mean_array-std_array, mean_array+std_array, color=colors[algo_name], alpha=0.2)
        ax[1].plot(t_vec, np_array.mean(axis=1), label=algo_name, color=colors[algo_name])
    ax.legend()
    ax[1].set_xlabel('$T$')
    ax[1].set_ylabel('$\log(\|\hat{\\theta}-\\theta\|^2)$')
    path = open_dir()
    ax[2] = utils.plot_betta_function()
    plt.show()
    plt.savefig(fname=path)


def n_growing_beta_distribution():
    plt.rcParams.update({'font.size': 14})
    rng = np.random.default_rng(seed=42)
    s = Setup(t=config["t_for_n_growing"], n=int(sum(config["range_for_n_growing"])/2), rng=rng)
    fig, axes = plt.subplots(3)
    fig.set_figheight(12)
    fig.set_figwidth(10)
    n_growing(axes[0], rng)
    utils.plot_betta_function(s, ax=axes[1])
    utils.plot_histogram(s, axes[2])
    fig.tight_layout()
    plt.show()


def Q_func_changing():
    rng = np.random.default_rng(seed=42)
    seeds = rng.integers(10000, size=config["num_seeds"])
    max_factor = config["max_factor"]
    errors = {}
    p_between_sources = {}
    p_target_sources = {}
    # OMER: I start only with our algorithm
    algorithms = ['ours']
    for algo_name in algorithms:
        errors[algo_name] = []
        p_between_sources[algo_name] = []
        p_target_sources[algo_name] = []
    iter_vec = np.linspace(0, 1, 200)
    for i in tqdm(iter_vec):
        for algo_name in algorithms:
            errors_for_n = []
            p_between_sources_for_n = []
            p_target_sources_for_n = []
            for seed in seeds:
                rng = np.random.default_rng(seed=seed)
                def q_func(t):
                    T = len(t)
                    # farther_half = int(T/2)
                    Q = np.zeros(T)
                    Q[1:] = max_factor*i
                    return Q
                config["Q_function"] = q_func
                s = Setup(t=int(config['constant_t']),n=config["constant_n"], rng=rng)
                curr_error, curr_task = run_algorithm(algo_name, s)
                if curr_task > 0:
                    p_target_sources_for_n.append(1)
                    p_between_sources_for_n.append(curr_task - 1)
                else:
                    p_target_sources_for_n.append(0)
                errors_for_n.append(curr_error)
            errors[algo_name].append(errors_for_n)
            p_between_sources[algo_name].append(np.array(p_between_sources_for_n).mean())
            p_target_sources[algo_name].append(np.array(p_target_sources_for_n).mean())
    fig, axes = plt.subplots(2, 1, sharex='row')
    for algo_name in algorithms:
        np_array = np.log(np.array(errors[algo_name]))
        mean_array = np_array.mean(axis=1)
        std_array = np_array.std(axis=1)
        ci = 1.96*std_array/np.sqrt(config["num_seeds"])
        axes[0].fill_between(max_factor*iter_vec, mean_array-ci, mean_array+ci, color=colors[algo_name], alpha=0.1)
        axes[0].plot(max_factor*iter_vec, np_array.mean(axis=1), label=algo_name, color=colors[algo_name])
        # axes[1].plot(max_factor * iter_vec, p_target_sources[algo_name], label="$p$")
        axes[1].plot(max_factor * iter_vec, p_between_sources[algo_name], label="$q$")
    # fig.suptitle('$Simulation\ 1$')
    # axes[0].legend()
    axes[0].set_ylabel('$\log(\|\hat{\\theta}-\\theta\|^2)$')
    # axes[1].legend()
    axes[1].set_xlabel('$\\tilde{Q}_2^2$')
    axes[1].set_ylabel('Probability')
    save_exp()
    plt.show()


def t_growing_different_locations(get_tau_alg=False):
    rng = np.random.default_rng(seed=42)
    seeds = rng.integers(10000, size=config["num_seeds"])
    errors = {}
    # OMER: I start only with our algorithm
    modes = ['close', 'middle', 'far']
    for mode in modes:
        errors[mode] = []
    t_vec = np.arange(3, 100, 1)
    for i in tqdm(t_vec):
        for mode in modes:
            errors_for_n = []
            for seed in seeds:
                rng = np.random.default_rng(seed=seed)
                def q_func(t):
                    t_num = len(t)
                    q_1 = [config['different_factors']['close']]
                    q_2 = [config['different_factors']['middle']]
                    q_3 = [config['different_factors']['far']]
                    if mode == 'close':
                        q_1 = (t_num-2)*[config['different_factors']['close']]
                    elif mode == 'middle':
                        q_2 = (t_num - 2) * [config['different_factors']['middle']]
                    elif mode == "far":
                        q_3 = (t_num - 2) * [config['different_factors']['far']]
                    return np.concatenate((q_1, q_2, q_3))

                config["Q_function"] = q_func
                s = Setup(t=i, n=config["constant_n"], rng=rng)

                curr_error, curr_task = run_algorithm('ours', s, get_tau_alg=get_tau_alg)
                errors_for_n.append(curr_error)
            errors[mode].append(errors_for_n)
    ax = plt.subplot()
    # ax.set_title('$Simulation\ 2$')
    for mode in modes:
        np_array = np.log(np.array(errors[mode]))
        mean_array = np_array.mean(axis=1)
        std_array = np_array.std(axis=1)
        ci = 1.96*std_array/np.sqrt(config["num_seeds"])
        ax.fill_between(t_vec, mean_array - ci, mean_array + ci, color=colors[mode],
                        alpha=0.1)
        ax.plot(t_vec, np_array.mean(axis=1), label=mode, color=colors[mode])
    ax.legend()
    ax.set_xlabel('T')
    ax.set_ylabel('$\log(\|\hat{\\theta}-\\theta\|^2)$')
    save_exp()
    plt.show()


def power_gamma(separate_betta_precision=True):
    data = []
    rng = np.random.default_rng(seed=42)
    seeds = rng.integers(10000, size=config["num_seeds"])
    if separate_betta_precision:
        # fig1 = plt.figure(1,layout='constrained')
        # ax1 = fig1.add_subplot()
        fig, axes = plt.subplots(1, 2, layout="constrained", figsize=(10,4))
    else:
        fig = plt.figure(layout='constrained')
        subfigs = fig.subfigures(2, 2, wspace=0.01)
    gamma_to_generate_betta = [1, 2, 3]
    if separate_betta_precision:
        gamma_vec = np.linspace(1,4,100)
        precision_arr = []
    else:
        gamma_vec = [1,2,3,4]
    for gamma in tqdm(gamma_vec):
        def q_func(t):
            T = config["constant_t"]
            close_tasks_n = int(np.floor(T*config["close_ratio"]))
            Q = np.zeros(T)
            Q[close_tasks_n:] = (t[:T - close_tasks_n])**gamma
            return Q
        precision_arr_gamma = []
        # recall_arr = []
        for seed in seeds:
            rng = np.random.default_rng(seed=seed)
            config["Q_function"] = q_func
            s = Setup(t=config["constant_t"], n=config["constant_n"], rng=rng)
            weak_oracle_set = s.get_weak_oracle_set(kappa=config['kappa'])
            _, _, tau_alg = run_algorithm("ours", s, True)
            # recall_arr.append(calc_recall_score(y_true=weak_oracle_set, tau_alg=tau_alg))
            precision_arr_gamma.append(calc_precision_score(y_true=weak_oracle_set, tau_alg=tau_alg))
        if separate_betta_precision:
            precision_arr.append(np.array(precision_arr_gamma).mean())
            if gamma in gamma_to_generate_betta:
                tau, beta, tau_min = s.beta_function_equal_source_variance(axes[0], plot=False)
                if gamma == gamma_to_generate_betta[0]:
                    axes[0].axvline(x=tau_min, color='red', label='$\\tau_{min}$')
                    axes[0].plot(tau, tau, label="$\\beta(\\tau)=\\tau$")
                axes[0].plot(tau, beta, label=f"$\gamma={gamma}$")
        else:
            axes = subfigs[int(gamma/2), gamma % 2].subplots(2, 1)
            subfigs[int(gamma/2), gamma % 2].suptitle(f"$\gamma={gamma}$")
            s.beta_function_equal_source_variance(axes[0])
            axes[1].bar(x=['Precision', 'Recall'], height=[np.array(precision_arr_gamma).mean(), np.array(recall_arr).mean()])
            axes[1].set_ylim(0,1.1)
            axes[1].set_ylabel("Score$")
    axes[0].legend()
    axes[0].set_xlabel("$\\tau$")
    axes[0].set_ylabel("$\\beta(\\tau$)")
    if separate_betta_precision:
        # fig2 = plt.figure(2)
        # ax2 = fig2.add_subplot()
        axes[1].plot(gamma_vec, precision_arr)
        axes[1].set_ylabel("Precision")
        axes[1].set_xlabel("$\gamma$")
        save_exp()
        plt.show()
    else:
        save_exp()
        plt.show()


def heat_map():
    data = []
    rng = np.random.default_rng(seed=42)
    seeds = rng.integers(10000, size=config["num_seeds"])
    tasks_vectors = 10*np.array(utils.generate_vectors(10))
    map = np.ones((11, 11))
    for vec in tqdm(tasks_vectors):
        def q_func(t):
            T = len(t)
            Q = np.zeros(T)
            q_1 = [config['different_factors']['close']]
            q_2 = [config['different_factors']['middle']]
            q_3 = [config['different_factors']['far']]
            Q[:vec[CLOSE]] = q_1
            Q[vec[CLOSE]:vec[CLOSE]+vec[MEDIUM]] = q_2
            Q[vec[CLOSE]+vec[MEDIUM]:] = q_3
            return Q
        error_vec = []
        config["Q_function"] = q_func
        for seed in seeds:
            rng = np.random.default_rng(seed=seed)
            s = Setup(t=100, n=config["constant_n"], rng=rng)
            curr_error, curr_task = run_algorithm('ours', s, get_tau_alg=False)
            error_vec.append(np.log(curr_error))
        map[int(vec[CLOSE]/10), int(vec[FAR]/10)] = np.array(error_vec).mean()
    mask = np.ones_like(map) - np.tril(np.ones_like(map))
    ax = plt.subplot()
    # ax.set_title("$Simulation\ 3$")
    sb.heatmap(np.flipud(map), mask=mask,ax=ax)
    data.append(np.flipud(map))
    ax.set_xlabel("Far Tasks")
    ax.set_ylabel('Close Tasks')
    ax.set_xticklabels(np.arange(0,101,10))
    ax.set_yticklabels(np.arange(0,101,10)[::-1])
    save_exp(data=data)
    plt.show()



def main_with_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp_type', dest='exp', help='Which experiment to run')
    args = parser.parse_args()
    if args.exp == 'n_growing':
        n_growing()
    elif args.exp == 't_growing':
        t_growing()
    elif args.exp == 'q_func':
        Q_func_changing()
    elif args.exp == 't_growing_different_locations':
        t_growing_different_locations()
    else:
        rng = np.random.default_rng(seed=42)
        s = Setup(t=config["t_for_n_growing"], n=int(sum(config["range_for_n_growing"]) / 2), rng=rng)
        utils.plot_histogram(s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp_type', dest='exp',default="", help='Which experiment to run')
    args = parser.parse_args()
    if args.exp != '':
        config["exp_type"] = args.exp
    exp_type = config["exp_type"]
    if exp_type == 'n_growing':
        n_growing()
    elif exp_type == 't_growing':
        t_growing()
    elif exp_type == 'q_func':
        Q_func_changing()
    elif exp_type == 't_growing_different_locations':
        t_growing_different_locations()
    elif exp_type == 'power_gamma':
        power_gamma()
    elif exp_type == 'heat_map':
        heat_map()
    else:
        print('unfamiliar experiment type')