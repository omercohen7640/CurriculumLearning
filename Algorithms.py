import numpy as np
from config import config
def Xu_Tewari(setup):
    # OMER: I sample here (T+1/T) samples
    # estimate all T tasks, each task with N/(2T) samples
    samples_n = int(setup.N/(2*(setup.T+1)))
    source_samples = [task.sample_task(size=samples_n) for task in setup.Tasks]
    source_samples = np.array(source_samples)
    # estimate sources
    all_source_estimator = source_samples.mean(axis=1)
    samples_val = setup.Tasks[0].sample_task(size=int(setup.N / 2))  # samples from target
    # choose the best estimation according to validation set.
    if setup.d > 1:
        val_error_vec = [(np.linalg.norm(source_estimator-samples_val, axis=1)**2).mean() for source_estimator in all_source_estimator]
    else:
        val_error_vec = [((source_estimator - samples_val) ** 2).mean() for source_estimator in
                         all_source_estimator]
    best_task = np.argmin(val_error_vec)
    return np.linalg.norm(setup.Tasks[0].Theta-all_source_estimator[best_task])**2, best_task


# Simply estimate theta_T
def naive_algo(setup):
    samples_T = setup.Tasks[0].sample_task(size=int(setup.N))
    estimation = samples_T.mean(axis=0)
    if setup.d > 1:
        return np.linalg.norm(setup.Tasks[0].Theta-estimation)**2, 0
    else:
        return (setup.Tasks[0].Theta-estimation)**2, 0

def weak_oracle(setup):
    """Knows if there are tasks that are better than the target (up to a constant) and chooses randomly from them. chooses the target if there aren't  """
    c_delta = np.log(2/config['delta'])
    if config['equal_source_variance']:
        source_tasks_Q = np.array([task.Q  for task in setup.Tasks[1:]])
        indices_of_better_then_target = np.where(source_tasks_Q <= c_delta*setup.Tasks[0].Var.sum()/setup.N)[0]
        if indices_of_better_then_target.size == 0:
            task_n = 0
        else:
            task_n = setup.rng.choice(indices_of_better_then_target)
        estimation = sample_and_estimate(setup.Tasks[task_n], setup.N)
        if setup.d > 1:
            return np.linalg.norm(setup.Tasks[0].Theta - estimation) ** 2, 0
        else:
            return (setup.Tasks[0].Theta - estimation) ** 2, 0
    else:
        raise Exception("weak oracle not m=implemented for different variances setting")


def strong_oracle(setup):
    tasks_Q_var_vec = [task.Q + task.Var.sum()/setup.N for task in setup.Tasks]
    best_task = np.argmin(tasks_Q_var_vec)
    return tasks_Q_var_vec[best_task], best_task


def lower_bound(setup):
    tasks_Q_var_vec = [task.Q + task.Var / setup.N for task in setup.Tasks]
    original_indciess_of_source_better_then_tagret = np.where(tasks_Q_var_vec >= tasks_Q_var_vec[0])[0]
    tasks_better_then_target_Q_var = tasks_Q_var_vec[original_indciess_of_source_better_then_tagret]
    median_index = np.argsort(tasks_better_then_target_Q_var)[len(tasks_better_then_target_Q_var) // 2]
    best_task = original_indciess_of_source_better_then_tagret[median_index]
    return tasks_Q_var_vec[best_task], best_task


def adaptive_algo_update_task_est(task, task_round_budget, new_total,old_estimator):
    new_samples = task.sample_task(task_round_budget)
    return (1-task_round_budget/new_total)*old_estimator + 1/new_total * new_samples.sum(), new_samples


# "our" algorithm
def task_elimination_algo(setup, delta, beta_bar, get_tau_alg=False):
    r_bar = set_r_bar(setup, beta_bar)
    # OMER: if r_bar<1 we are in small number of tasks
    N_bar = int(setup.N / (r_bar + 2))
    delta_bar = delta/(setup.T*r_bar + 2)
    theta0_hat = sample_and_estimate(task=setup.Tasks[0], n_samples=N_bar)
    prev_tau = np.arange(1, setup.T+1)
    tau_alg = prev_tau
    prev_T = setup.T
    omega = np.log(2/delta_bar)
    # c = 1
    for r in range(r_bar):
        curr_tau = prev_tau
        d_sigma_bar = np.array([(setup.Tasks[t].Var.sum()) for t in prev_tau]).mean()
        threshold = np.max([(d_sigma_bar * prev_T) / N_bar, (setup.Tasks[0].Var.sum()) / N_bar])
        for t in prev_tau:
            N_bar_t_r = np.ceil((N_bar*setup.Tasks[t].Var.sum())/(prev_T*d_sigma_bar)).astype(int)
            theta_t_hat = sample_and_estimate(setup.Tasks[t],N_bar_t_r)
            if np.linalg.norm(theta_t_hat-theta0_hat)**2 >= 10*omega*threshold:
                curr_tau = curr_tau[curr_tau != t]
        prev_tau = curr_tau
        prev_T = curr_tau.shape[0]
        tau_alg = curr_tau
        if prev_T <= setup.Tasks[0].Var.sum()/d_sigma_bar or curr_tau.shape[0] == 0:
            break
    if tau_alg.shape[0] == 0:
       best_task = 0
       estimate = theta0_hat
       tau_alg = []
    else:
        best_task = setup.rng.choice(tau_alg)
        estimate = sample_and_estimate(task=setup.Tasks[best_task], n_samples=N_bar)
    return np.linalg.norm(setup.Tasks[0].Theta - estimate) ** 2, best_task, tau_alg

def sample_and_estimate(task, n_samples):
    samples = task.sample_task(size=n_samples)  # sample_shape=n_samplesXd
    return samples.mean(axis=0)


def set_r_bar(setup, beta_bar):
    if config['r_bar_func'] == 'using_beta':
        d_sigma_bar = np.array([task.Var.sum() for task in setup.Tasks[1:]]).mean()
        tau_min = (setup.Tasks[0].Var.sum())/(setup.T * d_sigma_bar)
        r_bar = np.log(tau_min)/np.log(beta_bar)
    elif config['r_bar_func'] == 'log_T':
            r_bar = np.log(setup.T)
    # omer: I can also choose to use floor
    return np.max((1, np.ceil(r_bar).astype(int)))