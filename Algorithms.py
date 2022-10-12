import numpy as np


def Xu_Tewari(setup):
    # estimate all T tasks, each task with N/(2T) samples
    samples_n = int(setup.N/(2*setup.T))
    samples_mat = setup.sample_tasks(tasks=-1, size=samples_n)
    theta_estimations = 1/samples_n * samples_mat @ np.ones((samples_n, 1))
    # choose the best estimation according to validation set.
    samples_val = setup.sample_tasks(tasks=setup.Target_Task, size=int(setup.N/2))
    samples_val_expand = samples_val.repeat(setup.T, axis=0)
    theta_estimations_expand = theta_estimations.repeat(int(setup.N/2), axis=1)
    diff_mat = samples_val_expand - theta_estimations_expand
    diff_mat = diff_mat**2
    diff_mat = diff_mat.mean(axis=1)
    return theta_estimations[np.argmin(diff_mat)][0], np.argmin(diff_mat)


# Simply estimate theta_T
def naive_algo(setup):
    samples_T = setup.sample_tasks(tasks=setup.Target_Task, size=setup.N)
    estimation = samples_T.mean()
    return estimation


def adaptive_algo_update_task_est(task,task_round_budget,new_total,old_estimator):
    new_samples = task.sample_task(task_round_budget)
    return (1-task_round_budget/new_total)*old_estimator + 1/new_total * new_samples.sum(), new_samples

# "our" algorithm

def adaptive_algo(setup):
    available_source_tasks = list(range(setup.T-1)) # all tasks without the Target task
    rounds = int(np.log2(setup.T))
    round_budget = int(setup.N/np.log2(setup.T))
    estimators = np.zeros(setup.T)
    total_samples = np.zeros(setup.T)
    Target_samples=np.array([])
    for i in range(rounds):
        target_est_sample_n = int(round_budget/2)
        if len(available_source_tasks) == 0:
            samples_left = round_budget*(rounds-i)
            total_samples[setup.Target_Task] += samples_left
            estimators[setup.Target_Task],_ = adaptive_algo_update_task_est(setup.Tasks[setup.Target_Task],
                                                                          samples_left,
                                                                          total_samples[setup.Target_Task],
                                                                          estimators[setup.Target_Task])
            return estimators[setup.Target_Task], setup.Target_Task
        available_source_tasks_temp = available_source_tasks.copy()
        total_samples[setup.Target_Task] += target_est_sample_n
        estimators[setup.Target_Task], new_Target_samples = adaptive_algo_update_task_est(setup.Tasks[setup.Target_Task], target_est_sample_n, total_samples[setup.Target_Task],estimators[setup.Target_Task])
        Target_samples=np.concatenate((Target_samples, new_Target_samples))
        available_source_tasks_n = len(available_source_tasks)
        source_tasks_round_budget = int(round_budget/(2*available_source_tasks_n))
        for task_i in available_source_tasks:
            total_samples[task_i] += source_tasks_round_budget
            estimators[task_i],_ = adaptive_algo_update_task_est(setup.Tasks[task_i], source_tasks_round_budget, total_samples[task_i], estimators[task_i])
            # maybe use estimator for the variance
            # if (estimators[task_i]-estimators[setup.Target_Task])**2 >= 6*setup.Tasks[setup.Target_Task].Std**2/(round_budget*(i+1)):
            if (estimators[task_i] - estimators[setup.Target_Task]) ** 2 >= 6 * Target_samples.std()** 2 / (total_samples[setup.Target_Task]/2):
                available_source_tasks_temp.remove(task_i)
        available_source_tasks = available_source_tasks_temp.copy()
    if len(available_source_tasks) > 0:
        final_estimator = 0
        available_samples_sum = total_samples[available_source_tasks].sum()
        for task_i in available_source_tasks:
            final_estimator += available_samples_sum/total_samples[task_i] * estimators[task_i]
    else:
        final_estimator = estimators[setup.Target_Task]
    return final_estimator, available_source_tasks


def lower_bound(setup):
    thetas = setup.thetas()
    theta_T = thetas[setup.Target_Task]
    variances = np.array(setup.stds())**2/setup.N
    errors = (theta_T-thetas)**2 + variances
    return np.min(errors), np.argmin(errors)