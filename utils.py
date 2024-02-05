import numpy as np
from matplotlib import pyplot as plt
import argparse
from config import *
from Algorithms import set_r_bar
from matplotlib.ticker import PercentFormatter
import sklearn.metrics as metrics
from itertools import product
from scipy.interpolate import make_interp_spline

def constant_q_func(t):
    res1 = 0.2*np.ones(10)
    res2 = 5*np.ones(990)
    return np.concatenate((res1, res2))


class Task:
    def __init__(self, theta, var, q, rng):
        self.rng = rng
        self.Var = var
        self.Theta = theta
        self.Q = q
    def sample_task(self, size=1):
        if self.Theta.ndim == 0:
            return self.rng.normal(loc=self.Theta, scale=np.sqrt(self.Var), size=size)
        else:
            return self.rng.multivariate_normal(mean=self.Theta, cov=np.diag(self.Var), size=size)

class Setup:
    def __init__(self, t, n, rng, Q_func=None):
        self.rng = rng
        self.T = t
        self.Target_Task = 0
        self.N = n
        self.d = config["dimension"]
        vars = self.generate_stds().T
        thetas, Q_vec = self.generate_thetas()
        self.Q = Q_vec
        if self.d==1:
            self.Tasks = [Task(thetas[0,i], vars[i], Q_vec[i], self.rng) for i in range(self.T+1)]
        else:
            self.Tasks = [Task(thetas[:, i], vars[:, i], Q_vec[i], self.rng) for i in range(self.T+1)]

    def sample_unit_sphere(self, npoints, dim,norm="l2"):
        """generate dim X npoints matrix of npoints points on d unit sphere"""
        vec = self.rng.standard_normal(size=(dim, npoints))
        if norm == "l2":
            vec /= np.linalg.norm(vec, axis=0)
        elif norm == "sum":
            vec /= vec.sum(axis=0)
        else:
            print("Wrong norm selected")
            raise ValueError
        return vec

    def generate_thetas(self):
        # sample the target from d-normal distribution
        target_theta = self.rng.standard_normal(size=self.d)
        # sample the source from d-unit ball (boundary)
        source_thetas = self.sample_unit_sphere(npoints=self.T, dim=self.d, norm='l2')
        Q_vec = config["Q_function"](np.arange(self.T)+1)
        # omer: by Q_vec I mean Q squared
        if config['factor_q_with_target_error'] == True:
            target_error = (config["source_var_equal"]*config["target_var_factor"]*config["dimension"])/config["constant_n"]
            Q_vec = Q_vec*target_error
        # normalize the distances according to Q function
        source_thetas *= np.sqrt(Q_vec)
        Q_vec = np.append([0], Q_vec)
        # set the sources as target + distances
        source_thetas = target_theta[:,np.newaxis] + source_thetas
        all_thetas = np.insert(source_thetas, 0, values=target_theta, axis=1)
        return all_thetas, Q_vec

    def generate_stds(self):
        if config["equal_source_variance"]:
            source_vars = np.array(self.T*[config["source_var_equal"]])
            target_var = config["target_var_factor"] * config["source_var_equal"]
        else:
            source_vars = np.random.uniform(low=config["source_var_range"][0], high=config["source_var_range"][1], size=self.T)
            target_var = config["target_var_factor"]*config["source_var_range"][1]
        if self.d > 1:
            diag = self.rng.dirichlet(np.ones(self.d), size=self.T + 1)
            standard_vars = diag*self.d*np.concatenate([[target_var], source_vars])[:, np.newaxis]
        else:
            standard_vars = np.concatenate([[target_var], source_vars])
        return standard_vars

    def get_weak_oracle_set(self, kappa=1):
        """Return T sized binary array with 1 in indices of eak oracle tasks"""
        threshold = kappa * self.Tasks[0].Var.sum() / self.N
        labels_of_weak_oracle = np.array([1 if q <= threshold else 0 for q in self.Q[1:]])
        return labels_of_weak_oracle


    def beta_function_equal_source_variance(self, ax, nu=1, tau_len=1000, plot=True):
        r_bar = set_r_bar(self, config['beta_bar'])
        delta_bar = config['delta'] / (self.T * r_bar + 2)
        omega = 15*np.log(1 / delta_bar)
        d_sigma = self.Tasks[1].Var.sum()
        d_sigma0 = self.Tasks[0].Var.sum()
        # tau_min = (1 / self.T) * np.ceil(d_sigma0 / d_sigma)
        tau_min = np.min([(1 / self.T) * (d_sigma0 / d_sigma), 1])
        tau = np.linspace(0, 1, tau_len)
        tau1 = tau[tau >= tau_min]
        threshold_vec = (( omega * self.T * tau1 * d_sigma) / (nu * self.N))
        Q_vec = np.array([task.Q for task in self.Tasks[1:]])
        threshold_mat = np.tile(threshold_vec, (Q_vec.shape[0], 1)).T
        Q_mat = np.tile(Q_vec, (tau1.shape[0], 1))
        beta1 = (Q_mat <= threshold_mat).mean(axis=1)
        tau2 = tau[tau < tau_min]
        threshold2 = ((omega * d_sigma0) / (nu * self.N))
        beta2 = len(tau2) * [np.array([task.Q < threshold2 for task in self.Tasks[1:]]).mean()]
        beta = np.concatenate((beta2, beta1))
        if plot:
            ax.plot(tau,beta,label='$\\beta(\\tau)$')
            ax.plot(tau,tau,'k--', label='linear')
            ax.axvline(x=tau_min, color='red', label='$\\tau_{min}$')
            ax.set_ylim((0,1.1 ) )
            ax.set_xlabel("$\\tau$")
            ax.set_ylabel("$\\beta(\\tau)$")
            ax.legend()
            return ax
        else:
            return (tau, beta, tau_min)




def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)


def plot_tasks(setup):
    plt.figure()
    thetas = np.array([t.Theta for t in setup.Tasks])
    stds = np.array([t.Var.sum() for t in setup.Tasks])
    plt.hlines(0, np.min(thetas-stds)-10, np.max(thetas+stds)+10)  # Draw a horizontal line
    y = np.zeros(np.shape(thetas))  # Make all y values the same
    col = get_cmap(np.shape(thetas)[0])
    for i in range(len(thetas)):
        plt.plot((thetas - stds)[i], y[i], '<', c=col(i))  # Plot a line at each location specified in a
        plt.plot((thetas + stds)[i], y[i], '>', c=col(i))  # Plot a line at each location specified in a
    plt.show()


def arg_parser():
    parser = argparse.ArgumentParser(description='Curriculum learning experiment environment')
    parser.add_argument('--grow', type=str, required=False, choices=['T', 'N'])
    parser.add_argument('--mode', type=str, required=False, choices=['uniform','force_alternative_task'])
    parser.add_argument('--grouping_name', type=str, required=False, default="")
    return parser.parse_args()

def sample_uniform(a,b):
    """sample uniform from (-a,-b) and (a,b)"""
    sign = np.random.choice([1, -1])
    return sign * ((b - a) * np.random.random() + a)


def calc_precision_score(y_true, tau_alg):
    """Calculate precision score for weak oracle set predictions"""
    T = y_true.shape[0]
    y_pred = np.zeros(T)
    y_pred[tau_alg-1] = 1
    return metrics.precision_score(y_true, y_pred)

def calc_recall_score(y_true, tau_alg):
    """Calculate recall score for weak oracle set predictions"""
    T = y_true.shape[0]
    y_pred = np.zeros(T)
    y_pred[tau_alg-1] = 1
    return metrics.recall_score(y_true, y_pred)


def generate_vectors(sum_value):
    vectors = []
    for combination in product(range(sum_value + 1), repeat=3):
        if sum(combination) == sum_value:
            vectors.append(combination)
    return vectors

def smooth_beta_func(x,y):
    cs = make_interp_spline(x, y)
    xx = np.linspace(x[0], x[-1], 10)
    yy = [cs(x) for x in xx]
    # yy = np.interp(xx, x, y)
    return xx, yy