# coding=utf-8
from utils import *
from Algorithms import *
import numpy as np
import wandb
from config import config


def run():
    for seed in config["seeds"]:
        np.random.seed(seed)
        for algo in config["algorithms"]:
            wandb.init(project="CurriculumLearning", entity="omer-technion", group=algo, name=str(algo)+"_"+str(seed))
            wandb.config.update(config)
            for n in range(config["start_N"], config["end_N"]):
                s = Setup(config["tasks"], n)
                if algo == "Xu":
                    min_theta, xu_best = Xu_Tewari(s)
                    wandb.log({"xu_best":xu_best})
                elif algo == "adaptive":
                    min_theta, adaptive_best = adaptive_algo(s)
                    wandb.log({"adaptive_best":adaptive_best})
                elif algo == "naive":
                    min_theta = naive_algo(s)
                diff_theta = s.thetas() - s.thetas()[s.Target_Task]
                variances = (np.array(s.stds())**2)/s.N
                if algo != "lower_bound":
                    err = (min_theta-s.thetas()[s.Target_Task])**2
                else:
                    err = np.min(diff_theta**2 + variances)
                    true_best = np.argmin(diff_theta**2 + variances)
                    wandb.log({"true_best":true_best})
                wandb.log({"error_est": err, "N": n})
            wandb.finish()


if __name__ == '__main__':
        run()


