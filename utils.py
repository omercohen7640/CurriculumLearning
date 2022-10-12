import numpy as np
from matplotlib import pyplot as plt

class Task:
    def __init__(self, theta, std):
        self.Std = std
        self.Theta = theta

    def sample_task(self, size=1):
        return np.random.normal(self.Theta, self.Std, size=size)


class Setup:
    def __init__(self, t, n):
        self.T = t
        self.Target_Task=t-1
        self.N = n
        thetas = self.generate_thetas()
        stds = self.generate_stds()
        # here we arange the tasks so the target task will have theta=0 and largest variance
        temp = stds[self.T-1]

        task_with_largest_var = np.argmax(stds)
        stds[self.T-1] = stds[task_with_largest_var]
        stds[task_with_largest_var] = temp
        self.Tasks = [Task(thetas[i], stds[i]) for i in range(self.T)]

    def generate_thetas(self):
        # sample from unifrom distribution (10T/2,-1/\sqrt{N}]U[1/\sqrt{N},10T/2)
        sign = np.random.choice([1, -1], size=self.T, replace=True)
        a = 1/np.sqrt(self.N)
        b = 10*self.T/2
        thetas = sign*((b - a) * np.random.random(size=self.T) + a)
        thetas[self.T-1] = 0
        thetas[np.random.choice(list(range(self.T-1)))]=np.random.choice([1,-1])*1/np.sqrt(self.N)
        return thetas

    def generate_stds(self):
        # sample from uniform distribution [0,T/2)
        stds = self.T/2*np.random.random(size=self.T)
        return stds

    def sample_tasks(self,tasks=-1,size=1):
        """sample from tasks, if tasks=-1 sample from all tasks"""
        task_samples = []
        tasks = [tasks] if isinstance(tasks,int) and tasks != -1 else tasks
        for i in range(len(self.Tasks)):
            if tasks == -1 or i in tasks:
                task_samples.append(self.Tasks[i].sample_task(size=size))
        return np.row_stack(task_samples)

    def thetas(self):
        all_thetas = [task.Theta for task in self.Tasks]
        return all_thetas

    def stds(self):
        all_stds = [task.Std for task in self.Tasks]
        return all_stds

def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)


def plot_tasks(setup):
    plt.figure()
    thetas = np.array([t.Theta for t in setup.Tasks])
    stds = np.array([t.Std for t in setup.Tasks])
    plt.hlines(0, np.min(thetas-stds)-10, np.max(thetas+stds)+10)  # Draw a horizontal line
    y = np.zeros(np.shape(thetas))  # Make all y values the same
    col = get_cmap(np.shape(thetas)[0])
    for i in range(len(thetas)):
        plt.plot((thetas - stds)[i], y[i], '<', c=col(i))  # Plot a line at each location specified in a
        plt.plot((thetas + stds)[i], y[i], '>', c=col(i))  # Plot a line at each location specified in a
    plt.show()