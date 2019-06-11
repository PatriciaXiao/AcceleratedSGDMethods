import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

result_dir = "./result/"
# legend_location = "lower right"

class Dataset(object):
    def __init__(self, algorithm, dataset="MNIST", lr=0.02, mode=[1, 2]):
        self.loss_files = ["{0}_loss_{1}_SGD_{2}lr_CrossEntropyLoss_{3}.csv".format(algorithm, dataset, lr, tag) for tag in mode]
        self.acc_files = ["{0}_time_{1}_SGD_{2}lr_CrossEntropyLoss_{3}.csv".format(algorithm, dataset, lr, tag) for tag in mode]
        self.loss_dfs = [pd.read_csv(os.path.join(result_dir, file)) for file in self.loss_files]
        self.acc_dfs = [pd.read_csv(os.path.join(result_dir, file)) for file in self.acc_files]
        self.loss_x = np.array(range(len(self.loss_dfs[0]))) * (self.loss_dfs[0]["step"].values[1] - self.loss_dfs[0]["step"].values[0])
        self.acc_xs = [df["epoch"].values for df in self.acc_dfs]
        self.acc_x = self.acc_xs[0] # self.acc_xs[0][1:]
        self.loss = [df["training loss"].values for df in self.loss_dfs]
        self.acc = [df["testing acc"].values for df in self.acc_dfs] # [df["testing acc"].values[1:] for df in self.acc_dfs]
        self.time = np.array([df["running time"].values for df in self.acc_dfs])
        self.time_list = self.time.flatten()
        self.average_time = np.average([time for time in self.time_list if time > 0])
        #print(self.average_time)
        self.loss_avg = np.mean(self.loss, axis=0)
        # for acc
        self.acc_avg = np.mean(self.acc, axis=0)
        self.acc_max = np.max(self.acc, axis=0)
        self.acc_min = np.min(self.acc, axis=0)
        self.lower_quartile = self.acc_avg - self.acc_min
        self.upper_quartile = self.acc_max - self.acc_avg
        self.acc_err = np.array([self.lower_quartile, self.upper_quartile])

        self.label = algorithm


datasets_modes = [("MNIST", [1, 2], 0.02, "MNIST"), ("CIFAR", [1, 2, 3], 0.03, "CIFAR10")]
algorithm_list = ["standard", "AGD", "GOSE", "ANCM", "combined"]
# Dataset("standard")

for dataset, mode, lr, dlabel in datasets_modes:
    for algorithm in algorithm_list:
        data = Dataset(algorithm, dataset=dataset, lr=lr, mode=mode)
        print("algorithm: {0}, dataset: {1}, running time each epoch: {2}".format(algorithm, dataset, data.average_time))
# exit(0)


print("plotting the average loss")
xlabel = "step #"
ylabel = "training loss"
legend_location = "upper right"
for dataset, mode, lr, dlabel in datasets_modes:
    fig, ax = plt.subplots()
    title = "Average Loss on {0}".format(dlabel)
    filename = "loss_{0}.pdf".format(dlabel)
    for algorithm in algorithm_list:
        data = Dataset(algorithm, dataset=dataset, lr=lr, mode=mode)
        ax.plot(data.loss_x, data.loss_avg, "-", label=data.label)
    ax.set(xlabel=xlabel, ylabel=ylabel,
        title=title)
    ax.legend(loc=legend_location)

    ax.grid()
    fig.savefig(filename)
    plt.show()


print("plotting the accuracy")
xlabel = "epoch #"
ylabel = "test accuracy"
legend_location = "lower right"
errorevery = 1
for dataset, mode, lr, dlabel in datasets_modes:
    fig, ax = plt.subplots()
    title = "Test Accuracy on {0}".format(dlabel)
    filename = "acc_{0}.pdf".format(dlabel)
    for algorithm in algorithm_list:
        data = Dataset(algorithm, dataset=dataset, lr=lr, mode=mode)
        ax.errorbar(data.acc_x, data.acc_avg, yerr=data.acc_err, label=data.label, errorevery=errorevery)
    ax.set(xlabel=xlabel, ylabel=ylabel,
        title=title)
    ax.legend(loc=legend_location)

    ax.grid()
    fig.savefig(filename)
    plt.show()






