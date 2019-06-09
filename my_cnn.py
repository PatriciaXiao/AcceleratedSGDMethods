# -*- coding: utf-8 -*-
# reference:
#     https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
#     https://github.com/uclaml/GOSE/blob/master/cnn_gose.py
# Zhiping Xiao (Patricia) on June 8 2019

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader
# from PIL import Image
from torch.autograd import Variable
import copy
import numpy as np

import time

import pandas as pd
import sys
import os

import argparse

#num_cores = 0
#torch.set_num_threads(num_cores)
num_cores = torch.get_num_threads()

print("number of subthreads: {0}\n".format(num_cores))

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# https://docs.python.org/3/howto/argparse.html
# parser.add_argument("-v", "--verbose", help="output verbosity", action="store_true")

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

DATA_ROOT = "../data"
RESULT_DIR = "./result/"

parser = argparse.ArgumentParser(description='ECE236C 2019 Spring Code')

parser.add_argument("-d", "--dataset", type=str, default="MNIST", choices=["MNIST", "CIFAR"],
                    help="normalize the input data points or not")
parser.add_argument("-v", "--verbose", type=str2bool, default=True,
                    help="verbose during training or not")
parser.add_argument("-vi", "--verbose_report_interval", type=int, default=50,
                    help="the value of momentum")
parser.add_argument("-e", "--epoch", type=int, default=5, # 50
                    help="the number of training epochs")
parser.add_argument("-m", "--momentum", type=float, default=0.9, #0
                    help="the value of momentum")
parser.add_argument("-l", "--loss_function", type=str, default="CrossEntropyLoss", choices=["CrossEntropyLoss", "MSELoss"],
                    help="loss function")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.01,
                    help="learning rate")
parser.add_argument("-n", "--normalize", type=str2bool, default=True, # False
                    help="normalize the input data points or not")
parser.add_argument("-log", "--log_file", type=str2bool, default=True,
                    help="log the results into files or not")
parser.add_argument("-logs", "--log_step", action="store_true", help="if we log by steps")
parser.add_argument("-b", "--batch_size", type=int, default=128,
                    help="the training batch size")
parser.add_argument("-bp", "--batch_size_power", type=int, default=16,
                    help="the training batch size for power method")
parser.add_argument("-bt", "--batch_size_test", type=int, default=1024,
                    help="the training batch size")
parser.add_argument("-o", "--optimizer", type=str, default="SGD", choices=["SGD", "Adam", "Adagrad"],
                    help="choice of optimizer")
parser.add_argument('--NO_CUDA', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument("-a", "--algorithm", type=str, default="standard", choices=["standard", "AGD", "GOSE", "ANCM", "combined"],
                    help="choice of algorithm")
parser.add_argument('--NORM_THRESHOLD', type=float, default=0.001, metavar='LR',
                    help='threshold for gradient norm (default: 0.001)')

args = parser.parse_args()

n_epochs = args.epoch
learning_rate = args.learning_rate
momentum = args.momentum # 0, 0.5, 0.9
report_interval = args.verbose_report_interval
normalize_flag = args.normalize
normalize_string = "" if normalize_flag else "_woutnorm"
log_flag = args.log_file
log_step = args.log_step
verbose = args.verbose
DATASET_NAME = args.dataset
BATCH_SIZE = args.batch_size
BATCH_SIZE_POWER = args.batch_size_power
TEST_BATCH_SIZE = args.batch_size_test
LOSS_OPT = args.loss_function
optimizer_choice = args.optimizer
shuffle_data = True
algorithm = args.algorithm
NORM_THRESHOLD = args.NORM_THRESHOLD

args.cuda = not args.NO_CUDA and torch.cuda.is_available()
if algorithm == "ANCM":
    assert LOSS_OPT == "CrossEntropyLoss"
    assert optimizer_choice == "SGD"

if log_flag:
    loss_file = "{5}_loss_{0}_{1}_{2}momentum_{3}{4}.csv".format(DATASET_NAME, optimizer_choice, momentum, LOSS_OPT, normalize_string, algorithm)
    loss_file_data = {"step": [], "training loss": [], "testing acc": []}
    time_file = "{5}_time_{0}_{1}_{2}momentum_{3}{4}.csv".format(DATASET_NAME, optimizer_choice, momentum, LOSS_OPT, normalize_string, algorithm)
    time_file_data = {"epoch": [], "running time": [], "testing acc": []}


class MyTimer:
    def __init__(self):
        self.start = time.time()
        # self.end = time.time()
        self.count = 0. # self.end - self.start
    def reset(self):
        self.start = time.time()
        self.count = 0.
    def pause(self):
        self.count += time.time() - self.start
    def resume(self):
        self.start = time.time()
    def end(self):
        self.count += time.time() - self.start
        return self.count


class mnist:
    def __init__(self):
        self.normalizer = transforms.Normalize((0.1307,), (0.3081,))
        self.dataset = MNIST
        self.batch_size_train = BATCH_SIZE
        self.batch_size_train_power = BATCH_SIZE_POWER 
        self.batch_size_test = TEST_BATCH_SIZE
        self.dim_final = 576 # 24^2
        self.dim = 1
        self.n_class = 10
        self.classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

class cifar:
    def __init__(self):
        self.normalizer = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.dataset = CIFAR10
        self.batch_size_train = BATCH_SIZE
        self.batch_size_train_power = BATCH_SIZE_POWER 
        self.batch_size_test = TEST_BATCH_SIZE
        self.dim_final = 1024 # 32^2
        self.dim = 3
        self.n_class = 10
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def load_data(data_name="MNIST",normalize=False):
    assert data_name in ["MNIST", "CIFAR"], "unconformed dataset name"
    if data_name == "MNIST":
        data = mnist()
    elif data_name == "CIFAR":
        data = cifar()
    transform_commands = [transforms.ToTensor()]
    if normalize:
        transform_commands.append(data.normalizer)

    train_loader = DataLoader(
            data.dataset(DATA_ROOT, 
                    train=True, download=True, 
                    transform=transforms.Compose(transform_commands)
                ),
            batch_size=data.batch_size_train, 
            shuffle=shuffle_data #False #True
        )

    test_loader = DataLoader(
            data.dataset(DATA_ROOT, 
                    train=False, download=True,
                    transform=transforms.Compose(transform_commands)
                ),
            batch_size=data.batch_size_test, 
            shuffle=shuffle_data #False # True
        )
    if algorithm == "ANCM":
        train_loader_power = DataLoader(
                data.dataset(DATA_ROOT, 
                        train=True, download=True, 
                        transform=transforms.Compose(transform_commands)
                    ),
                batch_size=data.batch_size_train_power, 
                shuffle=shuffle_data #False #True
            )
    else:
        train_loader_power = None

    return data, train_loader, test_loader, train_loader_power


class MyCNN(nn.Module):
    def __init__(self, data_dim, n_class, dim_final):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1 or 3, 8m, 8m)
            nn.Conv2d(
                in_channels=data_dim,       # input dimensions
                out_channels=32,            # n_filters (output dimensions)
                kernel_size=5,              # filter size
                stride=1,                   
                padding=2,                  # same size after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (32, 8m, 8m)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(                   # choose max values in 2x2 areas
                kernel_size=2,              # output shape (32, 4m, 4m)
                stride=2,                   # by default stride=2 when kernel_size=2
            ),    
        )
        self.conv2 = nn.Sequential(         # input shape (32, 4m, 4m)
            nn.Conv2d(
                in_channels=32,             # input dimensions
                out_channels=64,            # n_filters (output dimensions)
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (64, 4m, 4m)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(
                kernel_size=2,              # output shape (64, 2m, 2m)
                stride=2,                   # by default stride=2 when kernel_size=2
            ),    
        )
        self.conv3 = nn.Sequential(         # input shape (64, 2m, 2m)
            nn.Conv2d(
                in_channels=64,             # input dimensions
                out_channels=64,            # n_filters (output dimensions)
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (64, 2m, 2m)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(
                kernel_size=2,              # output shape (64, m, m)
                stride=2,                   # by default stride=2 when kernel_size=2
            ),    
        )
        self.out = nn.Linear(dim_final, n_class)   # fully connected layer, output 10 classes, dim_final = m*m*64

    def forward(self, x):
        #print("input: ", x.shape)
        x = self.conv1(x)
        # print("after conv1: ", x.shape)
        x = self.conv2(x)
        # print("after conv2: ", x.shape)
        x = self.conv3(x)
        # print("after conv3: ", x.shape)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        # print("after reshaping: ", x.shape)
        output = self.out(x)
        return output

    def partial_grad(self, input_data, target, loss_function):
        output_data = self.forward(input_data)
        loss_partial = loss_function(output_data, target)
        loss_partial.backward()
        return loss_partial

    def calculate_loss_grad(self, dataset, loss_function, number_batch=8):
        large_batch_loss = 0.0
        large_batch_norm = 0.0

        for data_i, data in enumerate(dataset):
            # only calculate the sub-sampled large batch
            if data_i > number_batch - 1:
                break

            inputs, labels = data
            # wrap data and target into variable
            if args.cuda:
                inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            large_batch_loss += (1.0 / number_batch) * self.partial_grad(inputs, labels, loss_function).item() #data[0]

        # calculate the norm of the large-batch gradient
        for param in self.parameters():
            large_batch_norm += param.grad.data.norm(2) ** 2

        large_batch_norm = np.sqrt(large_batch_norm) / number_batch

        return large_batch_loss, large_batch_norm

    def power_method(self, dataset_power, loss_function, epoch_power=8, inner_iteration_power=16, lr_power=0.5, eta_neg_stepsize = 0.5, normalization_lambda_power=5.0):
        # load the parameters from args
        n_epoch = epoch_power
        inner_iteration = inner_iteration_power
        learning_rate = lr_power
        eta_neg = eta_neg_stepsize
        lambda_power = normalization_lambda_power

        # record the starting point x_0
        start_net = copy.deepcopy(self)
        # construct the iter point y_t
        iter_net = copy.deepcopy(self)
        # construct the auxiliary point z_1
        iter_net_aux_1 = copy.deepcopy(self)
        # construct the auxiliary point z_2
        iter_net_aux_2 = copy.deepcopy(self)
        # generate random vector y_0 with unit norm
        norm_iter = 0.0
        for param_iter in iter_net.parameters():
            param_iter.data = torch.randn(param_iter.data.size())
            norm_iter += param_iter.data.norm(2) ** 2
        norm_iter = np.sqrt(norm_iter)

        for param_iter in iter_net.parameters():
            param_iter.data /= norm_iter

        if args.cuda:
            iter_net.cuda()

        # estimate_value represents v^{T}Hv
        estimate_value = 0.0
        # SCSG for PCA
        for epoch in range(n_epoch):
            # set estimate_value equal to 0
            estimate_value = 0.0
            # set the inner iteration
            num_data_pass = inner_iteration
            # zero net_aux for sum up
            for param in iter_net_aux_1.parameters():
                param.data = torch.zeros(param.data.size())
            if args.cuda:
                iter_net_aux_1.cuda()
            iter_net_vr = copy.deepcopy(iter_net)
            # calculate the large batch Hessian vector product
            for data_iter, data in enumerate(dataset_power):
                if data_iter > num_data_pass - 1:
                    break
                # get the input and label
                inputs, labels = data
                # wrap data and target into variable
                if args.cuda:
                    input_data, labels = Variable(inputs).cuda(), Variable(labels).cuda()
                else:
                    input_data, labels = Variable(inputs), Variable(labels)
                # zero the gradient
                start_net.zero_grad()
                # get the output
                outputs = start_net.forward(input_data)
                # define the loss for calculating Hessian-vector product
                loss_self_defined = loss_function(outputs, labels)
                # compute the gradient
                grad_params = torch.autograd.grad(loss_self_defined, start_net.parameters(), create_graph=True)
                # compute the Hessian-vector product
                inner_product = 0.0
                for param_vr, param_grad in zip(iter_net_vr.parameters(), grad_params):
                    inner_product += torch.sum(param_vr * param_grad)
                h_v_vr = torch.autograd.grad(inner_product, start_net.parameters(), create_graph=True)

                ### sum up the hessian-vector product Hv and lambda * I
                # sum up Hv
                for param_h_v, param_aux_1, param_aux_2 in zip(h_v_vr, iter_net_aux_1.parameters(), iter_net_aux_2.parameters()):
                    param_aux_2 = param_h_v
                    param_aux_1.data -= param_aux_2.data / (num_data_pass * lambda_power)
                # sum up lambda * I
                for param_aux_1, param_iter_vr in zip(iter_net_aux_1.parameters(), iter_net_vr.parameters()):
                    param_aux_1.data += param_iter_vr.data / num_data_pass

            # large-batch term
            iter_net_vr = copy.deepcopy(iter_net_aux_1)
            # sgd term
            iter_net_pre_vr = copy.deepcopy(iter_net)
            # inner iteration
            num_data_pass = inner_iteration

            # inner update
            for data_iter, data in enumerate(dataset_power):

                if data_iter > num_data_pass - 1:
                    break
                # get the input and label
                inputs, labels = data
                # wrap data and target into variable
                if args.cuda:
                    input_data, labels = Variable(inputs).cuda(), Variable(labels).cuda()
                else:
                    input_data, labels = Variable(inputs), Variable(labels)

                # zero the gradient
                start_net.zero_grad()
                outputs = start_net.forward(input_data)

                loss_self_defined = loss_function(outputs, labels)

                # compute the gradient
                grad_params = torch.autograd.grad(loss_self_defined, start_net.parameters(), create_graph=True)

                # compute the Hessian-vector product for current point
                inner_product = 0.0
                for param_iter, param_grad in zip(iter_net.parameters(), grad_params):
                    inner_product += torch.sum(param_iter * param_grad)

                h_v = torch.autograd.grad(inner_product, start_net.parameters(), create_graph=True)

                # compute the Hessian-vector product for previous one
                inner_product_pre = 0.0
                for param_iter_vr, param_grad in zip(iter_net_pre_vr.parameters(), grad_params):
                    inner_product_pre += torch.sum(param_iter_vr * param_grad)

                h_v_pre_vr = torch.autograd.grad(inner_product_pre, start_net.parameters(), create_graph=True)

                # estimate the curvature
                for param_iter, param_h_v in zip(iter_net.parameters(), h_v):
                    estimate_value += torch.sum(param_iter * param_h_v)

                # print every epoch_len mini-batches
                epoch_len = num_data_pass
                if data_iter % epoch_len == epoch_len - 1:
                    estimate_value = float(estimate_value) / (1.0 * epoch_len)
                    # print('epoch: %d, estimate_value: %.8f' % (epoch, estimate_value))

                # update SCSG
                norm_iter = 0.0
                # power method
                for param_aux_1, param_h_v, param_h_v_pre, param_iter, param_iter_pre_vr, param_iter_vr in zip(
                        iter_net_aux_1.parameters(), h_v, h_v_pre_vr, iter_net.parameters(), iter_net_pre_vr.parameters(),
                        iter_net_vr.parameters()):
                    param_aux_1 = - param_h_v / lambda_power + param_h_v_pre / lambda_power
                    param_iter.data += learning_rate * (
                        param_aux_1.data + param_iter_vr.data + param_iter.data - param_iter_pre_vr.data)

                    norm_iter += param_iter.data.norm(2) ** 2

                # norm of iter_net
                norm_iter = np.sqrt(norm_iter)

                # normalization for iter_net
                for param_iter in iter_net.parameters():
                    param_iter.data /= norm_iter

        num_data_pass = inner_iteration

        # calculate a large batch gradient for choosing direction for negative curvature
        start_net.zero_grad()
        start_net.calculate_loss_grad(dataset_power, loss_function, num_data_pass)

        # update with negative curvature
        direction_value = 0.0

        # if estimate_value < 0, then take a negative curvature step
        if estimate_value < 0.0:
            for param_start, param_iter in zip(start_net.parameters(), iter_net.parameters()):
                # print(param_start.grad.shape) # [16, 3, 3, 3]
                # print(param_iter.shape) # [16, 3, 3, 3]
                # direction_value += torch.dot(param_start.grad, param_iter) # dot could not apply to 4D tensor
                direction_value += torch.dot(
                        param_start.grad.view(-1,), 
                        param_iter.view(-1,)
                    )

            # update the direction value
            direction_value = float(torch.sign(direction_value))
            # print the direction value
            print('direction_value:', float(direction_value))

            # take a negative curvature step along v
            for param_self, param_iter in zip(self.parameters(), iter_net.parameters()):
                param_self.data -= (direction_value * eta_neg) * param_iter.data
            return estimate_value

def test(network, test_loader):
    correct = 0
    total = 0
    for test_x, test_y in test_loader:
        test_output = network(test_x)
        pred_y = torch.max(test_output, 1)[1].data.numpy()
        correct += float((pred_y == test_y.data.numpy()).astype(int).sum()) # / float(test_y.size(0))
        total += float(test_y.size(0))
    accuracy = correct / total
    return accuracy

# https://blog.csdn.net/VictoriaW/article/details/72874637
def one_hot_endcoding(labels_1D, batch_size, n_class):
    labels = labels_1D.view(labels_1D.size(0), -1)
    one_hot = torch.zeros(batch_size, n_class).scatter_(1, labels, 1)
    return one_hot

timer = MyTimer()

data, train_loader, test_loader, train_loader_power = load_data(DATASET_NAME, normalize=normalize_flag) # load_data("CIFAR")  # "MNIST"
network = MyCNN(data.dim, data.n_class, data.dim_final)
print(network)
nesterov = algorithm in ["AGD", "ANCM", "combined"]
if optimizer_choice == "SGD" or nesterov:
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum, nesterov=nesterov)
elif optimizer_choice == "Adagrad":
    optimizer = optim.Adagrad(network.parameters(), lr=learning_rate)
elif optimizer_choice == "Adam":
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

# loss_func = nn.CrossEntropyLoss()
if LOSS_OPT == "CrossEntropyLoss":
    loss_func = nn.CrossEntropyLoss()
    convert_label = False
elif LOSS_OPT == "MSELoss":
    loss_func = nn.MSELoss(reduction='mean') # 'sum'
    convert_label = True

def main_standard():
    test_accuracy = test(network, test_loader)
    print('Epoch: ', 0, '| test accuracy: %.2f' % test_accuracy, '| epoch training time: haven\'t started')
    if log_flag:
        time_file_data["epoch"].append(0)
        time_file_data["running time"].append(0)
        time_file_data["testing acc"].append(test_accuracy)

    for epoch in range(n_epochs):
        timer.reset()
        for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
            output = network(b_x)           # cnn output
            if convert_label:
                b_y = one_hot_endcoding(b_y, b_y.shape[0], data.n_class)

            loss = loss_func(output, b_y)   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients

            timer.pause()
            if step % report_interval == 0:
                test_accuracy = test(network, test_loader)
                if verbose:
                    print("\ttrain loss: %.4f" % loss.data.numpy(), "test accuracy: %.4f" % test_accuracy)
                # log data
                if log_flag and log_step:
                    loss_file_data["step"].append(step)
                    loss_file_data["training loss"].append(loss.data.numpy())
                    loss_file_data["testing acc"].append(test_accuracy)
            timer.resume()

        time_count = timer.end()

        test_accuracy = test(network, test_loader)
        print('Epoch: ', epoch+1, '| test accuracy: %.2f' % test_accuracy, '| epoch training time: %.6f' % time_count)
        if log_flag:
            time_file_data["epoch"].append(epoch+1)
            time_file_data["running time"].append(time_count)
            time_file_data["testing acc"].append(test_accuracy)

    if log_flag:
        loss_df = pd.DataFrame(data=loss_file_data)
        time_df = pd.DataFrame(data=time_file_data)
        loss_df.to_csv(os.path.join(RESULT_DIR, loss_file), sep=",", index=False)
        time_df.to_csv(os.path.join(RESULT_DIR, time_file), sep=",", index=False)

def main_accelerated():
    test_accuracy = test(network, test_loader)
    print('Epoch: ', 0, '| test accuracy: %.2f' % test_accuracy, '| epoch training time: haven\'t started')
    if log_flag:
        time_file_data["epoch"].append(0)
        time_file_data["running time"].append(0)
        time_file_data["testing acc"].append(test_accuracy)

    for epoch in range(n_epochs):
        timer.reset()
        for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader

            # the additional step
            network.zero_grad()
            _, full_grad_norm = network.calculate_loss_grad(train_loader, loss_func)
            if full_grad_norm < NORM_THRESHOLD:
                network.power_method(train_loader_power, loss_func)

            output = network(b_x)           # cnn output
            if convert_label:
                b_y = one_hot_endcoding(b_y, b_y.shape[0], data.n_class)

            loss = loss_func(output, b_y)   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients

            timer.pause()
            if step % report_interval == 0:
                test_accuracy = test(network, test_loader)
                if verbose:
                    print("\ttrain loss: %.4f" % loss.data.numpy(), "test accuracy: %.4f" % test_accuracy)
                # log data
                if log_flag and log_step:
                    loss_file_data["step"].append(step)
                    loss_file_data["training loss"].append(loss.data.numpy())
                    loss_file_data["testing acc"].append(test_accuracy)
            timer.resume()

        time_count = timer.end()

        test_accuracy = test(network, test_loader)
        print('Epoch: ', epoch+1, '| test accuracy: %.2f' % test_accuracy, '| epoch training time: %.6f' % time_count)
        if log_flag:
            time_file_data["epoch"].append(epoch+1)
            time_file_data["running time"].append(time_count)
            time_file_data["testing acc"].append(test_accuracy)

    if log_flag:
        loss_df = pd.DataFrame(data=loss_file_data)
        time_df = pd.DataFrame(data=time_file_data)
        loss_df.to_csv(os.path.join(RESULT_DIR, loss_file), sep=",", index=False)
        time_df.to_csv(os.path.join(RESULT_DIR, time_file), sep=",", index=False)

def main_GOSE(TRACK_INTERVAL=8):
    test_accuracy = test(network, test_loader)
    print('Epoch: ', 0, '| test accuracy: %.2f' % test_accuracy, '| epoch training time: haven\'t started')
    if log_flag:
        time_file_data["epoch"].append(0)
        time_file_data["running time"].append(0)
        time_file_data["testing acc"].append(test_accuracy)

    for epoch in range(n_epochs):
        timer.reset()
        for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader

            output = network(b_x)           # cnn output
            if convert_label:
                b_y = one_hot_endcoding(b_y, b_y.shape[0], data.n_class)

            loss = loss_func(output, b_y)   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients

            if step % TRACK_INTERVAL == 0:
                # the additional step
                network.zero_grad()
                _, full_grad_norm = network.calculate_loss_grad(train_loader, loss_func)
                if full_grad_norm < NORM_THRESHOLD:
                    network.power_method(train_loader_power, loss_func)

            timer.pause()
            if step % report_interval == 0:
                test_accuracy = test(network, test_loader)
                if verbose:
                    print("\ttrain loss: %.4f" % loss.data.numpy(), "test accuracy: %.4f" % test_accuracy)
                # log data
                if log_flag and log_step:
                    loss_file_data["step"].append(step)
                    loss_file_data["training loss"].append(loss.data.numpy())
                    loss_file_data["testing acc"].append(test_accuracy)
            timer.resume()

        time_count = timer.end()

        test_accuracy = test(network, test_loader)
        print('Epoch: ', epoch+1, '| test accuracy: %.2f' % test_accuracy, '| epoch training time: %.6f' % time_count)
        if log_flag:
            time_file_data["epoch"].append(epoch+1)
            time_file_data["running time"].append(time_count)
            time_file_data["testing acc"].append(test_accuracy)

    if log_flag:
        loss_df = pd.DataFrame(data=loss_file_data)
        time_df = pd.DataFrame(data=time_file_data)
        loss_df.to_csv(os.path.join(RESULT_DIR, loss_file), sep=",", index=False)
        time_df.to_csv(os.path.join(RESULT_DIR, time_file), sep=",", index=False)

if algorithm in ["standard", "AGD"]:
    main = main_standard
elif algorithm == "ANCM":
    main = main_accelerated
elif algorithm == "GOSE":
    main = main_GOSE
elif algorithm == "combined":
    main = main_GOSE # combined difference: optimizer

main()