#!/usr/bin/python
# coding:utf-8
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

import nets.tcnn as tcnn
from nets.fcnn import FCNN
from nets.util import prepare_trees
from utils import writeCsv
from utils.process_txt import read_txt

torch.manual_seed(10)
net = nn.Sequential(
    tcnn.BinaryTreeConv(15, 128),
    tcnn.TreeLayerNorm(),
    tcnn.TreeActivation(nn.ReLU()),
    tcnn.BinaryTreeConv(128, 64),
    tcnn.TreeLayerNorm(),
    tcnn.TreeActivation(nn.ReLU()),
    tcnn.BinaryTreeConv(64, 32),
    tcnn.TreeLayerNorm(),
    tcnn.TreeActivation(nn.ReLU()),
    tcnn.DynamicPooling(),
    FCNN(32)
)


# function to extract the left child of a node
def left_child(x):
    assert isinstance(x, tuple)
    if len(x) == 1:
        # leaf.
        return None
    return x[1]


# function to extract the right child of node
def right_child(x):
    assert isinstance(x, tuple)
    if len(x) == 1:
        # leaf.
        return None
    return x[2]


# function to transform a node into a (feature) vector,
# should be a numpy array.
def transformer(x):
    return np.array(x[0])


def numCount(arr, target):
    arr = np.array(arr)
    mask = (arr == target)
    arr_new = arr[mask]
    return arr_new.size


if __name__ == '__main__':
    data = read_txt("./data/TBOH_data_tpch.txt")
    label = read_txt("./data/TBCNN_label_tpch.txt")
    # partition the data set
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    size = 73
    count = 9
    batch = count * 20
    for i in range(size):
        for j in range(count):
            train_data.extend(data[i * batch + j * 20:i * batch + j * 20 + 14])
            train_label.extend(label[i * batch + j * 20:i * batch + j * 20 + 14])
            test_data.extend(data[i * batch + j * 20 + 14:i * batch + j * 20 + 20])
            test_label.extend(label[i * batch + j * 20 + 14:i * batch + j * 20 + 20])

    ## add job_dataset
    # partition
    job_train_index = [0, 1, 2, 4, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 19, 20, 23, 24, 26, 27, 28, 30, 31, 32, 34, 35,
                       37, 38, 39, 41, 42,
                       44, 45, 46, 48, 49, 51, 52, 53, 55, 56, 57, 59, 60, 61, 62, 65, 66, 68, 69, 70, 72, 73, 75, 76,
                       78, 79, 80, 82, 83,
                       85, 87, 88, 90, 91, 93, 94, 96, 97, 99, 100, 102, 103, 105, 106, 108, 110, 111, 113, 115, 117,
                       119, 121, 123, 125,
                       127, 129, 131, 133, 135]
    job_test_index = [3, 7, 10, 13, 16, 21, 22, 25, 29, 33, 36, 40, 43, 47, 50, 54, 58, 63, 64, 67, 71, 74, 77, 81, 84,
                      86, 89, 92, 95,
                      98, 101, 104, 107, 109, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136]
    job_data = read_txt("./data/TBOH_data_job.txt")
    job_label = read_txt("./data/TBCNN_label_job.txt")
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for i in job_train_index:
        train_data.append(job_data[i])
        train_label.append(job_label[i])
    for i in job_test_index:
        test_data.append(job_data[i])
        test_label.append(job_label[i])
    train_len = len(train_data)
    test_len = len(test_data)
    print("Training set length: ", train_len)
    print("Test set length: ", test_len)

    train_trees = prepare_trees(train_data, transformer, left_child, right_child)
    test_trees = prepare_trees(test_data, transformer, left_child, right_child)
    model = net
    training_step = 30000  # 5000
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    # loss function
    loss_func = nn.CrossEntropyLoss()
    train_label_tensor = torch.LongTensor(train_label)
    test_label_tensor = torch.LongTensor(test_label)
    # start training
    px = []
    py = []
    pz = []
    py1 = []
    pz1 = []
    T1 = time.time()
    for step in range(training_step):
        model.train()
        train_pre = model(train_trees)
        train_loss = loss_func(train_pre, train_label_tensor)
        train_acc = (np.argmax(train_pre.data.numpy(), axis=1) == train_label).sum() / train_len
        # backward
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        #############################
        model.eval()
        T3 = time.time()
        test_pre = model(test_trees)
        T4 = time.time()
        test_loss = loss_func(test_pre, test_label_tensor)
        test_acc = (np.argmax(test_pre.data.numpy(), axis=1) == test_label).sum() / test_len
        #############################
        if (step + 1) % 10 == 0:
            print("Iteration %d:" % (step + 1))
            print("\t train_loss=%f, train_acc=%f" % (float(train_loss.data), float(train_acc)))
            print("\t test_loss=%f, test_acc=%f" % (float(test_loss.data), float(test_acc)))
        if step == training_step - 1:
            print("Predict time:%ss" % (T4 - T3))
        px.append(step)
        py.append(train_loss.data)
        pz.append(test_loss.data)
        py1.append(train_acc)
        pz1.append(test_acc)
        if step + 1 == training_step:
            writeCsv.writecsv("./data/TBOH_test_pre_job.csv", np.argmax(test_pre.data.cpu().numpy(), axis=1))
    T2 = time.time()
    print('Training:%ss' % (T2 - T1))
    torch.save(model, "TBOH_job_model.pkl")
    # store train_loss and test_loss
    writeCsv.writecsv("./data/TBOH_job_train_loss.csv", np.array(py))
    writeCsv.writecsv("./data/TBOH_job_test_loss.csv", np.array(pz))
    # store train_acc and test_acc
    writeCsv.writecsv("./data/TBOH_job_train_acc.csv", np.array(py1))
    writeCsv.writecsv("./data/TBOH_job_test_acc.csv", np.array(pz1))
    ax1 = plt.subplot(1, 2, 1)
    p1 = ax1.plot(px, py, "r-", lw=1)
    p2 = ax1.plot(px, pz, "b-", lw=1)
    ax1.legend(["train loss", "test loss"], loc='upper right')
    ax2 = plt.subplot(1, 2, 2)
    p3 = ax2.plot(px, py1, "r-", lw=1)
    p4 = ax2.plot(px, pz1, "b-", lw=1)
    ax2.legend(["train acc", "test acc"], loc='upper right')
    plt.savefig("./images/train_TBOH_job.png")
