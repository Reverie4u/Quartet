#!/usr/bin/python
# coding:utf-8

import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from nets.fcnn import FCNN
from utils import writeCsv
from utils.readCsv import readcsv


def numCount(arr, target):
    arr = np.array(arr)
    mask = (arr == target)
    arr_new = arr[mask]
    return arr_new.size


net = FCNN(13)
if __name__ == '__main__':
    # import data
    data = readcsv("./data/Quartet_data_tpch.csv")
    label = readcsv("./data/Quartet_label_tpch.csv")
    train_len = data.shape[0]
    n_feature = data.shape[1]
    # normalization
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(data)
    data = min_max_scaler.transform(data)
    joblib.dump(min_max_scaler, 'tpch_scaler.pkl')
    # partition the data set
    dataset1_train = np.empty((0, n_feature))
    dataset1_train_target = np.empty(0)
    dataset1_test = np.empty((0, n_feature))
    dataset1_test_target = np.empty(0)
    dataset2_train = np.empty((0, n_feature))
    dataset2_train_target = np.empty(0)
    dataset2_test = np.empty((0, n_feature))
    dataset2_test_target = np.empty(0)
    start_size = 0
    end_size = 30
    max_size = 60
    count = 9
    batch = count * 20
    for i in range(start_size, end_size):
        for j in range(count):
            dataset1_train = np.append(dataset1_train, data[i * batch + j * 20:i * batch + j * 20 + 14], axis=0)
            dataset1_train_target = np.append(dataset1_train_target, label[i * batch + j * 20:i * batch + j * 20 + 14],
                                              axis=0)
            dataset1_test = np.append(dataset1_test, data[i * batch + j * 20 + 14:i * batch + j * 20 + 20], axis=0)
            dataset1_test_target = np.append(dataset1_test_target,
                                             label[i * batch + j * 20 + 14:i * batch + j * 20 + 20], axis=0)

    for i in range(end_size, max_size):
        for j in range(count):
            dataset2_train = np.append(dataset2_train, data[i * batch + j * 20:i * batch + j * 20 + 14], axis=0)
            dataset2_train_target = np.append(dataset2_train_target,
                                              label[i * batch + j * 20:i * batch + j * 20 + 14], axis=0)
            dataset2_test = np.append(dataset2_test, data[i * batch + j * 20 + 14:i * batch + j * 20 + 20], axis=0)
            dataset2_test_target = np.append(dataset2_test_target,
                                             label[i * batch + j * 20 + 14:i * batch + j * 20 + 20], axis=0)
    train = dataset2_train
    train_target = dataset2_train_target
    test = dataset2_test
    test_target = dataset2_test_target
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_len = train_target.shape[0]
    test_len = test_target.shape[0]
    dataset1_test_len = dataset1_test_target.shape[0]
    dataset2_test_len = dataset2_test_target.shape[0]
    print("Training set length: ", train_len)
    print("Test set length: ", test_len)
    torch.manual_seed(100)
    training_step = 5000

    model = net.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # loss function
    loss_func = nn.CrossEntropyLoss()

    train = torch.FloatTensor(train).to(device)
    train_target_tensor = torch.LongTensor(train_target).to(device)

    test = torch.FloatTensor(test).to(device)
    test_target_tensor = torch.LongTensor(test_target).to(device)
    dataset1_test = torch.FloatTensor(dataset1_test).to(device)
    dataset2_test = torch.FloatTensor(dataset2_test).to(device)

    # start training
    px = []
    py = []
    pz = []
    py1 = []
    pz1 = []
    T1 = time.time()
    for step in range(training_step):
        model.train()
        train_pre = model(train)
        train_loss = loss_func(train_pre, train_target_tensor)
        train_acc = (np.argmax(train_pre.data.cpu().numpy(), axis=1) == train_target).sum() / train_len
        # backward
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        #############################
        model.eval()
        test_pre = model(test)
        test_loss = loss_func(test_pre, test_target_tensor)
        test_acc = (np.argmax(test_pre.data.cpu().numpy(), axis=1) == test_target).sum() / test_len
        dataset1_test_pre = model(dataset1_test)
        dataset1_acc = (np.argmax(dataset1_test_pre.data.cpu().numpy(),
                                  axis=1) == dataset1_test_target).sum() / dataset1_test_len
        #############################
        if (step + 1) % 100 == 0:
            print("Iteration %d:" % (step + 1))
            print("\t train_loss=%f, train_acc=%f" % (float(train_loss.data), float(train_acc)))
            print("\t test_loss=%f, test_acc=%f" % (float(test_loss.data), float(test_acc)))
        if step + 1 == training_step:
            print("\t dataset1_acc=%f" % (dataset1_acc))
        px.append(step)
        py.append(train_loss.data.cpu())
        pz.append(test_loss.data.cpu())
        py1.append(train_acc)
        pz1.append(test_acc)
    T2 = time.time()
    print('Training time:%ss' % (T2 - T1))
    # store train_loss and test_loss
    writeCsv.writecsv("./data/OC_tpch_train_loss.csv", np.array(py))
    writeCsv.writecsv("./data/OC_tpch_test_loss.csv", np.array(pz))
    # store train_acc and test_acc
    writeCsv.writecsv("./data/OC_tpch_train_acc.csv", np.array(py1))
    writeCsv.writecsv("./data/OC_tpch_test_acc.csv", np.array(pz1))
    # drawing loss
    ax1 = plt.subplot(1, 2, 1)
    p1 = ax1.plot(px, py, "r-", lw=1)
    p2 = ax1.plot(px, pz, "b-", lw=1)
    ax1.legend(["train loss", "test loss"], loc='upper right')
    ax2 = plt.subplot(1, 2, 2)
    p3 = ax2.plot(px, py1, "r-", lw=1)
    p4 = ax2.plot(px, pz1, "b-", lw=1)
    ax2.legend(["train acc", "test acc"], loc='upper right')
    plt.savefig("./images/train_OC_tpch.png")
