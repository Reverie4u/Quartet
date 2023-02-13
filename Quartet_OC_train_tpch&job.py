#!/usr/bin/python
# coding:utf-8
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from nets.fcnn import FCNN
from utils.readCsv import readcsv

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

    # partition the data set
    train = np.empty((0, n_feature))
    train_target = np.empty(0)
    test = np.empty((0, n_feature))
    test_target = np.empty(0)
    size = 73
    count = 9
    batch = count * 20
    for i in range(size):
        for j in range(count):
            train = np.append(train, data[i * batch + j * 20:i * batch + j * 20 + 14], axis=0)
            train_target = np.append(train_target, label[i * batch + j * 20:i * batch + j * 20 + 14], axis=0)
            test = np.append(test, data[i * batch + j * 20 + 14:i * batch + j * 20 + 20], axis=0)
            test_target = np.append(test_target, label[i * batch + j * 20 + 14:i * batch + j * 20 + 20], axis=0)

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
    job_data = readcsv("./data/Quartet_data_job.csv")
    job_label = readcsv("./data/Quartet_label_job.csv")
    job_data = min_max_scaler.transform(job_data)
    tpch_test = test
    tpch_test_target = test_target
    train = np.append(train, job_data[job_train_index], axis=0)
    train_target = np.append(train_target, job_label[job_train_index], axis=0)
    test = np.append(test, job_data[job_test_index], axis=0)
    test_target = np.append(test_target, job_label[job_test_index], axis=0)
    job_test = job_data[job_test_index]
    job_test_target = job_label[job_test_index]
    # train = job_data[job_train_index]
    # train_target = job_label[job_train_index]
    # test = job_data[job_test_index]
    # test_target = job_label[job_test_index]
    print(train.shape)
    print(test.shape)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_len = train_target.shape[0]
    test_len = test_target.shape[0]
    job_test_len = job_test_target.shape[0]
    tpch_test_len = tpch_test_target.shape[0]
    print("Training set length: ", train_len)
    print("Test set length: ", test_len)
    print("Test set tpch length: ", tpch_test_len)
    print("Test set job length: ", job_test_len)
    # torch.manual_seed(100)
    torch.manual_seed(100)

    training_step = 10000

    model = net.to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # loss function
    loss_func = nn.CrossEntropyLoss()

    train = torch.FloatTensor(train).to(device)
    train_target_tensor = torch.LongTensor(train_target).to(device)

    test = torch.FloatTensor(test).to(device)
    test_target_tensor = torch.LongTensor(test_target).to(device)

    tpch_test = torch.FloatTensor(tpch_test).to(device)
    job_test = torch.FloatTensor(job_test).to(device)
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
        tpch_test_pre = model(tpch_test)
        job_test_pre = model(job_test)
        tpch_acc = (np.argmax(tpch_test_pre.data.cpu().numpy(), axis=1) == tpch_test_target).sum() / tpch_test_len
        job_acc = (np.argmax(job_test_pre.data.cpu().numpy(), axis=1) == job_test_target).sum() / job_test_len
        #############################
        if (step + 1) % 100 == 0:
            print("Iteration %d:" % (step + 1))
            print("\t train_loss=%f, train_acc=%f" % (float(train_loss.data), float(train_acc)))
            print("\t test_loss=%f, test_acc=%f" % (float(test_loss.data), float(test_acc)))
        px.append(step)
        py.append(train_loss.data.cpu())
        pz.append(test_loss.data.cpu())
        py1.append(train_acc)
        pz1.append(test_acc)
        if step + 1 == training_step:
            print("tpch_acc=%f, job_acc=%f" % (tpch_acc, job_acc))
    T2 = time.time()
    print('Training :%ss' % (T2 - T1))
    # drawing loss
    ax1 = plt.subplot(1, 2, 1)
    p1 = ax1.plot(px, py, "r-", lw=1)
    p2 = ax1.plot(px, pz, "b-", lw=1)
    ax1.legend(["train loss", "test loss"], loc='upper right')
    ax2 = plt.subplot(1, 2, 2)
    p3 = ax2.plot(px, py1, "r-", lw=1)
    p4 = ax2.plot(px, pz1, "b-", lw=1)
    ax2.legend(["train acc", "test acc"], loc='upper right')
    plt.savefig("./images/train_OC.png")
