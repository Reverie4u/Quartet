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
    data = readcsv("./data/Quartet_data.csv")
    label = readcsv("./data/Quartet_label.csv")
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_len = train_target.shape[0]
    test_len = test_target.shape[0]
    print("Training set length: ", train_len)
    print("Test set length: ", test_len)
    torch.manual_seed(100)
    training_step = 10000

    model = net.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # loss function
    loss_func = nn.CrossEntropyLoss()

    train = torch.FloatTensor(train).to(device)
    train_target_tensor = torch.LongTensor(train_target).to(device)

    test = torch.FloatTensor(test).to(device)
    test_target_tensor = torch.LongTensor(test_target).to(device)

    # start training
    px = []
    py = []
    pz = []
    py1 = []
    pz1 = []
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
    torch.save(model, "Quartet_model.pkl")
    # drawing loss
    ax1 = plt.subplot(1, 2, 1)
    p1 = ax1.plot(px, py, "r-", lw=1)
    p2 = ax1.plot(px, pz, "b-", lw=1)
    ax1.legend(["train loss", "test loss"], loc='upper right')
    ax2 = plt.subplot(1, 2, 2)
    p3 = ax2.plot(px, py1, "r-", lw=1)
    p4 = ax2.plot(px, pz1, "b-", lw=1)
    ax2.legend(["train acc", "test acc"], loc='upper right')
    plt.savefig("./images/train_Quartet.png")
