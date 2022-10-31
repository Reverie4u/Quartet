import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

import nets.tcnn as tcnn
from nets.fcnn import FCNN
from nets.util import prepare_trees
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


if __name__ == '__main__':
    data = read_txt("./data/TBRT_data.txt")
    label = read_txt("./data/TBCNN_label.txt")
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
    train_len = len(train_data)
    test_len = len(test_data)
    print("Training set length: ", train_len)
    print("Test set length: ", test_len)

    train_trees = prepare_trees(train_data, transformer, left_child, right_child)
    test_trees = prepare_trees(test_data, transformer, left_child, right_child)
    model = net
    training_step = 5000
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    # loss function
    loss_func = nn.CrossEntropyLoss()  #
    train_label_tensor = torch.LongTensor(train_label)
    test_label_tensor = torch.LongTensor(test_label)
    # start training
    px = []
    py = []
    pz = []
    py1 = []
    pz1 = []
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
        test_pre = model(test_trees)
        test_loss = loss_func(test_pre, test_label_tensor)
        test_acc = (np.argmax(test_pre.data.numpy(), axis=1) == test_label).sum() / test_len
        #############################
        if (step + 1) % 10 == 0:
            print("Iteration %d:" % (step + 1))
            print("\t train_loss=%f, train_acc=%f" % (float(train_loss.data), float(train_acc)))
            print("\t test_loss=%f, test_acc=%f" % (float(test_loss.data), float(test_acc)))
        px.append(step)
        py.append(train_loss.data)
        pz.append(test_loss.data)
        py1.append(train_acc)
        pz1.append(test_acc)
    torch.save(model, "TBRT_model.pkl")
    ax1 = plt.subplot(1, 2, 1)
    p1 = ax1.plot(px, py, "r-", lw=1)
    p2 = ax1.plot(px, pz, "b-", lw=1)
    ax1.legend(["train loss", "test loss"], loc='upper right')
    ax2 = plt.subplot(1, 2, 2)
    p3 = ax2.plot(px, py1, "r-", lw=1)
    p4 = ax2.plot(px, pz1, "b-", lw=1)
    ax2.legend(["train acc", "test acc"], loc='upper right')
    plt.savefig("./images/train_TBRT.png")
