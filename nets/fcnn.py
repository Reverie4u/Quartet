import torch
import torch.nn as nn


class FCNN(torch.nn.Module):

    def __init__(self, n_feature):
        super(FCNN, self).__init__()
        self.hidden1 = nn.Linear(n_feature, 150)  # hidden layer
        self.hidden2 = nn.Linear(150, 150)
        self.out = nn.Linear(150, 4)
        self.dr1 = nn.Dropout(p=0.5)
        self.dr2 = nn.Dropout(p=0.5)

    def forward(self, inputs):
        x = self.hidden1(inputs)
        x = torch.relu(x)  # activation function
        x = self.dr1(x)
        x = self.hidden2(x)
        x = torch.relu(x)  # activation function
        x = self.dr2(x)
        x = self.out(x)
        return x
