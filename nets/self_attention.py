#!/usr/bin/python
# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention_Layer(nn.Module):

    def __init__(self, input_dim):
        super(Attention_Layer, self).__init__()
        self.Q_linear = nn.Linear(input_dim, input_dim, bias=False)
        self.K_linear = nn.Linear(input_dim, input_dim, bias=False)
        self.V_linear = nn.Linear(input_dim, input_dim, bias=False)

    def forward(self, inputs):
        # 计算生成QKV矩阵
        Q = self.Q_linear(inputs)
        K = self.K_linear(inputs)
        V = self.V_linear(inputs)
        alpha = torch.matmul(Q, K.T)
        alpha = F.softmax(alpha, dim=-1)
        out = torch.matmul(alpha, V)
        return out


if __name__ == "__main__":
    x = [
        [1, 0, 1, 0],  # input 1
        [0, 2, 0, 2],  # input 2
        [1, 1, 1, 1],  # input 3
    ]
    x = torch.tensor(x, dtype=torch.float32)
    # print(b)
    net = Attention_Layer(4)
    print(net(x))
