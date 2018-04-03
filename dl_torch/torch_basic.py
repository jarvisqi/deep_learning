# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.autograd import Variable


def main():
    t = torch.Tensor(5, 3)
    print(t)
    t = torch.Tensor(5, 5).uniform_(-10, 10)
    print(t)

    # x = torch.cuda.HalfTensor(5, 3).uniform_(-1, 1)
    # y = torch.cuda.HalfTensor(3, 5).uniform_(-1, 1)
    # p = torch.matmul(x, y)
    # print(p)

    # 以下转化CPU张量为GPU张量
    x = torch.FloatTensor(5, 3).uniform_(-1, 1)
    print(x)
    x = x.cuda(device=0)
    print(x)
    x = x.cpu()
    print(x)


    print('Finished')


def grad():

    x = Variable(torch.Tensor(5, 3).uniform_(-1, 1), requires_grad=True)
    y = Variable(torch.Tensor(5, 3).uniform_(-1, 1), requires_grad=True)

    z = x**2 + 3*y
    # 反向传播求导
    z.backward(gradient=torch.ones(5, 3))
    print(torch.eq(x.grad, 2*x))
    print("\n----------------------------------------\n")
    print(x.grad)
    print(y.grad)


    x1 = Variable(torch.Tensor(5, 3).uniform_(-1, 1), requires_grad=True)
    y2 = Variable(torch.Tensor(5, 3).uniform_(-1, 1), requires_grad=True)
    z = x**2 + 3*y
    
    # autograd 求导
    dz_dx = torch.autograd.grad(z, x, grad_outputs=torch.ones(5, 3))
    dz_dy = torch.autograd.grad(z, y, grad_outputs=torch.ones(5, 3))
    print(dz_dx)
    print(dz_dy)


def net():
    # number of neurons in each layer
    input_num_units = 28*28
    hidden_num_units = 500
    output_num_units = 10

    # set remaining variables
    epochs = 5
    batch_size = 128
    learning_rate = 0.001
    
    model = torch.nn.Sequential(
        torch.nn.Linear(input_num_units, hidden_num_units),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_num_units, output_num_units)
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



if __name__ == '__main__':
    # main()

    grad()
