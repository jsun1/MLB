import features
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LinearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize, hidden_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(inputSize, hidden_size)
        self.hidden = nn.Linear(hidden_size, outputSize)

    def forward(self, x):
        out = self.linear(x)
        out = self.hidden(out)
        return out


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    # get the training data from features
    merged = features.merge_features()
    x_train = merged.drop(['date', 'engagementMetricsDate', 'playerId', 'jerseyNum', 'target1', 'target2', 'target3', 'target4'], 1).to_numpy(dtype=np.float32)
    # x_train = merged[['caughtStealing', 'sacBunts']].to_numpy(dtype=np.float32)
    # print(len(x_values[0]))

    y_train = merged[['target1', 'target2', 'target3', 'target4']].to_numpy(dtype=np.float32)
    # y_train = y_train.reshape(-1, 4)
    print('x_train type', type(x_train), x_train.shape, y_train.shape)
    # y_values = merged['target1']

    # create dummy data for training
    # x_values = [[i for i in range(11)], [0 for i in range(11)]]
    # x_train = np.array(x_values, dtype=np.float32)
    # x_train = x_train.reshape(11, 2)

    # y_values = [2 * i + 1 for i in range(1)]
    # y_train = np.array(y_values, dtype=np.float32)
    # y_train = y_train.reshape(-1, 1)
    # print(x_train.shape, y_train.shape)

    inputDim = 76  # takes variable 'x'
    outputDim = 4  # takes variable 'y'
    learningRate = 0.00001
    epochs = 100  # 2506176

    model = LinearRegression(inputDim, outputDim, 100)

    criterion = torch.nn.MSELoss()  # TODO: figure out what this loss means (26.?)
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

    for epoch in range(epochs):
        # Converting inputs and labels to Variable
        # inputs = Variable(torch.from_numpy(x_train[epoch]))
        # labels = Variable(torch.from_numpy(y_train[epoch]))
        inputs = Variable(torch.from_numpy(x_train))
        labels = Variable(torch.from_numpy(y_train))

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        # get output from the model, given the inputs
        outputs = model(inputs)

        # get loss for the predicted output
        loss = criterion(outputs, labels)
        # print(loss)
        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))
    # TODO: internally validate the model

    # TODO: save the model and run a submission
    


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
