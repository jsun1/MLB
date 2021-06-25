import datetime
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# Put this model in the Notebook!
class LinearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize, hidden_size):
        super(LinearRegression, self).__init__()
        self.batch = nn.LayerNorm(inputSize)
        self.linear = nn.Linear(inputSize, hidden_size)
        self.relu = nn.ReLU()
        self.batch1 = nn.LayerNorm(hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.batch2 = nn.LayerNorm(hidden_size)
        # self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, outputSize)

    def forward(self, x):
        out = self.batch(x)
        out = self.linear(out)
        out = self.relu(out)
        out = self.batch1(out)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.batch2(out)
        # out = self.linear2(out)
        out = self.hidden(out)
        return out


class CustomLoss(torch.nn.Module):

    def __init__(self):
        super(CustomLoss,self).__init__()

    def forward(self, outputs, labels):
        metric = torch.mean(torch.abs(outputs - labels), 0)
        # print(metric.detach().numpy())
        metric = torch.mean(metric)
        return torch.square(metric)


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


# Put this in the Notebook!
def model_inputs(merged):
    inputs = merged.drop(['date', 'engagementMetricsDate', 'playerId', 'jerseyNum', 'target1', 'target2', 'target3', 'target4'], 1).to_numpy(dtype=np.float32)
    return inputs


# Put this in the Notebook!
def make_model():
    model = LinearRegression(76, 4, 256)
    return model


def main():
    # get the training data from features
    merged = pd.read_pickle('mlb-merged-data/merged.pkl')
    x_train = model_inputs(merged)
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

    learningRate = 0.00003
    epochs = 50# 300

    model = make_model()

    criterion = CustomLoss()  # torch.nn.MSELoss()  # TODO: figure out what this loss means (26.?)
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    print('TYPE', type(model), type(x_train))

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

        # Get the evaluation metric
        if epoch % 5 == 4:
            metric = np.mean(np.mean(np.absolute(outputs.detach().numpy() - labels.detach().numpy()), axis=0))
            print('metric', metric)

        # print(loss)
        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()

    # print(loss.item())
    # print('epoch {}, loss {}'.format(epoch, np.sqrt(loss.item())))
    # TODO: internally validate the model

    # save the model
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    link = 'saved-models/' + date + '.pt'
    torch.save(model.state_dict(), link)
    print('Saved at', link)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
