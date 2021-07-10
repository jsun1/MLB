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
        drop_p = 0.5
        # self.batch = nn.LayerNorm(inputSize)
        # self.dropout = nn.Dropout(p=drop_p)
        self.linear = nn.Linear(inputSize, hidden_size)
        # self.tanh = nn.Tanh()
        # self.relu = nn.ReLU()
        # self.batch1 = nn.LayerNorm(hidden_size)
        # self.dropout1 = nn.Dropout(p=drop_p)
        # self.linear1 = nn.Linear(hidden_size, hidden_size)
        # self.batch2 = nn.LayerNorm(hidden_size)
        # self.dropout2 = nn.Dropout(p=drop_p)
        # self.linear2 = nn.Linear(hidden_size, hidden_size)
        # self.batch3 = nn.LayerNorm(hidden_size)
        # self.dropout3 = nn.Dropout(p=drop_p)
        self.hidden = nn.Linear(hidden_size, outputSize)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        # out = self.batch(x)
        # out = self.dropout(x)
        out = self.linear(x)  # changed out to x
        out = self.leaky_relu(out)
        # out = self.tanh(out)
        # out = self.batch1(out)
        # out = self.dropout1(out)

        # out = self.linear1(out)
        # out = self.relu(out)
        # out = self.batch2(out)
        # out = self.dropout2(out)

        # out = self.linear2(out)
        # out = self.relu(out)
        # out = self.batch3(out)
        # out = self.dropout3(out)

        out = self.hidden(out)
        # out = self.extra(out)
        # out = self.relu(out)
        # out = torch.sqrt(out)
        out = self.leaky_relu(out)
        return out


class CustomLoss(torch.nn.Module):

    def __init__(self):
        super(CustomLoss,self).__init__()

    def forward(self, outputs, labels):
        metric = torch.mean(torch.abs(outputs - labels), 0)
        # print(metric.detach().numpy())
        metric = torch.mean(metric)
        # return torch.square(metric)
        return metric


# Put this in the Notebook!
def model_inputs(merged):
    # inputs = merged.drop(['date', 'engagementMetricsDate', 'playerId', 'jerseyNum', 'target1', 'target2', 'target3', 'target4', 'year', 'dayOfWeek', 'day', 'week'], 1).to_numpy(dtype=np.float32)
    inputs = merged.drop(['date', 'engagementMetricsDate', 'playerId', 'jerseyNum', 'target1', 'target2', 'target3', 'target4'], 1).to_numpy(dtype=np.float32)
    return inputs


# Put this in the Notebook!
def make_model():
    model = LinearRegression(105, 4, 16)
    return model


def main():
    # get the training data from features
    merged = pd.read_pickle('mlb-merged-data/merged.pkl')
    split_date = pd.to_datetime('2021-04-01')
    training = True
    if training:
        merged_train = merged.loc[merged.date < split_date]
    else:
        merged_train = merged  # train on all data
    merged_val = merged.loc[merged.date >= split_date]
    x_train = model_inputs(merged_train)
    x_val = model_inputs(merged_val)

    # x_train = merged[['caughtStealing', 'sacBunts']].to_numpy(dtype=np.float32)
    # print(len(x_values[0]))

    y_train = merged_train[['target1', 'target2', 'target3', 'target4']].to_numpy(dtype=np.float32)
    y_val = merged_val[['target1', 'target2', 'target3', 'target4']].to_numpy(dtype=np.float32)
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

    learningRate = 0.003
    epochs = 600

    model = make_model()
    model.train()

    criterion = CustomLoss()  # torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
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
        if epoch % 20 == 19:
            model.eval()
            metric = np.mean(np.mean(np.absolute(np.clip(outputs.detach().numpy(), 0, 100) - labels.detach().numpy()), axis=0))
            # Validation
            inputs_val = Variable(torch.from_numpy(x_val))
            labels_val = Variable(torch.from_numpy(y_val))
            outputs_val = model(inputs_val)
            metric_val = np.mean(np.mean(np.absolute(np.clip(outputs_val.detach().numpy(), 0, 100) - labels_val.detach().numpy()), axis=0))
            print('metric', epoch + 1, loss.item(), metric, metric_val)
            model.train()

        # print(loss)
        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()

    # print(loss.item())
    # print('epoch {}, loss {}'.format(epoch, np.sqrt(loss.item())))

    # save the model
    if not training:
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        link = 'saved-models/' + date + '.pt'
        torch.save(model.state_dict(), link)
        print('Saved at', link)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
