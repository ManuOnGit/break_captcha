import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.autograd import Variable

from data_loader import Dataset
from preprocessing import preprocessing_full, preprocessing_basic
from tools import *

load_model = True
batch_size = 128
lr_Adam = 0.00001
tresh = Variable(torch.Tensor([0.5]))
max_epoch = 10
name = 'ConvNet_with_basic_prepro'
# name = 'ConvNet_with_full_prepro'
preprocessing_func = preprocessing_basic if 'basic' in name else preprocessing_full

class ConvNet(nn.Module):
    def __init__(self, name):
        super(ConvNet, self).__init__()
        self.name = name
        self.relu = F.relu
        self.sig = nn.Sigmoid()
        self.pool2d2 = F.max_pool2d
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.fc1 = nn.Linear(18 * 18 * 64, 500)
        self.fc2 = nn.Linear(500, 1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool2d2(x, 2, 2)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2d2(x, 2, 2)
        x = x.view(-1, 18 * 18 * 64)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sig(x)
        return x

model = ConvNet(name)

dataset_train = Dataset(transfP = preprocessing_func, train = True)
dataset_test = Dataset(transfP = preprocessing_func, train = False)

train_loader = DataLoader(dataset_train, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset_test,  batch_size = batch_size, shuffle = True)

print('==>>> total trainning batch number: {}'.format(len(train_loader)))
print('==>>> total testing batch number: {}'.format(len(test_loader)))

if load_model:
    model.load_state_dict(torch.load(MODELS + '/' + model.name))
else:
    # train model
    optimizer = optim.Adam(model.parameters(), lr = lr_Adam)
    criterion = nn.BCELoss()
    evaluator = Evaluator(dataset_train, MODELS, 'training')
    for epoch in range(max_epoch):
        evaluator.new_epoch()
        for batch_id, (images, targets, indexes) in enumerate(train_loader):
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            predictions = (predictions > tresh).float() * 1
            evaluator.update_all(targets = targets, predictions = predictions, indexes = indexes, loss = loss)
            if batch_id % 100 == 0 or batch_id == len(train_loader) - 1:
                print(evaluator)
    # save model
    torch.save(model.state_dict(), MODELS + '/' + model.name)

# test model
evaluator = Evaluator(dataset_test, MODELS, 'testing')
evaluator.new_epoch()
for batch_id, (images, targets, indexes) in enumerate(test_loader):
    predictions = model(images)
    predictions = (predictions > tresh).float() * 1
    evaluator.update_all(targets = targets, predictions = predictions, indexes = indexes)
    if batch_id % 100 == 0 or batch_id == len(test_loader) - 1:
        print(evaluator)

print(evaluator)
