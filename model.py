import os
from collections import Counter
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np

from data_loader import Dataset
from preprocessing import preprocessing_full

error_track_test = Counter()
error_track_train = Counter()
dataset = Dataset(preprocessing_full)
shuffle_dataset = True
random_seed = 42
batch_size = 128
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.2 * dataset_size))

if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

train_indices, test_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(dataset, batch_size = batch_size, sampler = train_sampler)
test_loader = DataLoader(dataset, batch_size = batch_size, sampler = test_sampler)

print('==>>> total trainning batch number: {}'.format(len(train_loader)))
print('==>>> total testing batch number: {}'.format(len(test_loader)))

class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 5, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 5, 1)
        self.fc1 = torch.nn.Linear(18 * 18 * 64, 500)
        self.fc2 = torch.nn.Linear(500, 1)
        self.out_act = torch.nn.Sigmoid()
    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = x.view(-1, 18 * 18 * 64)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = self.out_act(x)
        return x
    def name(self):
        return "ConvNet"

model = ConvNet()

optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

criterion = torch.nn.BCELoss()

tresh = Variable(torch.Tensor([0.5]))

for epoch in range(10):
    # trainning
    ave_loss = 0
    for batch_idx, (x, target, indexes) in enumerate(train_loader):
        optimizer.zero_grad()
        x, target = Variable(x), Variable(target)
        out = model(x)
        loss = criterion(out, target)
        ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1
        loss.backward()
        optimizer.step()
        out = (out > tresh).float() * 1
        if (out == target).sum() != len(target):
            error_track_train += Counter(list(indexes[((out == target) == 0).nonzero()[:,0]].flatten().numpy()))
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
            print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}, acc: {:.3f}'.format(
                epoch, batch_idx + 1, ave_loss, float((out == target).sum()) / len(target)))

# testing
correct_cnt, ave_loss = 0, 0
total_cnt = 0
for batch_idx, (x, target, indexes) in enumerate(test_loader):
    x, target = Variable(x), Variable(target)
    out = model(x)
    loss = criterion(out, target)
    total_cnt += x.data.size()[0]
    out = (out > tresh).float() * 1
    correct_cnt += (out == target).sum()
    ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1
    if (out == target).sum() != len(target):
        error_track_test += Counter(list(indexes[((out == target) == 0).nonzero()[:,0]].flatten().numpy()))
    if(batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(test_loader):
        print('==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(
            epoch, batch_idx + 1, ave_loss, float(correct_cnt) * 1.0 / total_cnt))

torch.save(model.state_dict(), model.name())
