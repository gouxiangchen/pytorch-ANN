import torch
from leakage.ann_model import ANN
import numpy as np
import torch.nn as nn
import os
import random
import time
from leakage.logger import Logger


def get_file_name(dir):
    file_list = []
    dir_list = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            file_list.append(os.path.join(root, file))
        for dir in dirs:
            dir_list.append(dir)
    return file_list, dir_list


def get_data_from_txt(txt_path):
    f = open(txt_path)
    label = txt_path.split('/')[-2][0]
    lines = f.readlines()
    data = [float(label)]
    _data = []
    # print(label)
    for line in lines:
        _data.append(float(line.strip(' ').split(',')[1]))
    # print(data)
    f.close()
    _data = np.asarray(_data)
    _data = np.fft.fft(_data)
    _data = np.abs(_data)
    # print(_data)
    _data = _data.tolist()
    # print(_data)
    data.extend(_data)
    # print(len(data), data[0])
    return np.asarray(data).reshape((1, 1025))


def get_test_data_from_txt(txt_path):
    f = open(txt_path)
    # label = txt_path.split('/')[-2][0]
    lines = f.readlines()
    # data = [float(label)]
    data = []
    # print(label)
    # for line in lines:
    for i in range(1024):
        data.append(float(lines[i].strip(' ').split(',')[1]))
    # print(len(data))
    f.close()
    _data = np.asarray(data)
    _data = np.fft.fft(_data)
    _data = np.abs(_data)
    print(_data)
    _data = _data.tolist()
    # print(_data)
    data = _data
    # print(len(data), data[0])
    data = np.asarray(data).reshape((1, 1024))
    return torch.from_numpy(data).cuda()


class Train(object):
    def __init__(self):
        super(Train, self).__init__()
        self.model = ANN().cuda()
        self.train_memory = np.zeros((1, 1025))    # 0 of 1025 is the label
        self.test_memory = np.zeros((1, 1025))
        self.train_label = None
        self.test_label = None
        self.train_memory_count = 0
        self.test_memory_count = 0

        # self.train_memory_count = 60000   # for MNIST
        # self.test_memory_count = 10000

        self.batch_size = 64
        self.learn_rate = 0.001
        self.loss_func = nn.CrossEntropyLoss()
        self.logger = Logger('./logs')
        self.steps = 0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learn_rate)

    def save_model(self, path):
        torch.save(self.model.state_dict(), './models/' + path)
        print('model is saved at ' + path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        print(path + 'has been loaded')

    def load_MNIST(self, path):
        # t10k-images.idx3-ubyte
        # t10k-labels.idx1-ubyte
        # train-images.idx3-ubyte
        # train-labels.idx1-ubyte
        print('begin loading MNIST')
        f = open(path + 'train-images.idx3-ubyte')
        p = np.fromfile(f, dtype=np.uint8)
        print(p[:16])
        self.train_memory = p[16:].reshape((60000, 784)).astype(np.float)
        f.close()
        f = open(path + 'train-labels.idx1-ubyte')
        p = np.fromfile(f, dtype=np.uint8)
        self.train_label = p[8:].reshape(60000).astype(np.float)
        f.close()
        f = open(path + 't10k-images.idx3-ubyte')
        p = np.fromfile(f, dtype=np.uint8)
        self.test_memory = p[16:].reshape((10000, 784)).astype(np.float)
        f.close()
        f = open(path + 't10k-labels.idx1-ubyte')
        p = np.fromfile(f, dtype=np.uint8)
        self.test_label = p[8:].reshape(10000).astype(np.float)
        print('loaded!')

    def load_dataset(self, path):
        start = time.time()
        print('begin data loading...')
        filenames, dirs = get_file_name(path)
        for filename in filenames:
            data = get_data_from_txt(filename)
            p = random.random()
            if p < 0.15:
                if self.test_memory_count == 0:
                    self.test_memory = data
                else:
                    self.test_memory = np.append(self.test_memory, data, axis=0)
                    # print(self.test_memory.shape)
                self.test_memory_count += 1

            else:
                if self.train_memory_count == 0:
                    self.train_memory = data
                else:
                    self.train_memory = np.append(self.train_memory, data, axis=0)
                    # print(self.test_memory.shape)
                self.train_memory_count += 1
        end = time.time()
        print('loading over, using time : ', end-start)
        print('dataset overview : ', self.train_memory.shape, self.test_memory.shape, self.train_memory_count, self.test_memory_count)

    def learn(self):
        batch_index = np.random.choice(self.train_memory_count, self.batch_size)
        batch_sample = self.train_memory[batch_index, :]
        batch_label = batch_sample[:, 0]
        batch_sample = batch_sample[:, 1:1025]
        batch_sample = torch.FloatTensor(batch_sample).cuda()
        batch_label = torch.LongTensor(batch_label).cuda()

        out = self.model(batch_sample)
        loss = self.loss_func(out, batch_label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        info = {'loss': loss.data.item()}
        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, step=self.steps)

        batch_test_index = np.random.choice(self.test_memory_count, self.batch_size)
        batch_test_sample = self.test_memory[batch_test_index, :]
        batch_test_label = batch_test_sample[:, 0]
        batch_test_sample = batch_test_sample[:, 1:1025]
        batch_test_sample = torch.FloatTensor(batch_test_sample).cuda()
        batch_test_label = torch.LongTensor(batch_test_label)
        pre_y = self.model(batch_test_sample)
        pre_y = torch.argmax(pre_y.cpu(), 1)
        # print(pre_y)
        # print(torch.argmax(pre_y, 1))
        accuracy = float((pre_y == batch_test_label).sum().data) / float(self.batch_size)
        info = {'accuracy': accuracy}
        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, step=self.steps)

        self.steps += 1

    def learn_MNIST(self):
        batch_index = np.random.choice(self.train_memory_count, self.batch_size)
        batch_sample = self.train_memory[batch_index, :]
        batch_label = self.train_label[batch_index]
        # batch_sample = batch_sample[:, 1:1025]
        batch_sample = torch.FloatTensor(batch_sample).cuda()
        batch_label = torch.LongTensor(batch_label).cuda()

        out = self.model(batch_sample)
        loss = self.loss_func(out, batch_label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # print(out, batch_label)

        info = {'loss': loss.data.item()}
        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, step=self.steps)

        batch_test_index = np.random.choice(self.test_memory_count, self.batch_size)
        batch_test_sample = self.test_memory[batch_test_index, :]
        batch_test_label = self.test_label[batch_test_index]
        # batch_test_sample = batch_test_sample[:, 1:1025]
        batch_test_sample = torch.FloatTensor(batch_test_sample).cuda()
        batch_test_label = torch.LongTensor(batch_test_label)
        pre_y = self.model(batch_test_sample)
        pre_y = torch.argmax(pre_y.cpu(), 1)
        # print(pre_y)
        # print(torch.argmax(pre_y, 1))
        # print(pre_y, batch_test_label)
        # print((pre_y == batch_test_label).sum())
        accuracy = float((pre_y == batch_test_label).sum().data) / float(self.batch_size)
        info = {'accuracy': accuracy}
        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, step=self.steps)

        self.steps += 1
