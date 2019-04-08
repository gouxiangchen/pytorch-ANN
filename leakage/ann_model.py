from torch import nn


class ANN(nn.Module):
    def __init__(self, in_dimensions=1024, out_class=4):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(in_dimensions, 128)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, 64)
        # self.fc4 = nn.Linear(64, out_class)
        # self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(128, out_class)

        self.ReLu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        nn.init.kaiming_normal_(self.fc1.weight.data)
        # nn.init.kaiming_normal_(self.fc2.weight.data)
        # nn.init.kaiming_normal_(self.fc3.weight.data)
        nn.init.kaiming_normal_(self.fc4.weight.data)

    def forward(self, x):
        x = self.ReLu(self.fc1(x))
        # x = self.ReLu(self.fc2(x))
        # x = self.ReLu(self.fc3(x))
        x = self.fc4(x)
        x = self.ReLu(x)
        x = self.softmax(x)
        # print(x)
        return x

