import torch.nn as nn
import torch

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.convfc1 = nn.Linear(144, 126)



        self.pool = nn.MaxPool2d(kernel_size=3)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        board_inps = x[:, :3, :, :]
        ints_matrix = x[:, 3:, :, :]
        int_inps = ints_matrix[:, :, 0, 0]

        cnn_ouput = self.pool(F.relu(self.conv2(F.relu(self.conv1(board_inps)))))
        cnn_ouput = cnn_ouput.view(cnn_ouput.size(0), -1)
        # print(cnn_ouput.shape)
        cnn_ouput = self.convfc1(cnn_ouput)

        # print(cnn_ouput.shape)

        #turn back into batch_size x others matrix
        x = torch.cat((cnn_ouput, int_inps), dim=1)

        # print(x.shape)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x