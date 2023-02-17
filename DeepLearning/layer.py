import torch.nn as nn


def conv3x3(in_channels, out_channels, stride):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_channels, out_channels, stride):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()

        self.layer = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(),
            conv3x3(in_channels=in_channels, out_channels=out_channels, stride=stride),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            conv3x3(in_channels=in_channels, out_channels=out_channels * BasicBlock.expansion, stride=1)
        )

        non_identity_condition1 = stride != 1
        non_identity_condition2 = in_channels != out_channels * BasicBlock.expansion

        if non_identity_condition1 or non_identity_condition2:
            self.shortcut = conv1x1(in_channels=in_channels, out_channels=out_channels * BasicBlock.expansion, stride=stride)
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        identity = self.shortcut(x)
        out = identity + self.layer(x)

        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride):
        super(BottleNeck, self).__init__()

        self.layer = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(),
            conv1x1(in_channels=in_channels, out_channels=out_channels, stride=stride),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            conv3x3(in_channels=out_channels, out_channels=out_channels, stride=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            conv1x1(in_channels=out_channels, out_channels=out_channels * BottleNeck.expansion, stride=1),
        )

        non_identity_condition1 = stride != 1
        non_identity_condition2 = in_channels != out_channels * BottleNeck.expansion

        if non_identity_condition1 or non_identity_condition2:
            self.shortcut = conv1x1(in_channels=in_channels, out_channels=out_channels * BottleNeck.expansion, stride=stride)
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        identity = self.shortcut(x)
        out = identity + self.layer(x)

        return out
