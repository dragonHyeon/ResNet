import torch.nn as nn

from DeepLearning.layer import BasicBlock, BottleNeck


class ResNet(nn.Module):
    def __init__(self, block_type, num_blocks_list, in_channels=3, num_classes=100, init_weights=True):
        """
        * 모델 구조 정의
        :param block_type: BasicBlock / BottleNeck 선택
        :param num_blocks_list: 스테이지 당 블록 몇 개씩 쌓을지
        :param num_classes: 출력 클래스 개수
        :param init_weights: 가중치 초기화 여부
        """

        super(ResNet, self).__init__()

        #
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        #
        self.stage1 = self._make_layer(block_type=block_type,
                                       num_blocks=num_blocks_list[0],
                                       in_channels=64,
                                       out_channels=64,
                                       stride=1)

        #
        self.stage2 = self._make_layer(block_type=block_type,
                                       num_blocks=num_blocks_list[1],
                                       in_channels=64 * block_type.expansion,
                                       out_channels=128,
                                       stride=2)

        #
        self.stage3 = self._make_layer(block_type=block_type,
                                       num_blocks=num_blocks_list[2],
                                       in_channels=128 * block_type.expansion,
                                       out_channels=256,
                                       stride=2)

        #
        self.stage4 = self._make_layer(block_type=block_type,
                                       num_blocks=num_blocks_list[3],
                                       in_channels=256 * block_type.expansion,
                                       out_channels=512,
                                       stride=2)

        #
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        #
        self.fc = nn.Linear(in_features=512 * block_type.expansion,
                            out_features=num_classes,
                            bias=True)

        # 가중치 초기화
        if init_weights:
            self._initialize_weights()

    @staticmethod
    def _make_layer(block_type, num_blocks, in_channels, out_channels, stride):
        """

        :param block_type:
        :param num_blocks:
        :param in_channels:
        :param out_channels:
        :param stride:
        :return:
        """

        layers = list()

        layers.append(block_type(in_channels=in_channels, out_channels=out_channels, stride=stride))

        in_channels = out_channels * block_type.expansion

        for _ in range(num_blocks - 1):
            layers.append(block_type(in_channels=in_channels, out_channels=out_channels, stride=stride))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        * 순전파
        :param x: 배치 개수 만큼의 입력. (N, 3, 224, 224)
        :return: 배치 개수 만큼의 출력. (N, 100)
        """

        #
        out = self.conv(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.avgpool(out)
        out = self.fc(out)

        return out

    def _initialize_weights(self):
        """
        * 모델 가중치 초기화
        :return: 모델 가중치 초기화 진행됨
        """

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(tensor=m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(tensor=m.bias, val=0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(tensor=m.weight, val=1)
                nn.init.constant_(tensor=m.bias, val=0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(tensor=m.weight, mean=0, std=0.01)
                nn.init.constant_(tensor=m.bias, val=0)


def resnet18():
    """
    * ResNet-18
    :return: ResNet 18-layer 모델
    """

    return ResNet(block_type=BasicBlock, num_blocks_list=[2, 2, 2, 2])


def resnet50():
    """
    * ResNet-50
    :return: ResNet 50-layer 모델
    """

    return ResNet(block_type=BottleNeck, num_blocks_list=[3, 4, 6, 3])
