"""CIFAR ResNet-18 with a unit-normalized embedding head."""

import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, stride, bias=False),
            nn.BatchNorm2d(channels),
        ) if stride != 1 or in_channels != channels else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(out)) + self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self, d=128, num_classes=100):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.in_channels = 64
        self.layers = nn.Sequential(
            self._stage(64, 2, 1), self._stage(128, 2, 2),
            self._stage(256, 2, 2), self._stage(512, 2, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, d)
        self.classifier = nn.Linear(512, num_classes)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _stage(self, channels, blocks, stride):
        layers = [BasicBlock(self.in_channels, channels, stride)]
        self.in_channels = channels
        layers.extend(BasicBlock(channels, channels) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def forward_features(self, x):
        x = self.pool(self.layers(self.stem(x))).flatten(1)
        return x

    def classify(self, x):
        """Return CIFAR class logits from the shared ResNet feature trunk."""
        return self.classifier(self.forward_features(x))

    def forward_with_logits(self, x):
        """Return the normalized retrieval embedding and auxiliary class logits."""
        features = self.forward_features(x)
        return F.normalize(self.fc(features), dim=1), self.classifier(features)

    def forward(self, x):
        return F.normalize(self.fc(self.forward_features(x)), dim=1)
