"""MCIA's DSRANet adaptation for 12-electrode spectrograms.

Architecture reference: hdy6438/DSRANet@c0b04378055d542a39a56114c48670a8d9916ffa.
"""
from torch import nn


def separable(inputs, outputs, kernel, padding=0):
    return nn.Sequential(nn.Conv2d(inputs, inputs, kernel, padding=padding, groups=inputs, bias=False),
                         nn.Conv2d(inputs, outputs, 1, bias=False))


def conv_norm(inputs, outputs, kernel, padding=0):
    return nn.Sequential(separable(inputs, outputs, kernel, padding), nn.BatchNorm2d(outputs))


class ResidualSpectralStage(nn.Module):
    def __init__(self, inputs, outputs, dropout):
        super().__init__()
        self.residual = nn.Sequential(conv_norm(inputs, outputs, 3, 1), nn.SiLU(),
                                      nn.Dropout2d(dropout), conv_norm(outputs, outputs, 3, 1))
        self.shortcut = conv_norm(inputs, outputs, 1)
        self.activation = nn.SiLU()
        self.pool = nn.MaxPool2d((2, 1))
        self.frequency = separable(outputs, outputs, (7, 1), (3, 0))
        self.time = separable(outputs, outputs, (1, 7), (0, 3))
        self.channel = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                     nn.Linear(outputs, outputs // 4), nn.SiLU(),
                                     nn.Linear(outputs // 4, outputs), nn.Sigmoid())

    def forward(self, x):
        x = self.pool(self.activation(self.residual(x) + self.shortcut(x)))
        return x * self.frequency(x).sigmoid() * self.time(x).sigmoid() * self.channel(x)[:, :, None, None]


class DSRANet(nn.Sequential):
    """Full depthwise residual and frequency/time/channel attention backbone."""

    def __init__(self):
        super().__init__(conv_norm(12, 32, 3, 1), nn.SiLU(),
                         ResidualSpectralStage(32, 64, 0.5),
                         ResidualSpectralStage(64, 128, 0.4),
                         ResidualSpectralStage(128, 256, 0.3),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.4), nn.Linear(256, 49))
