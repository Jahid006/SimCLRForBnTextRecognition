import torch.nn as nn
import torchvision.models as models


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size,
        stride,
        padding,
        max_pooling=(0, 0),
        batch_norm=False,
        leaky_relu=False,
    ):
        super(ConvBlock, self).__init__()

        layers = []
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_ch))

        layers.append(
            nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
        )
        if max_pooling[0]:
            layers.append(nn.MaxPool2d(kernel_size=max_pooling))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class FeatureExtractor(nn.Module):
    def __init__(self, cnn_config):
        super(FeatureExtractor, self).__init__()

        self.channels = cnn_config["channels"]
        self.kernel_sizes = cnn_config["kernel_sizes"]
        self.strides = cnn_config["strides"]
        self.paddings = cnn_config["paddings"]
        self.batch_normms = cnn_config["batch_normms"]
        self.max_pooling = cnn_config["max_pooling"]
        self.leaky_relu = cnn_config["leaky_relu"]

        self.feature_extractor = nn.ModuleList(
            [
                ConvBlock(
                    self.channels[i],
                    self.channels[i + 1],
                    self.kernel_sizes[i],
                    self.strides[i],
                    self.paddings[i],
                    self.max_pooling[i],
                    self.batch_normms[i],
                    self.leaky_relu,
                )
                for i in range(len(self.kernel_sizes) - 1)
            ]
        )
        self.pooler = nn.AvgPool2d((2, 1))

    def forward(self, image):
        for layer in self.feature_extractor:
            image = layer(image)
        image = self.pooler(image)
        return image


class ImageEmbedding(nn.Module):
    def __init__(self, in_channel=1, out_dim=256, FeatureExtractor: nn.Module = None):
        super(ImageEmbedding, self).__init__()

        self.fe = FeatureExtractor
        self.pool1 = nn.Conv2d(512, 128, kernel_size=(1, 1), stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu_1 = nn.ReLU()
        self.conv = nn.Conv2d(128, 16, kernel_size=1, stride=1)
        self.linear = nn.Linear(1024, out_dim)
        self.relu_2 = nn.ReLU()

    def forward(self, x):
        x = self.fe(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu_1(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        x = self.relu_2(x)

        return x
