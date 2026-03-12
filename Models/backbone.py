import torch
import torch.nn as nn
import torchvision.models as models

class RFBackbone(nn.Module):

    def __init__(self):

        super().__init__()

        # ResNet18 backbone
        resnet = models.resnet18(pretrained=False)

        # modify first layer for 1-channel spectrogram
        resnet.conv1 = nn.Conv2d(
            1,64,kernel_size=7,stride=2,padding=3,bias=False
        )

        # remove classifier
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.pool = nn.AdaptiveAvgPool2d((1,32))

    def forward(self,x):

        x = self.cnn(x)

        x = self.pool(x)

        x = x.squeeze(2)

        x = x.permute(0,2,1)

        lstm_out,_ = self.lstm(x)

        features = lstm_out[:,-1,:]

        return features