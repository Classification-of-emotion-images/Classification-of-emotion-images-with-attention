import torch.nn as nn
from torchvision import models


class pretrained_resnet152(nn.Module):

    def __init__(self, freeze=True, n_classes=6):
        super(pretrained_resnet152, self).__init__()
        
        self.pretrained = models.resnet152(pretrained=True)

        if freeze:
            for param in self.pretrained.parameters():
                param.requires_grad = False

        n_inputs = self.pretrained.fc.out_features

        self.l1 = nn.Linear(n_inputs, 1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.l2 = nn.Linear(1024, n_classes)
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input):
        x = self.pretrained(input)
        x = x.view(x.size(0),-1)
        x = self.l1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.l2(x)
        x = self.LogSoftmax(x)
        
        return x
