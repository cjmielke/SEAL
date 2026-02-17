
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from collections import OrderedDict

class EncoderResNet50(nn.Module): 
    def __init__(self):
        super(EncoderResNet50, self).__init__()

        resnet = resnet50(weights=ResNet50_Weights.DEFAULT) 

        # Remove the head from ResNet50 network to obtain the feature encoder (feature embedding dim = 1024)
        self.model = nn.Sequential(*list(resnet.children())[:-3]) # [:-2]
        

    def forward(self, x):
        x = self.model(x)
        x = x.mean([2, 3])  # Global Average Pooling (GAP)
        
        return x

        
    