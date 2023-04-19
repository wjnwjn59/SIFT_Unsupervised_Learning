import torch
import torch.nn as nn
import numpy as np
from torchvision.models import vgg16

class SiameseVGG16(nn.Module):
    def __init__(self):
        super(SiameseVGG16, self).__init__()
        self.vgg16 = vgg16(pretrained=True).features
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 1)

    def forward_once(self, x):
        x = self.vgg16(x)
        x = x.view(-1, 512 * 7 * 7)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

# Define the contrastive loss function
def contrastive_loss(output1, output2, label, margin=2.0):
    euclidean_distance = nn.functional.pairwise_distance(output1, output2)
    loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                  (label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss_contrastive
