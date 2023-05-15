import torch
import torch.nn as nn
import torchvision


def initialize_weights(
    model, 
    initialization_method
):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            initialization_method(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

class SiameseVGG16(nn.Module):
    def __init__(self):
        super(SiameseVGG16, self).__init__()
        vgg16 =  torchvision.models.vgg16(weights=None)
        initialize_weights(vgg16, nn.init.kaiming_normal_)
        self.encoder = nn.Sequential(*list(vgg16.features.children()))
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        # self.fc2 = nn.Linear(4096, 1)
        self.fc2 = nn.Linear(4096, 2)


    def forward_once(self, x):
        x = self.encoder(x)
        # x = x.view(-1, 512 * 7 * 7)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)

        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2

def contrastive_loss(output1, output2, euclid_dist, label, margin):
    loss_contrastive = torch.mean((1-label) * torch.pow(euclid_dist, 2) +
                                  (label) * torch.pow(torch.clamp(margin - euclid_dist, min=0.0), 2))
    return loss_contrastive
