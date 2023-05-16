import torch
import torch.nn as nn
import torch.nn.functional as F

# class ContrastiveLoss(nn.Module):
#     def __init__(self, margin=1.0):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin

#     def forward(self, x0, x1, y):
#         # euclidian distance
#         diff = x0 - x1
#         dist_sq = torch.sum(torch.pow(diff, 2), 1)
#         dist = torch.sqrt(dist_sq)

#         mdist = self.margin - dist
#         dist = torch.clamp(mdist, min=0.0)
#         loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
#         loss = torch.sum(loss) / 2.0 / x0.size()[0]
#         return loss
    


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive