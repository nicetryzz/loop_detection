import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
  def __init__(self, margin=1.0):
    super(ContrastiveLoss, self).__init__()
    self.margin = margin

  def forward(self, anchor, target, label):
    distance = torch.norm(anchor - target, p=2, dim=1) # 求欧几里得距离
    loss = label * distance.pow(2) + (1 - label) * torch.clamp(self.margin - distance, min=0).pow(2) # 计算对比损失
    loss = loss.mean() # 求平均值
    return loss
