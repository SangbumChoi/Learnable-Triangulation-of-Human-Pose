import torch

class MSEloss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.threshold = 20

    def forward(self, y_j, y_j_gt, alpha):
        dimension = y_j.shape[-1]
        y_j_gt = y_j_gt.shape[-1]
        # MSEloss
        diff = (y_j - y_j_gt) ** 2 * alpha
        diff[diff >= self.threshold] = torch.pow(diff[diff > self.threshold], 0.1) * (self.threshold ** 0.9)
        loss = torch.sum(diff) / (dimension * max(1, torch.sum(alpha).item()))
        return loss

class L1loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, y_j, y_j_gy,beta):
        pass

