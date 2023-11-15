import torch
import torch.nn as nn
import torch.nn.functional as F

# Focal loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # LogSoftmax on the inputs
        log_probas = F.log_softmax(inputs, dim=1)

        # Select the log proba associated with the true class for each sample
        targets_one_hot = F.one_hot(targets, num_classes=log_probas.shape[1]).float()
        selected_log_probas = torch.sum(log_probas * targets_one_hot, dim=1)

        # Compute the focal loss
        loss = -self.alpha * (1 - selected_log_probas.exp()) ** self.gamma * selected_log_probas

        # Reduce the loss based on reduction parameter
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss
