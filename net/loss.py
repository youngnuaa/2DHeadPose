import torch.nn as nn
import torch


class angle_focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(angle_focal_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma


    def forward(self, pred, gt):
        '''
            Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
          Arguments:
            pred (batch x c)
            gt_regr (batch x c)
        '''

        #pred = torch.sigmoid(pred)
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)
        pos_loss =  torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss =  torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        loss =  - (pos_loss + neg_loss)
        return loss




