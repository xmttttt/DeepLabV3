import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss_func(nn.Module):
    def __init__(self):
        super(Loss_func, self).__init__()

        self.loss = nn.BCEWithLogitsLoss()


    def forward(self, pred, gt):

        # print(pred.shape,gt.shape)

        Loss = self.loss(pred[:,0,:,:],gt)

        return Loss
