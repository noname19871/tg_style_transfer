import torch.nn as nn

from style_transfer.utils import gram_matrix


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = None

    def forward(self, x):
        gm = gram_matrix(x)
        self.loss = nn.functional.mse_loss(gm, self.target)
        return x


class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()
        self.loss = None

    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x