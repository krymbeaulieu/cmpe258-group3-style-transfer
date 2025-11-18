import torch
import torch.nn.functional as F
import torch.nn as nn


VGG_FEATURES = []

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input
        
class StyleLossBatch(nn.Module):
    def __init__(self, target_feature):
        super(StyleLossBatch, self).__init__()
        with torch.no_grad():
           self.target = gram_matrix(target_feature[0].unsqueeze(0)).detach()  # Use 1 style image in [1, C, H, W]

    def forward(self, input):
        # Compute Gram for each image in the batch and sum losses
        self.loss = 0
        for i in range(input.size(0)):
            G = gram_matrix(input[i].unsqueeze(0))
            self.loss += F.mse_loss(G, self.target)
        self.loss /= input.size(0)  # average over batch
        return input

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input



def mean_std(feat, eps=1e-5):
    # Calculate per-channel mean and std
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    assert content_feat.size()[:2] == style_feat.size()[:2]
    style_mean, style_std = mean_std(style_feat)
    content_mean, content_std = mean_std(content_feat)
    normalized_feat = (content_feat - content_mean) / content_std
    return normalized_feat * style_std + style_mean