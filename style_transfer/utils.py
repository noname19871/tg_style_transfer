import torch


def gram_matrix(x):
    """
    Computes Gram Matrix of Tensor x

    :param x: Tensor of size (A x B x C x D)
    :return: Tensor, representing Gram Matrix of given Tensor x
    """

    # a = batch size(=1)
    # b = number of feature maps
    # (c, d) = dimensions of a feature map (N = c * d)
    a, b, c, d = x.size()

    # resize F_XL into \hat F_XL
    features = x.view(a * b, c * d)

    # compute the gram product
    gm = torch.mm(features, features.t())

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return gm.div(a * b * c * d)
