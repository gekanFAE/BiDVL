"""
Implementation of residual blocks for discriminator and generator.
We follow the official SNGAN Chainer implementation as closely as possible:
https://github.com/pfnet-research/sngan_projection
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GBlock(nn.Module):
    r"""
    Residual block for generator.

    Uses bilinear (rather than nearest) interpolation, and align_corners
    set to False. This is as per how torchvision does upsampling, as seen in:
    https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/_utils.py

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        hidden_channels (int): The channel size of intermediate feature maps.
        upsample (bool): If True, upsamples the input feature map.
        num_classes (int): If more than 0, uses conditional batch norm instead.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=None,
                 upsample=False,
                 num_classes=0,
                 spectral_norm=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels if hidden_channels is not None else out_channels
        self.learnable_sc = in_channels != out_channels or upsample
        self.upsample = upsample

        self.num_classes = num_classes
        self.spectral_norm = spectral_norm

        # Build the layers
        # Note: Can't use something like self.conv = SNConv2d to save code length
        # this results in somehow spectral norm working worse consistently.
        if self.spectral_norm:
            self.c1 = SNConv2d(self.in_channels,
                               self.hidden_channels,
                               3,
                               1,
                               padding=1)
            self.c2 = SNConv2d(self.hidden_channels,
                               self.out_channels,
                               3,
                               1,
                               padding=1)
        else:
            self.c1 = nn.Conv2d(self.in_channels,
                                self.hidden_channels,
                                3,
                                1,
                                padding=1)
            self.c2 = nn.Conv2d(self.hidden_channels,
                                self.out_channels,
                                3,
                                1,
                                padding=1)

        if self.num_classes == 0:
            self.b1 = nn.BatchNorm2d(self.in_channels)
            self.b2 = nn.BatchNorm2d(self.hidden_channels)
        else:
            self.b1 = ConditionalBatchNorm2d(self.in_channels,
                                             self.num_classes)
            self.b2 = ConditionalBatchNorm2d(self.hidden_channels,
                                             self.num_classes)

        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.c1.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c2.weight.data, math.sqrt(2.0))

        # Shortcut layer
        if self.learnable_sc:
            if self.spectral_norm:
                self.c_sc = SNConv2d(in_channels,
                                     out_channels,
                                     1,
                                     1,
                                     padding=0)
            else:
                self.c_sc = nn.Conv2d(in_channels,
                                      out_channels,
                                      1,
                                      1,
                                      padding=0)

            nn.init.xavier_uniform_(self.c_sc.weight.data, 1.0)

    def _upsample_conv(self, x, conv):
        r"""
        Helper function for performing convolution after upsampling.
        """
        return conv(
            F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=False))

    def _residual(self, x):
        r"""
        Helper function for feedforwarding through main layers.
        """
        h = x
        h = self.b1(h)
        h = self.activation(h)
        h = self._upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h)
        h = self.activation(h)
        h = self.c2(h)

        return h

    def _residual_conditional(self, x, y):
        r"""
        Helper function for feedforwarding through main layers, including conditional BN.
        """
        h = x
        h = self.b1(h, y)
        h = self.activation(h)
        h = self._upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h, y)
        h = self.activation(h)
        h = self.c2(h)

        return h

    def _shortcut(self, x):
        r"""
        Helper function for feedforwarding through shortcut layers.
        """
        if self.learnable_sc:
            x = self._upsample_conv(
                x, self.c_sc) if self.upsample else self.c_sc(x)
            return x
        else:
            return x

    def forward(self, x, y=None):
        r"""
        Residual block feedforward function.
        """
        if y is None:
            return self._residual(x) + self._shortcut(x)

        else:
            return self._residual_conditional(x, y) + self._shortcut(x)


class DBlock(nn.Module):
    """
    Residual block for discriminator.

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        hidden_channels (int): The channel size of intermediate feature maps.
        downsample (bool): If True, downsamples the input feature map.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.        
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=None,
                 downsample=False,
                 spectral_norm=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels if hidden_channels is not None else in_channels
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        self.spectral_norm = spectral_norm

        # Build the layers
        if self.spectral_norm:
            self.c1 = SNConv2d(self.in_channels, self.hidden_channels, 3, 1, 1)
            self.c2 = SNConv2d(self.hidden_channels, self.out_channels, 3, 1,
                               1)
        else:
            self.c1 = nn.Conv2d(self.in_channels, self.hidden_channels, 3, 1,
                                1)
            self.c2 = nn.Conv2d(self.hidden_channels, self.out_channels, 3, 1,
                                1)

        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.c1.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c2.weight.data, math.sqrt(2.0))

        # Shortcut layer
        if self.learnable_sc:
            if self.spectral_norm:
                self.c_sc = SNConv2d(in_channels, out_channels, 1, 1, 0)
            else:
                self.c_sc = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

            nn.init.xavier_uniform_(self.c_sc.weight.data, 1.0)

    def _residual(self, x):
        """
        Helper function for feedforwarding through main layers.
        """
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)

        return h

    def _shortcut(self, x):
        """
        Helper function for feedforwarding through shortcut layers.
        """
        if self.learnable_sc:
            x = self.c_sc(x)
            return F.avg_pool2d(x, 2) if self.downsample else x

        else:
            return x

    def forward(self, x):
        """
        Residual block feedforward function.
        """
        return self._residual(x) + self._shortcut(x)


class DBlockOptimized(nn.Module):
    """
    Optimized residual block for discriminator. This is used as the first residual block,
    where there is a definite downsampling involved. Follows the official SNGAN reference implementation
    in chainer.

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.        
    """
    def __init__(self, in_channels, out_channels, spectral_norm=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spectral_norm = spectral_norm

        # Build the layers
        if self.spectral_norm:
            self.c1 = SNConv2d(self.in_channels, self.out_channels, 3, 1, 1)
            self.c2 = SNConv2d(self.out_channels, self.out_channels, 3, 1, 1)
            self.c_sc = SNConv2d(self.in_channels, self.out_channels, 1, 1, 0)
        else:
            self.c1 = nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1)
            self.c2 = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
            self.c_sc = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)

        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.c1.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c2.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c_sc.weight.data, 1.0)

    def _residual(self, x):
        """
        Helper function for feedforwarding through main layers.
        """
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = F.avg_pool2d(h, 2)

        return h

    def _shortcut(self, x):
        """
        Helper function for feedforwarding through shortcut layers.
        """
        return self.c_sc(F.avg_pool2d(x, 2))

    def forward(self, x):
        """
        Residual block feedforward function.
        """
        return self._residual(x) + self._shortcut(x)


class SelfAttention(nn.Module):
    """
    Self-attention layer based on version used in BigGAN code:
    https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py
    """
    def __init__(self, num_feat, spectral_norm=True):
        super().__init__()
        self.num_feat = num_feat
        self.spectral_norm = spectral_norm

        if self.spectral_norm:
            self.theta = SNConv2d(self.num_feat,
                                  self.num_feat >> 3,
                                  1,
                                  1,
                                  padding=0,
                                  bias=False)
            self.phi = SNConv2d(self.num_feat,
                                self.num_feat >> 3,
                                1,
                                1,
                                padding=0,
                                bias=False)
            self.g = SNConv2d(self.num_feat,
                              self.num_feat >> 1,
                              1,
                              1,
                              padding=0,
                              bias=False)
            self.o = SNConv2d(self.num_feat >> 1,
                              self.num_feat,
                              1,
                              1,
                              padding=0,
                              bias=False)

        else:
            self.theta = nn.Conv2d(self.num_feat,
                                   self.num_feat >> 3,
                                   1,
                                   1,
                                   padding=0,
                                   bias=False)
            self.phi = nn.Conv2d(self.num_feat,
                                 self.num_feat >> 3,
                                 1,
                                 1,
                                 padding=0,
                                 bias=False)
            self.g = nn.Conv2d(self.num_feat,
                               self.num_feat >> 1,
                               1,
                               1,
                               padding=0,
                               bias=False)
            self.o = nn.Conv2d(self.num_feat >> 1,
                               self.num_feat,
                               1,
                               1,
                               padding=0,
                               bias=False)

        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x):
        """
        Feedforward function. Implementation differs from actual SAGAN paper,
        see note from BigGAN:
        https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py#L142

        See official TF Implementation:
        https://github.com/brain-research/self-attention-gan/blob/master/non_local.py

        Args:
            x (Tensor): Input feature map.

        Returns:
            Tensor: Feature map weighed with attention map.
        """
        N, C, H, W = x.shape
        location_num = H * W
        downsampled_num = location_num >> 2

        # Theta path
        theta = self.theta(x)
        theta = theta.view(N, C >> 3, location_num)  # (N, C>>3, H*W)

        # Phi path
        phi = self.phi(x)
        phi = F.max_pool2d(phi, [2, 2], stride=2)
        phi = phi.view(N, C >> 3, downsampled_num)  # (N, C>>3, H*W>>2)

        # Attention map
        attn = torch.bmm(theta.transpose(1, 2), phi)
        attn = F.softmax(attn, -1)  # (N, H*W, H*W>>2)
        # print(torch.sum(attn, axis=2)) # (N, H*W)

        # Conv value
        g = self.g(x)
        g = F.max_pool2d(g, [2, 2], stride=2)
        g = g.view(N, C >> 1, downsampled_num)  # (N, C>>1, H*W>>2)

        # Apply attention
        attn_g = torch.bmm(g, attn.transpose(1, 2))  # (N, C>>1, H*W)
        attn_g = attn_g.view(N, C >> 1, H, W)  # (N, C>>1, H, W)

        # Project back feature size
        attn_g = self.o(attn_g)

        # Weigh attention map
        output = x + self.gamma * attn_g

        return output


def SNConv2d(*args, default=True, **kwargs):
    r"""
    Wrapper for applying spectral norm on conv2d layer.
    """
    if default:
        return nn.utils.spectral_norm(nn.Conv2d(*args, **kwargs))

    else:
        return snconv2d(*args, **kwargs)


def SNLinear(*args, default=True, **kwargs):
    r"""
    Wrapper for applying spectral norm on linear layer.
    """
    if default:
        return nn.utils.spectral_norm(nn.Linear(*args, **kwargs))

    else:
        return snlinear(*args, **kwargs)


def SNEmbedding(*args, default=True, **kwargs):
    r"""
    Wrapper for applying spectral norm on embedding layer.
    """
    if default:
        return nn.utils.spectral_norm(nn.Embedding(*args, **kwargs))

    else:
        return snembedding(*args, **kwargs)


class ConditionalBatchNorm2d(nn.Module):
    r"""
    Conditional Batch Norm as implemented in
    https://github.com/pytorch/pytorch/issues/8985

    Attributes:
        num_features (int): Size of feature map for batch norm.
        num_classes (int): Determines size of embedding layer to condition BN.
    """
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(
            1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:,
                               num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        r"""
        Feedforwards for conditional batch norm.

        Args:
            x (Tensor): Input feature map.
            y (Tensor): Input class labels for embedding.

        Returns:
            Tensor: Output feature map.
        """
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(
            2, 1)  # divide into 2 chunks, split from dim 1.
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(
            -1, self.num_features, 1, 1)

        return out


class SpectralNorm(object):
    r"""
    Spectral Normalization for GANs (Miyato 2018).

    Inheritable class for performing spectral normalization of weights,
    as approximated using power iteration.

    Details: See Algorithm 1 of Appendix A (Miyato 2018).

    Attributes:
        n_dim (int): Number of dimensions.
        num_iters (int): Number of iterations for power iter.
        eps (float): Epsilon for zero division tolerance when normalizing.
    """
    def __init__(self, n_dim, num_iters=1, eps=1e-12):
        self.num_iters = num_iters
        self.eps = eps

        # Register a singular vector for each sigma
        self.register_buffer('sn_u', torch.randn(1, n_dim))
        self.register_buffer('sn_sigma', torch.ones(1))

    @property
    def u(self):
        return getattr(self, 'sn_u')

    @property
    def sigma(self):
        return getattr(self, 'sn_sigma')

    def _power_iteration(self, W, u, num_iters, eps=1e-12):
        with torch.no_grad():
            for _ in range(num_iters):
                v = F.normalize(torch.matmul(u, W), eps=eps)
                u = F.normalize(torch.matmul(v, W.t()), eps=eps)

        # Note: must have gradients, otherwise weights do not get updated!
        sigma = torch.mm(u, torch.mm(W, v.t()))

        return sigma, u, v

    def sn_weights(self):
        r"""
        Spectrally normalize current weights of the layer.
        """
        W = self.weight.view(self.weight.shape[0], -1)

        # Power iteration
        sigma, u, v = self._power_iteration(W=W,
                                            u=self.u,
                                            num_iters=self.num_iters,
                                            eps=self.eps)

        # Update only during training
        if self.training:
            with torch.no_grad():
                self.sigma[:] = sigma
                self.u[:] = u

        return self.weight / sigma


class snconv2d(nn.Conv2d, SpectralNorm):
    r"""
    Spectrally normalized layer for Conv2d.

    Attributes:
        in_channels (int): Input channel dimension.
        out_channels (int): Output channel dimensions.
    """
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        nn.Conv2d.__init__(self, in_channels, out_channels, *args, **kwargs)

        SpectralNorm.__init__(self,
                              n_dim=out_channels,
                              num_iters=kwargs.get('num_iters', 1))

    def forward(self, x):
        return F.conv2d(input=x,
                        weight=self.sn_weights(),
                        bias=self.bias,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups)


class snlinear(nn.Linear, SpectralNorm):
    r"""
    Spectrally normalized layer for Linear.

    Attributes:
        in_features (int): Input feature dimensions.
        out_features (int): Output feature dimensions.
    """
    def __init__(self, in_features, out_features, *args, **kwargs):
        nn.Linear.__init__(self, in_features, out_features, *args, **kwargs)

        SpectralNorm.__init__(self,
                              n_dim=out_features,
                              num_iters=kwargs.get('num_iters', 1))

    def forward(self, x):
        return F.linear(input=x, weight=self.sn_weights(), bias=self.bias)


class snembedding(nn.Embedding, SpectralNorm):
    r"""
    Spectrally normalized layer for Embedding.

    Attributes:
        num_embeddings (int): Number of embeddings.
        embedding_dim (int): Dimensions of each embedding vector
    """
    def __init__(self, num_embeddings, embedding_dim, *args, **kwargs):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, *args,
                              **kwargs)

        SpectralNorm.__init__(self, n_dim=num_embeddings)

    def forward(self, x):
        return F.embedding(input=x, weight=self.sn_weights())
