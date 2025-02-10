import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np
from torch.nn import functional as F


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


track_running_stats = True
momentum = 0.1
batch_size = 32 * 2
alpha = 0.5

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


class DistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].

    """

    def __init__(self, batch_size, feature_dim, p=0.5, eps=1e-6):
        super(DistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0

        self.var_mu= torch.zeros([1, feature_dim]).cuda()
        self.var_std = torch.zeros([1, feature_dim]).cuda()
        self.alpha = 0.2

        print('Using alpha:', self.alpha)


    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=False) + self.eps).sqrt()
        return t

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        instance_mean = x.mean(dim=[2, 3], keepdim=False)  # instance level
        instance_std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt() # instance level
   

        batch_var_mu = instance_mean.var(dim=0, keepdim=False)
        batch_var_std = instance_std.var(dim=0, keepdim=False)

        self.var_mu = (1-self.alpha)*self.var_mu + self.alpha * batch_var_mu.data
        self.var_std = (1-self.alpha)*self.var_std + self.alpha * batch_var_std.data

        sqrtvar_mu = (self.var_mu + self.eps).sqrt().repeat(x.shape[0], 1)
        sqrtvar_std = (self.var_std + self.eps).sqrt().repeat(x.shape[0], 1)
    
        beta = self._reparameterize(instance_mean, sqrtvar_mu)
        gamma = self._reparameterize(instance_std, sqrtvar_std)

        x = (x - instance_mean.reshape(x.shape[0], x.shape[1], 1, 1)) / instance_std.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)

        return x
        




class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1, track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=momentum, track_running_stats=track_running_stats)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1, track_running_stats=track_running_stats)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1, track_running_stats=track_running_stats)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=0.1, track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class UResNet(nn.Module):

    def __init__(
            self, block, layers, pertubration=None, uncertainty=0.0, **kwargs
    ):
        self.inplanes = 64
        super().__init__()

        # backbone network
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64, momentum=0.1, track_running_stats=track_running_stats) ### BN
        self.relu = nn.ReLU(inplace=True)

        # resnet18
        self.pertubration0 = pertubration(batch_size=batch_size, feature_dim=64, p=uncertainty) if pertubration else nn.Identity()
        self.pertubration1 = pertubration(batch_size=batch_size, feature_dim=64, p=uncertainty) if pertubration else nn.Identity()
        self.pertubration2 = pertubration(batch_size=batch_size, feature_dim=64, p=uncertainty) if pertubration else nn.Identity()
        self.pertubration3 = pertubration(batch_size=batch_size, feature_dim=128, p=uncertainty) if pertubration else nn.Identity()
        self.pertubration4 = pertubration(batch_size=batch_size, feature_dim=256, p=uncertainty) if pertubration else nn.Identity()
        self.pertubration5 = pertubration(batch_size=batch_size, feature_dim=512, p=uncertainty) if pertubration else nn.Identity()


        # # resnet50 & 101
        # self.pertubration0 = pertubration(batch_size=batch_size, feature_dim=64, p=uncertainty) if pertubration else nn.Identity()
        # self.pertubration1 = pertubration(batch_size=batch_size, feature_dim=64, p=uncertainty) if pertubration else nn.Identity()
        # self.pertubration2 = pertubration(batch_size=batch_size, feature_dim=256, p=uncertainty) if pertubration else nn.Identity()
        # self.pertubration3 = pertubration(batch_size=batch_size, feature_dim=512, p=uncertainty) if pertubration else nn.Identity()
        # self.pertubration4 = pertubration(batch_size=batch_size, feature_dim=1024, p=uncertainty) if pertubration else nn.Identity()
        # self.pertubration5 = pertubration(batch_size=batch_size, feature_dim=2048, p=uncertainty) if pertubration else nn.Identity()


        

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)

        self._out_features = 512 * block.expansion

        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.1, track_running_stats=track_running_stats), ### BN
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x, label=None, disturb=None):
        x = self.conv1(x)
        if disturb:
            x = self.pertubration0(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if disturb:
            x = self.pertubration1(x)
        x = self.layer1(x)
        if disturb:
            x = self.pertubration2(x)
        x = self.layer2(x)
        if disturb:
            x = self.pertubration3(x)
        x = self.layer3(x)
        if disturb:
            x = self.pertubration4(x)
        x = self.layer4(x)
        if disturb:
            x = self.pertubration5(x)

        return x

    def forward(self, x, label=None, disturb=True):
        if label == None:
            f = self.featuremaps(x, disturb=disturb)
        else:
            f = self.featuremaps(x, label=label, disturb=disturb)

        v = self.global_avgpool(f)
        return v.view(v.size(0), -1)


def init_pretrained_weights(model, model_url):
    pretrain_dict = model_zoo.load_url(model_url)
    model.load_state_dict(pretrain_dict, strict=False)


"""
Residual network configurations:
--
resnet18: block=BasicBlock, layers=[2, 2, 2, 2]
resnet34: block=BasicBlock, layers=[3, 4, 6, 3]
resnet50: block=Bottleneck, layers=[3, 4, 6, 3]
resnet101: block=Bottleneck, layers=[3, 4, 23, 3]
resnet152: block=Bottleneck, layers=[3, 8, 36, 3]
"""
"""
Standard residual networks
"""


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def uresnet18(pretrained=True, uncertainty=0.5, **kwargs):
    model = UResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                    pertubration=DistributionUncertainty, uncertainty=uncertainty)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model


def uresnet50(pretrained=True, uncertainty=0.5, **kwargs):
    model = UResNet(block=Bottleneck, layers=[3, 4, 6, 3],
                    pertubration=DistributionUncertainty, uncertainty=uncertainty)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

def uresnet101(pretrained=True, uncertainty=0.5, **kwargs):
    model = UResNet(block=Bottleneck, layers=[3, 4, 23, 3],
                    pertubration=DistributionUncertainty, uncertainty=uncertainty)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet101'])

    return model



# def resnet18(pretrained=True, **kwargs):
#     model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
#     return model


# def resnet34(pretrained=True, **kwargs):
#     model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
#     return model


# def resnet50(pretrained=True, **kwargs):
#     model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
#     return model


# def resnet101(pretrained=True, **kwargs):
#     model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
#     return model


# def resnet152(pretrained=True, **kwargs):
#     model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
#     return model




