from models import ResNet
from models import UResNet
from models import UResNet_gsum
from models import UResNet_gsum_o
from models import UResNet_cwgsum

from models import classifier


encoders_map = {
    'resnet18': ResNet.resnet18,
    'resnet50': ResNet.resnet50,
    'convnet': ResNet.ConvNet,
    'uresnet18': UResNet.uresnet18,
    'uresnet50': UResNet.uresnet50,
    'uresnet101': UResNet.uresnet101,
    'uresnet18_global_sum': UResNet_gsum.uresnet18,
    'uresnet18_gsum_o': UResNet_gsum_o.uresnet18,
    'uresnet18_cwgsum': UResNet_cwgsum.uresnet18,
    'uresnet50_global_sum': UResNet_gsum.uresnet50,
    'uresnet101_global_sum': UResNet_gsum.uresnet101,
}

classifiers_map = {
    'base': classifier.Classifier,
}

def get_encoder(name):
    if name not in encoders_map:
        raise ValueError('Name of network unknown %s' % name)

    def get_network_fn(**kwargs):
        return encoders_map[name](**kwargs)

    return get_network_fn


def get_encoder_from_config(config):
    return get_encoder(config["name"])()


def get_classifier(name):
    if name not in classifiers_map:
        raise ValueError('Name of network unknown %s' % name)

    def get_network_fn(**kwargs):
        return classifiers_map[name](**kwargs)

    return get_network_fn


def get_classifier_from_config(config):
    return get_classifier(config["name"])(
        in_dim=config["in_dim"],
        num_classes=config["num_classes"]
    )

def get_multi_bi_classifier_from_config(config):
    return get_classifier(config["name"])(
        in_dim=config["in_dim"],
        num_classes=config["num_classes"]*2
    )