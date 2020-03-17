from algonauts.src.lib.networks.alexnet import AlexNet
from algonauts.src.lib.networks.vgg import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from algonauts.src.lib.networks.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from algonauts.src.lib.networks.squeezenet import SqueezeNet1_0, SqueezeNet1_1
from algonauts.src.lib.networks.densenet import densenet121, densenet161, densenet169, densenet201
from algonauts.src.lib.networks.alexnet import AlexNet
from algonauts.src.lib.networks.inceptionv3 import inception_v3
from algonauts.src.lib.networks.googlenet import googlenet


models = {
    'alexnet': AlexNet,
    'vgg11': vgg11,
    'vgg11_bn': vgg11_bn,
    'vgg13': vgg13,
    'vgg13_bn': vgg13_bn,
    'vgg16': vgg16,
    'vgg16_bn': vgg16_bn,
    'vgg19': vgg19,
    'vgg19_bn': vgg19_bn,
    'sqnet1_0': SqueezeNet1_0,
    'sqnet1_1': SqueezeNet1_1,
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'densenet121': densenet121,
    'densenet161': densenet161,
    'densenet169': densenet169,
    'densenet201': densenet201,
    'googlenet': googlenet,
    'inception': inception_v3
}
