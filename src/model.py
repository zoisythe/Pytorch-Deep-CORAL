from torch import nn
from torchvision.models import AlexNet_Weights, alexnet


class Net(nn.Module):

    def __init__(self, num_classes, pretrained=False):
        super().__init__()

        # check https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
        self.model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1 if pretrained else None, num_classes=num_classes)

        # if we want to feed 448x448 images
        # self.model.avgpool = nn.AdaptiveAvgPool2d(1)

        # In case we want to apply the loss to any other layer than the last
        # we need a forward hook on that layer
        # def save_features_layer_x(module, input, output):
        #     self.layer_x = output

        # This is a forward hook. Is executed each time forward is executed
        # self.model.layer4.register_forward_hook(save_features_layer_x)

    def forward(self, x):
        out = self.model(x)
        return out  # , self.layer_x
