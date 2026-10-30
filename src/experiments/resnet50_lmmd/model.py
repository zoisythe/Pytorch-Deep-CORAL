"""Author DSAN topology with explicitly requested torchvision V2 weights."""

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class DSANV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.fc = nn.Identity()
        self.bottleneck = nn.Linear(2048, 256)
        self.classifier = nn.Linear(256, 31)

    def forward(self, images, return_features=False):
        net = self.backbone
        spatial = net.maxpool(net.relu(net.bn1(net.conv1(images))))
        spatial = net.layer4(net.layer3(net.layer2(net.layer1(spatial))))
        features = self.bottleneck(spatial.mean(dim=(-2, -1)))
        logits = self.classifier(features)
        return (logits, features) if return_features else logits

    def batchnorm_counts(self):
        return {name: module.num_batches_tracked.item() for name, module in self.named_modules()
                if isinstance(module, nn.BatchNorm2d)}

    def batchnorm_snapshot(self):
        return {name: buffer.detach().clone() for name, buffer in self.named_buffers()
                if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}

    def assert_batchnorm_unchanged(self, before):
        buffers = dict(self.named_buffers())
        if any(not torch.equal(value, buffers[name]) for name, value in before.items()):
            raise RuntimeError('Evaluation changed BatchNorm statistics')
