"""ResNet-50 V2 with spatial, pooled and classifier CORAL features."""

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

RESNET_LAYERS = ('layer4', 'pooled', 'logits')


class ResNetNet(nn.Module):
    def __init__(self, num_classes, pretrained=True, coral_layers=RESNET_LAYERS):
        super().__init__()
        if not set(coral_layers).issubset(RESNET_LAYERS):
            raise ValueError('ResNet supports layer4, pooled and logits')
        self.coral_layers = tuple(layer for layer in RESNET_LAYERS if layer in coral_layers)
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.model.fc = nn.Linear(2048, num_classes)
        nn.init.normal_(self.model.fc.weight, mean=0, std=0.005)
        nn.init.constant_(self.model.fc.bias, 0.01)

    def forward(self, x, return_features=False):
        net = self.model
        x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
        x = net.layer4(net.layer3(net.layer2(net.layer1(x))))
        # Equivalent global average pool, without CUDA adaptive-pool backward.
        pooled = x.mean(dim=(-2, -1))
        logits = net.fc(pooled)
        if not return_features:
            return logits
        features = {'layer4': x, 'pooled': pooled, 'logits': logits}
        features = {layer: features[layer] for layer in self.coral_layers}
        expected = {'layer4': (x.shape[0], 2048, 7, 7),
                    'pooled': (x.shape[0], 2048), 'logits': (x.shape[0], net.fc.out_features)}
        for layer, value in features.items():
            if tuple(value.shape) != expected[layer]:
                raise ValueError(f'Unexpected ResNet {layer} shape: {tuple(value.shape)}')
            if torch.is_grad_enabled() and self.training and not value.requires_grad:
                raise RuntimeError(f'Disconnected ResNet feature: {layer}')
        return logits, features

    def optimizer_groups(self, lr):
        head_ids = {id(parameter) for parameter in self.model.fc.parameters()}
        return [{'params': [parameter for parameter in self.parameters() if id(parameter) not in head_ids]},
                {'params': list(self.model.fc.parameters()), 'lr': 10 * lr}]

    def batchnorm_counts(self):
        return {name: module.num_batches_tracked.item() for name, module in self.named_modules()
                if isinstance(module, nn.BatchNorm2d)}

    def batchnorm_snapshot(self):
        return {name: buffer.detach().clone() for name, buffer in self.named_buffers()
                if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}

    def assert_batchnorm_unchanged(self, before):
        buffers = dict(self.named_buffers())
        if any(not torch.equal(value, buffers[name]) for name, value in before.items()):
            raise RuntimeError('Evaluation modified BatchNorm statistics')
