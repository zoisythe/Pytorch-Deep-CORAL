"""AlexNet with differentiable, pre-ReLU convolutional and linear features."""

from torch import nn
from torchvision.models import AlexNet_Weights, alexnet

FC_LAYERS = ('fc6', 'fc7', 'fc8')
CORAL_LAYERS = ('conv5', *FC_LAYERS)
LAYER_INDICES = {'fc6': 1, 'fc7': 4, 'fc8': 6}


class IdentityAwareAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def forward(self, input):
        # For 224x224 AlexNet inputs, features are already 6x6. Avoid calling
        # a CUDA adaptive-pool backward that rejects strict determinism,
        # when the operation is exactly identity anyway.
        if input.shape[-2:] == self.output_size:
            return input
        return super().forward(input)


class Net(nn.Module):
    def __init__(self, num_classes, pretrained=False, coral_layers=('fc8',)):
        super().__init__()
        if not coral_layers or not set(coral_layers).issubset(CORAL_LAYERS):
            raise ValueError('coral_layers must select conv5, fc6, fc7 and/or fc8')
        self.coral_layers = tuple(layer for layer in CORAL_LAYERS if layer in coral_layers)
        # Load the original ImageNet head before replacing it with the Office head.
        self.model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1 if pretrained else None)
        self.model.avgpool = IdentityAwareAdaptiveAvgPool2d((6, 6))
        self.model.classifier[6] = nn.Linear(4096, num_classes)
        nn.init.normal_(self.model.classifier[6].weight, mean=0, std=5e-3)
        nn.init.constant_(self.model.classifier[6].bias, 0.01)
        self._features = None
        for layer in self.coral_layers:
            module = self.model.features[10] if layer == 'conv5' else self.model.classifier[LAYER_INDICES[layer]]
            module.register_forward_hook(self._hook(layer))

    def _hook(self, layer):
        def save_features(module, inputs, output):
            if self._features is not None:
                # AlexNet's next ReLU is in-place. Clone now, without detach(),
                # to keep the Conv/Linear output and the CORAL gradient intact.
                self._features[layer] = output.clone()
        return save_features

    def forward(self, x, return_features=False):
        features = {} if return_features else None
        self._features = features
        try:
            logits = self.model(x)
        finally:
            # Do not retain an autograd graph on the module between forwards.
            self._features = None
        return (logits, features) if return_features else logits
