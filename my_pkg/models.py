import torchvision.utils
import copy
import torch
import torchvision

# ================================================
# ============  Segmentation Models  =============
# ================================================

class FCNClassifier(torch.nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.classifier = torchvision.models.segmentation.fcn.FCNHead(in_channels, n_classes)
        self.backbone_layer = 'layer4'

    def forward(self, x):
        return self.classifier(x[self.backbone_layer])


class FCNUpsampleBlock(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int):

        super().__init__()

        up_rate = 2
        self.upsample = torch.nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=3, stride=up_rate, padding=1, output_padding=up_rate-1)
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels + out_channels, out_channels=out_channels, kernel_size=1, stride=1)

    def forward(self, x, skip_connection):
        x = self.conv(torch.concat([self.upsample(x), skip_connection], dim=1))
        return x


class FCNSegmentation(torch.nn.Module):
    def __init__(self, n_classes, type_=32, backbone_model='resnet50'):
        super().__init__()

        if type_ not in (32, 16, 8, 4):
            raise ValueError(f"Input type '{type_}' is not in (32, 16, 8, 4)")
        self.type_ = type_

        self.backbone_layers = []

        if backbone_model == 'resnet50':
            upsample_in_channels = [1024, 512, 256]
        elif backbone_model == 'resnet18':
            upsample_in_channels = [256, 128, 64]
        else:
            raise RuntimeError(f'Unsupported backbone model "{backbone_model}"...')

        if self.type_ < 32:
            self.upsample_x2 = FCNUpsampleBlock(in_channels=upsample_in_channels[0], out_channels=n_classes)
            self.backbone_layers.append("layer3")

        if self.type_ < 16:
            self.upsample_x4 = FCNUpsampleBlock(in_channels=upsample_in_channels[1], out_channels=n_classes)
            self.backbone_layers.append("layer2")

        if self.type_ < 8:
            self.upsample_x8 = FCNUpsampleBlock(in_channels=upsample_in_channels[2], out_channels=n_classes)
            self.backbone_layers.append("layer1")

    def forward(self, classifier_output, backbone_layers, input_shape):

        x = classifier_output
        if self.type_ < 32:
            x = self.upsample_x2(x, backbone_layers['layer3'])

        if self.type_ < 16:
            x = self.upsample_x4(x, backbone_layers['layer2'])

        if self.type_ < 8:
            x = self.upsample_x8(x, backbone_layers['layer1'])

        x = torch.nn.functional.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)

        return x


class ResnetSegmentation(torch.nn.Module):

    def __init__(self, n_classes: int, seg_type=4, backbone_trainable=True, resnet_type='resnet50'):
        super().__init__()
        self._config = {}
        self._init_config(locals())

        self.train_history = []

        self.n_classes = n_classes

        if resnet_type == 'resnet18':
            create_resnet = torchvision.models.resnet18
        elif resnet_type == 'resnet50':
            create_resnet = torchvision.models.resnet50
        else:
            raise RuntimeError(f'Unsupported resnet type "{resnet_type}"...')

        self.backbone = create_resnet(pretrained=False, progress=False)

        if not backbone_trainable:
            for param in self.backbone.parameters():
                param.requires_grad = False

        bb_features = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Flatten(start_dim=1)  # LambdaLayer(lambda x: torch.flatten(x,1))

        backbone_out_layers = {}
        self.classifier = FCNClassifier(bb_features, n_classes)
        self.segmentation = FCNSegmentation(n_classes, type_=seg_type, backbone_model=resnet_type)
        backbone_out_layers[self.classifier.backbone_layer] = self.classifier.backbone_layer
        for layer in self.segmentation.backbone_layers:
            backbone_out_layers[layer] = layer

        from torchvision.models._utils import IntermediateLayerGetter
        self.backbone = IntermediateLayerGetter(self.backbone, return_layers=backbone_out_layers)

    def _init_config(self, locals_dict):
        self._config = locals_dict
        self._config.pop('self')
        self._config.pop('__class__')

    def config_state_dict(self):
        return {'config': copy.deepcopy(self._config), 'state_dict': copy.deepcopy(self.state_dict()),
                'class_name': self.__class__.__name__,
                'train_history': self.train_history}

    @classmethod
    def from_config_state_dict(cls, s):
        class_name = s.pop('class_name', None)
        if not class_name:
            raise RuntimeError('Failed to load class name...')
        if class_name != cls.__name__:
            raise RuntimeError(f"Loaded class {class_name} != from called class {cls.__name__}")

        model = cls(**s['config'])
        model.load_state_dict(s['state_dict'])
        model.train_history = s['train_history']
        return model

    def save(self, filename):
        import pickle
        pickle.dump(self.config_state_dict(), open(filename, 'wb'))

    @classmethod
    def load(cls, filename):
        import pickle
        return cls.from_config_state_dict(pickle.load(open(filename, 'rb')))

    def forward(self, x):
        input_shape = x.shape[-2:]

        bb_out = self.backbone(x)

        out = {}
        x = self.classifier(bb_out)
        out['seg_mask'] = self.segmentation(x, bb_out, input_shape)

        return out

    def output(self, x, return_prob=False):

        seg_mask = self.forward(x)['seg_mask']
        out = torch.argmax(seg_mask, dim=1, keepdim=False)

        if return_prob:
            prob = torch.max(torch.nn.functional.softmax(seg_mask, dim=1), dim=1, keepdim=False)[0]
            out = (out, prob)

        return out
