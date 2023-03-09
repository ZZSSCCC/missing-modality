from torch import nn
import torchvision

BACKBONES = [
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "densenet121",
    "densenet161",
    "densenet169",
    "densenet201",
    "alexnet",
]


class BackboneBuilder(nn.Module):
    """Build backbone with the last fc layer removed"""

    def __init__(self, backbone_name):
        super().__init__()

        assert backbone_name in BACKBONES

        if backbone_name.startswith("vgg"):
            assert backbone_name in ["vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn"]
            self.extractor, self.output_features_size = self.vgg(complete_backbone)
        elif backbone_name.startswith("resnet"):
            assert backbone_name in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
            self.extractor, self.output_features_size = self.resnet(backbone_name, complete_backbone)
        elif backbone_name.startswith("densenet"):
            assert backbone_name in ["densenet121", "densenet161", "densenet169", "densenet201"]
            self.extractor, self.output_features_size = self.densenet(backbone_name, complete_backbone)
        elif backbone_name == "alexnet":
            self.extractor, self.output_features_size = self.alexnet(complete_backbone)
        else:
            raise NotImplementedError

    def forward(self, x):
        patch_features = self.extractor(x)

        return patch_features

    def vgg(self, complete_backbone):
        output_features_size = 512 * 7 * 7
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))

        return extractor, output_features_size

    def resnet(self, backbone_name, complete_backbone):
        if backbone_name in ["resnet18", "resnet34"]:
            output_features_size = 512 * 1 * 1
        else:
            output_features_size = 2048 * 1 * 1
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))

        return extractor, output_features_size

    def densenet(self, backbone_name, complete_backbone):
        if backbone_name == "densenet121":
            output_features_size = 1024 * 7 * 7
        elif backbone_name == "densenet161":
            output_features_size = 2208 * 7 * 7
        elif backbone_name == "densenet169":
            output_features_size = 1664 * 7 * 7
        else:
            output_features_size = 1920 * 7 * 7
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))

        return extractor, output_features_size

    def alexnet(self, complete_backbone):
        output_features_size = 256 * 6 * 6
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))

        return extractor, output_features_size
