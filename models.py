import torch
from torch import nn
from backbones.backbone_builder import BackboneBuilder
from attention import AttentionAggregator

class ClinicalOnly(nn.Module):
    """Training with clinical only"""

    def __init__(self, num_classes, clinical_data_size=5, expand_times=10):
        super().__init__()

        print('training with clinical only')
        self.clinical_data_size = clinical_data_size
        self.expand_times = expand_times
        self.classifier = nn.Sequential(
            nn.Linear(self.clinical_data_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, clinical_data=None):
        result = self.classifier(clinical_data.float())
        return result

class MILNetImageOnly_(nn.Module):
    """Training with image only"""

    def __init__(self, num_classes, backbone_name):
        super().__init__()

        print('training with image only')
        self.image_feature_extractor = BackboneBuilder(backbone_name)
        self.attention_aggregator = AttentionAggregator(self.image_feature_extractor.output_features_size, 1)  # inner_feature_size=1
        self.classifier = nn.Sequential(
            nn.Linear(self.attention_aggregator.L, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, bag_data, clinical_data=None):
        bag_data = bag_data.squeeze(0)  # [1 (batch size), N, C, H, W] --> [N, C, H, W], remove the batch dimension
        patch_features = self.image_feature_extractor(bag_data)
        aggregated_feature, attention = self.attention_aggregator(patch_features)
        result = self.classifier(aggregated_feature)

        return result, aggregated_feature, aggregated_feature

class MILNetImageOnly(nn.Module):
    """Training with image only"""

    def __init__(self, num_classes, backbone_name):
        super().__init__()

        print('training with image only')
        self.image_feature_extractor = BackboneBuilder(backbone_name)
        self.attention_aggregator = AttentionAggregator(self.image_feature_extractor.output_features_size, 1)  # inner_feature_size=1
        self.classifier = nn.Sequential(
            nn.Linear(self.attention_aggregator.L+50, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        self.pro_clinicalData = nn.Sequential(
            nn.Linear(5, 100),
            nn.ReLU(),
            nn.Linear(100,50)
        )

        self.unk = nn.Parameter(torch.randn((1, 5), requires_grad=True))
        self.register_parameter("Ablah", self.unk)

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, bag_data, clinical_data=None):
        bag_data = bag_data.squeeze(0)  # [1 (batch size), N, C, H, W] --> [N, C, H, W], remove the batch dimension
        patch_features = self.image_feature_extractor(bag_data)
        aggregated_feature, attention = self.attention_aggregator(patch_features)
        prompt = self.pro_clinicalData(self.unk.float())
        fused_data = torch.cat([aggregated_feature, prompt], dim=-1)
        result = self.classifier(fused_data)

        return result, prompt, aggregated_feature


class MILNetWithClinicalData(nn.Module):
    """Training with image and clinical data"""

    def __init__(self, num_classes, backbone_name, clinical_data_size=5, expand_times=10):
        super().__init__()

        print('training with image and clinical data')
        self.clinical_data_size = clinical_data_size
        self.expand_times = expand_times  # expanding clinical data to match image features in dimensions

        self.image_feature_extractor = BackboneBuilder(backbone_name)
        self.attention_aggregator = AttentionAggregator(self.image_feature_extractor.output_features_size, 1)  # inner_feature_size=1
        self.pro_clinicalData = nn.Linear(5, 50)
        self.classifier = nn.Sequential(
            nn.Linear(self.attention_aggregator.L + self.clinical_data_size * self.expand_times, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, bag_data, clinical_data):
        bag_data = bag_data.squeeze(0)  # [1 (batch size), N, C, H, W] --> [N, C, H, W], remove the batch dimension
        patch_features = self.image_feature_extractor(bag_data)
        aggregated_feature, attention = self.attention_aggregator(patch_features)
        prompt = self.pro_clinicalData(clinical_data.reshape(1,5).float())
        fused_data = torch.cat([aggregated_feature, prompt], dim=-1)  # feature fusion
        result = self.classifier(fused_data)

        return result, prompt, aggregated_feature
