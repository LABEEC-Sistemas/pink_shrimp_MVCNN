import torch.nn as nn
import torchvision.models as models
 
class MVCNNModel(nn.Module):
    def __init__(self, num_classes):
        super(MVCNNModel, self).__init__()
        self.feature_extractor = models.resnet18(pretrained=True) #utilizando resnet18
        num_ftrs = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Identity()
        self.aggregation_layer = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Linear(num_ftrs, num_classes)
 
    def forward(self, x):
        batch_size, num_views, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        features = self.feature_extractor(x)
        features = features.view(batch_size, num_views, -1).transpose(1, 2)
        aggregated_features = self.aggregation_layer(features).squeeze(2)
        output = self.classifier(aggregated_features)
        return output