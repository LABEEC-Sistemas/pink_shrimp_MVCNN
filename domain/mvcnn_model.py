import torch
import torch.nn as nn
import torchvision.models as models

class MVCNNModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )# Backbone sem avgpool e sem fc

        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )  #Adaptação necessária para plotar o Grad-CAM para cada imagem processada pela rede

        self.avgpool = backbone.avgpool

        self.aggregation = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Linear(
            backbone.fc.in_features,
            num_classes
        )

        self.feature_maps = None
        self.features = None

    def forward(self,x,return_maps=False, return_features=False): #Adaptação necessária para plotar o Grad-CAM para cada imagem processada pela rede
        batch_size, num_views, c, h, w = x.shape

        x = x.view(batch_size * num_views,c,h,w)

        feature_maps = self.backbone(x) # Mapas convolucionais

        self.feature_maps = feature_maps

        if feature_maps.requires_grad:

            feature_maps.retain_grad()

        pooled = self.avgpool(feature_maps) # Global Average Pooling

        pooled = torch.flatten(pooled,1)

        pooled = pooled.view(batch_size,num_views,-1) # MVCNN

        self.features = pooled

        aggregated = self.aggregation(pooled.transpose(1,2)).squeeze(-1)

        logits = self.classifier(aggregated)

        outputs = (logits,)

        if return_maps:
            _, C, H, W = feature_maps.shape

            outputs += (
                feature_maps.view(batch_size,num_views,C,H,W),
            )

        if return_features:
            outputs += (pooled,)

        if len(outputs) == 1:
            return outputs[0]
        
        return outputs