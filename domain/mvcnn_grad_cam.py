import torch.nn.functional as F

class MVCNNGradCAM:

    def __init__(self, model):

        self.model = model

    def generate(
        self,
        images,
        target_class=None
    ):

        self.model.eval()

        device = next(
            self.model.parameters()
        ).device

        images = images.unsqueeze(0).to(device)

        output = self.model(images)

        if target_class is None:
            target_class = output.argmax(1)

        self.model.zero_grad()

        output[:,target_class].backward()

        feature_maps = self.model.feature_maps

        gradients = feature_maps.grad

        weights = gradients.mean(
            dim=(2,3),
            keepdim=True
        )

        cam = (
            weights*feature_maps
        ).sum(1)

        cam = F.relu(cam)

        cam -= cam.min(
            dim=1,
            keepdim=True
        )[0]

        cam /= (
            cam.max(
                dim=1,
                keepdim=True
            )[0]
            +1e-8
        )

        return cam.cpu().detach().numpy()