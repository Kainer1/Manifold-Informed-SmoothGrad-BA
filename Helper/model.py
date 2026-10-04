"""The fixed VGG16 classifier used by the saved-sample analyses."""

from torchvision.models import VGG16_Weights, vgg16


def load_vgg16(device):
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    return model.to(device).eval()
