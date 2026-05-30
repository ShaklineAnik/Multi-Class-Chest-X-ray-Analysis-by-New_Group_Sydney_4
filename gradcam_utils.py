import os
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from config import IMG_SIZE, RESULT_FOLDER
from model_utils import preprocess_image


class SimpleGradCAM:
    """
    Manual Grad-CAM implementation for your EfficientNetV2-S model.
    It hooks the last Conv2d layer automatically.
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self.target_layer = self._find_last_conv_layer()
        self._register_hooks()

    def _find_last_conv_layer(self):
        last_conv = None
        for _, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError("No Conv2d layer found for Grad-CAM.")
        return last_conv

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[:, target_class]
        score.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1).squeeze()
        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy()

        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam, target_class


def create_gradcam_image(model, device, image_path, output_name):
    os.makedirs(RESULT_FOLDER, exist_ok=True)

    original_image, tensor = preprocess_image(image_path)
    tensor = tensor.to(device)

    gradcam = SimpleGradCAM(model)
    cam, target_class = gradcam.generate(tensor)

    # Prepare original image as RGB at model input size.
    original_resized = original_image.resize((IMG_SIZE, IMG_SIZE))
    original_np = np.array(original_resized).astype(np.float32) / 255.0
    original_rgb = np.stack([original_np, original_np, original_np], axis=-1)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    overlay = 0.55 * original_rgb + 0.45 * heatmap
    overlay = np.clip(overlay, 0, 1)

    out_path = os.path.join(RESULT_FOLDER, output_name)
    Image.fromarray(np.uint8(overlay * 255)).save(out_path)

    return out_path
