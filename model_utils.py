import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import timm

from config import IMG_SIZE, NUM_CLASSES, CLASS_NAMES, NORM_MEAN, NORM_STD, MODEL_PATH


class ModernCXRModel(nn.Module):
    """
    Same model structure used in your Colab notebook:
    EfficientNetV2-S backbone + custom classification head.
    """
    def __init__(self, model_name="tf_efficientnetv2_s.in21k_ft_in1k", num_classes=4, in_chans=1, drop=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            in_chans=in_chans,
            global_pool="avg"
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    device = get_device()
    model = ModernCXRModel(num_classes=NUM_CLASSES, in_chans=1).to(device)

    if not os.path.exists(MODEL_PATH):
        return model, device, False

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    # Supports both direct state_dict and checkpoint dictionaries.
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, device, True


def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")

    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[NORM_MEAN], std=[NORM_STD])
    ])

    tensor = transform(image).unsqueeze(0)
    return image, tensor


def predict_image(model, device, image_path):
    _, tensor = preprocess_image(image_path)
    tensor = tensor.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    probability_table = [
        {"class_name": CLASS_NAMES[i], "probability": float(probs[i]), "percentage": float(probs[i] * 100)}
        for i in range(len(CLASS_NAMES))
    ]
    probability_table = sorted(probability_table, key=lambda x: x["probability"], reverse=True)

    return {
        "pred_class": pred_class,
        "confidence": confidence,
        "confidence_percent": confidence * 100,
        "probabilities": probability_table,
        "raw_probabilities": probs.tolist()
    }
