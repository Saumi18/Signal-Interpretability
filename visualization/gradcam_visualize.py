import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys

sys.path.append("..")

from Models.full_model import RFClassifier
from utils.gradcam import GradCAM

MODEL_PATH = "../modulation_model.pth"

SPEC_PATH = "../spectrogram_matrices_dataset/clean/BPSK/0.npy"


mod_map = {
0:"BPSK",
1:"QPSK",
2:"8PSK",
3:"16QAM",
4:"64QAM",
5:"2FSK",
6:"4FSK",
7:"GMSK",
8:"AM",
9:"FM"
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RFClassifier().to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model.eval()


target_layer = model.backbone.conv3

gradcam = GradCAM(model, target_layer)

spec = np.load(SPEC_PATH)

input_tensor = torch.tensor(spec).float().unsqueeze(0).unsqueeze(0).to(device)

features = model.backbone(input_tensor)

output = model.mod_classifier(features)

pred_class = torch.argmax(output,1).item()

print("Predicted:", mod_map[pred_class])

cam = gradcam.generate(input_tensor, pred_class)

heatmap = cv2.applyColorMap(
    np.uint8(255 * cam),
    cv2.COLORMAP_JET
)

heatmap = heatmap.astype(np.float32) / 255


spec_norm = (spec - spec.min()) / (spec.max() - spec.min())

spec_rgb = np.stack([spec_norm]*3, axis=2)

overlay = heatmap*0.5 + spec_rgb

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Spectrogram")
plt.imshow(spec, cmap="viridis")
plt.axis("off")

plt.subplot(1,3,2)
plt.title("GradCAM")
plt.imshow(cam, cmap="jet")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Overlay")
plt.imshow(overlay)
plt.axis("off")

plt.tight_layout()
plt.show()
