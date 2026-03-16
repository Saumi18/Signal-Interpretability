import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("..")

from models.full_model import RFClassifier
from utils.gradcam import GradCAM

DATASET_PATH = "../spectrogram_matrices_dataset/clean/BPSK"

MODEL_PATH = "../modulation_model.pth"

NUM_SAMPLES = 30


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RFClassifier().to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model.eval()

target_layer = model.backbone.conv3

gradcam = GradCAM(model, target_layer)

files = os.listdir(DATASET_PATH)

files = files[:NUM_SAMPLES]

avg_cam = np.zeros((128,128))

for f in files:

    path = os.path.join(DATASET_PATH, f)

    spec = np.load(path)

    input_tensor = torch.tensor(spec).float().unsqueeze(0).unsqueeze(0).to(device)

    features = model.backbone(input_tensor)

    output = model.mod_classifier(features)

    pred_class = torch.argmax(output,1).item()

    cam = gradcam.generate(input_tensor, pred_class)

    avg_cam += cam


avg_cam = avg_cam / NUM_SAMPLES

plt.figure(figsize=(6,5))

plt.title("Average GradCAM (BPSK)")

plt.imshow(avg_cam, cmap="jet")

plt.colorbar()

plt.axis("off")

plt.show()
