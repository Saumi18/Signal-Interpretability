import torch
import numpy as np

from Models.full_model import RFClassifier
from utils.signal_to_spectrogram import iq_to_spectrogram
from utils.gradcam import GradCAM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SignalInterpreter:

    def __init__(self):

        self.model = RFClassifier().to(device)

        self.model.load_state_dict(
            torch.load("models/trained_model.pth", map_location=device)
        )

        self.model.eval()

        self.gradcam = GradCAM(self.model, self.model.backbone.layer4)


    def predict(self, iq_signal):

        spec = iq_to_spectrogram(iq_signal)

        x = torch.tensor(spec).float().unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():

            features = self.model.backbone(x)

            jam_pred = self.model.jam_classifier(features)

        jam_class = torch.argmax(jam_pred).item()

        if jam_class == 1:

            cam = self.gradcam.generate(x,1)

            return {
                "type":"JAMMED",
                "heatmap":cam,
                "spectrogram":spec
            }


        mix_pred = self.model.mix_classifier(features)

        mix_class = torch.argmax(mix_pred).item()


        if mix_class == 0:

            mod_pred = self.model.mod_classifier(features)

            mod_class = torch.argmax(mod_pred).item()

            cam = self.gradcam.generate(x,mod_class)

            return {
                "type":"CLEAN",
                "modulation":mod_class,
                "heatmap":cam,
                "spectrogram":spec
            }


        else:

            dual_pred = self.model.dual_classifier(features)

            probs = torch.sigmoid(dual_pred)

            mods = torch.where(probs>0.5)[1].cpu().numpy()

            cam = self.gradcam.generate(x,mods[0])

            return {
                "type":"MIXED",
                "modulations":mods,
                "heatmap":cam,
                "spectrogram":spec
            }
