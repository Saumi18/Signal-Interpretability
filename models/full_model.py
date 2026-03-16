import torch.nn as nn
from models.backbone import RFBackbone
from models.classifiers import JammingClassifier, MixedClassifier, ModulationClassifier, DualModulationClassifier

class RFClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = RFBackbone()

        self.jam_classifier = JammingClassifier()

        self.mix_classifier = MixedClassifier()

        self.mod_classifier = ModulationClassifier()

        self.dual_classifier = DualModulationClassifier()

    def forward(self, x):
        features = self.backbone(x)
        jam_pred = self.jam_classifier(features)
        mix_pred = self.mix_classifier(features)
        mod_pred = self.mod_classifier(features)
        return jam_pred, mix_pred, mod_pred
