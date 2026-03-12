import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.backbone import RFBackbone
from models.classifiers import JammingClassifier, MixedClassifier, ModulationClassifier, DualModulationClassifier
import torch.nn as nn

class RFSignalModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = RFBackbone()

        self.jam_classifier = JammingClassifier()

        self.mix_classifier = MixedClassifier()

        self.mod_classifier = ModulationClassifier()

        self.dual_classifier = DualModulationClassifier()

    def forward(self,x):

        features = self.backbone(x)

        jam_out = self.jam_classifier(features)

        mix_out = self.mix_classifier(features)

        mod_out = self.mod_classifier(features)

        dual_out = self.dual_classifier(features)

        return jam_out,mix_out,mod_out,dual_out