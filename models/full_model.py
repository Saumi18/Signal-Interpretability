import torch.nn as nn
from .backbone import RFBackbone
from .classifiers import *

class RFClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = RFBackbone()

        self.jam_classifier = JammingClassifier()

        self.mix_classifier = MixedClassifier()

        self.mod_classifier = ModulationClassifier()

        self.dual_classifier = DualModulationClassifier()
