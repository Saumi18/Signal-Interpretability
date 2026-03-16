import torch
import torch.nn as nn
import torchvision.models as models
#Jamming Classifier
# 0- Not Jammed
# 1- Jammed
class JammingClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,2)
        )

    def forward(self,x):

        return self.fc(x)
    
#Mixed vs clean classifier
#classifies only if not jammed
# 0 - clean
# 1 - mixed
class MixedClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,2)
        )

    def forward(self,x):

        return self.fc(x)
    
# Clean Modulation classifier
class ModulationClassifier(nn.Module):

    def __init__(self,num_classes=10):

        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,num_classes)
        )

    def forward(self,x):

        return self.fc(x)

#Mixed modulations classifier
class DualModulationClassifier(nn.Module):

    def __init__(self,num_classes=10):

        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,num_classes),
            nn.Sigmoid()
        )

    def forward(self,x):

        return self.fc(x)
