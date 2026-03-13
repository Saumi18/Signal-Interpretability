import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.dataset_loader import RFdataset
from Models.classifiers import RFClassifier
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-3

dataset_path = "/content/drive/MyDrive/spectrogram_matrices_dataset"

train_dataset = RFdataset(dataset_path,"dual","train")
val_dataset = RFdataset(dataset_path,"dual","val")

train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE)

model = RFClassifier().to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(),lr=LR)

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for x,y in tqdm(train_loader):

        x = x.to(device)
        y = y.to(device).float()

        optimizer.zero_grad()

        features = model.backbone(x)

        out = model.dual_classifier(features)

        loss = criterion(out,y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print("Epoch",epoch,"Loss",total_loss/len(train_loader))

    model.eval()

    with torch.no_grad():

        total_loss = 0

        for x,y in val_loader:

            x = x.to(device)
            y = y.to(device).float()

            features = model.backbone(x)

            out = model.dual_classifier(features)

            loss = criterion(out,y)

            total_loss += loss.item()

    print("Validation Loss:",total_loss/len(val_loader))

torch.save(model.state_dict(),"dual_modulation_model.pth")
