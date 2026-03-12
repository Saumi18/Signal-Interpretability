import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.dataset_loader import RFdataset
from models.full_model import RFSignalModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = RFdataset("dataset","dual")

loader = DataLoader(dataset,batch_size=32,shuffle=True)

model = RFSignalModel().to(device)

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

for epoch in range(15):

    for x,y in loader:

        x = x.to(device)
        y = y.to(device).float()

        _,_,_,dual_out = model(x)

        loss = criterion(dual_out,y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    print("Epoch",epoch,"done")