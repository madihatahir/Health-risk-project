# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:15:49 2026

@author: Uesr
"""

# train.py
import torch
from dataset import HealthDataset
from models.lstm import LSTMModel
from models.vae import VAE
from models.fusion import FusionModel

dataset = HealthDataset("data/health.csv")
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

lstm = LSTMModel(input_size=10)
vae = VAE(input_dim=50)
fusion = FusionModel()

opt = torch.optim.Adam(list(lstm.parameters()) +
                        list(vae.parameters()) +
                        list(fusion.parameters()), lr=1e-3)

loss_fn = torch.nn.BCELoss()

for epoch in range(20):
    for x, y in loader:

        # split dummy structure
        physio = x[:, :10].unsqueeze(1)   # time-series
        omics = x[:, 10:60]

        # LSTM features
        physio_feat = lstm(physio)

        # VAE features
        mu, logvar = vae.encode(omics)
        omics_feat = mu

        # prediction
        pred = fusion(physio_feat, omics_feat)

        loss = loss_fn(pred.squeeze(), y)

        opt.zero_grad()
        loss.backward()
        opt.step()

    print("Epoch:", epoch, "Loss:", loss.item())