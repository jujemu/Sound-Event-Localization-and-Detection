# import click
import os
from datetime import datetime
import joblib
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import Audio2Vector
from models import Network


# @click.command()
# @click.option('--input', 'input_dir', help='Directory of wave files and metadata', required=True)
def main(
    input_dir='Development Datasets',
):  
    date_time = datetime.now().strftime("%Y-%m-%d_%H%M")
    weight_output = os.path.join('./checkpoint', date_time)
    os.makedirs(weight_output, exist_ok=True)

    device = 'cpu'

    model = Network()
    model = model.to(device, dtype=torch.float64)

    ds = Audio2Vector(input_dir, './foa_wts.pkl')
    train_idx = joblib.load('./train_idx.pkl')
    val_idx = joblib.load('./val_idx.pkl')

    batch_size = 1
    lr = 1e-3
    num_cls = 11

    train_loader = DataLoader(ds, batch_size, sampler=train_idx)
    val_loader = DataLoader(ds, batch_size, sampler=val_idx)

    sed_criterion = nn.MultiLabelSoftMarginLoss()
    doa_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr)

    with tqdm(train_loader, total=len(train_loader), desc="Train ") as iterator:
        for X, y in iterator:
            sed_y, doa_y = y[:, :, :num_cls], y[:, :, num_cls:]
            X, y = X.to(device, dtype=torch.float64), y.to(device, dtype=torch.float64)

            sed_output, doa_output = model(X)
            sed_loss = sed_criterion(sed_output, sed_y)
            doa_loss = doa_criterion(doa_output, doa_y)
            loss = sed_loss + doa_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            log = f'loss: {loss}'
            iterator.set_postfix_str(log)


if __name__ == "__main__":
    main()
