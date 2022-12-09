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
import torch_xla.core.xla_model as xm

from utils import Audio2Vector
from models import Network


# @click.command()
# @click.option('--input', 'input_dir', help='Directory of wave files and metadata', required=True)
def main(
    input_dir='./Development Datasets/',
    model_weight_path='./checkpoint/2022-12-05_0158/00_model_weight.pkl'
):  
    date_time = datetime.now().strftime("%Y-%m-%d_%H%M")
    weight_output_path = os.path.join('./checkpoint', date_time) + '/'
    os.makedirs(weight_output_path, exist_ok=True)

    # device = 'cpu'
    device = xm.xla_device()

    model = Network()
    model = model.to(device)
    if model_weight_path:
        print(f'Model is loaded with {model_weight_path}\n')
        model.load_state_dict(torch.load(model_weight_path))

    ds = Audio2Vector(input_dir, './foa_wts.pkl')
    train_idx = joblib.load('./train_idx.pkl')
    val_idx = joblib.load('./val_idx.pkl')

    num_cls = 11
    epochs = 100
    batch_size = 1
    lr = 1e-3
    patience = 3
    best_loss = np.inf
    bad_learning_cnt = 0

    train_loader = DataLoader(ds, batch_size, sampler=train_idx)
    val_loader = DataLoader(ds, batch_size, sampler=val_idx)

    sed_criterion = nn.MultiLabelSoftMarginLoss()
    doa_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr)

    print(f'Model weight is saved on {weight_output_path}')
    for epoch in range(1, epochs+1):
        mean_loss = 0.
        
        model.train()
        print(f'Epoch: {epoch}')
        with tqdm(train_loader, total=len(train_loader), desc="Train ") as iterator:
            for idx, (X, y) in enumerate(iterator, start=1):
                X, y = X.to(device), y.to(device)
                sed_y, doa_y = y[:, :, :num_cls], y[:, :, num_cls:]                                

                sed_output, doa_output = model(X)
                sed_loss = sed_criterion(sed_output, sed_y)
                doa_loss = doa_criterion(doa_output, doa_y)
                loss = sed_loss + doa_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                xm.mark_step()
                
                mean_loss += loss.item()
                log = f'loss: {mean_loss/idx:.3f}'  
                iterator.set_postfix_str(log)
        weight_output = weight_output_path + str(epoch).zfill(2) + '_' + 'model_weight.pkl'
        torch.save(model.state_dict(), weight_output)
        
        model.eval()
        with tqdm(val_loader, total=len(val_loader), desc="Validation ") as iterator:
            mean_loss = 0.
            for idx, (X, y) in enumerate(iterator, start=1):
                X, y = X.to(device), y.to(device)
                sed_y, doa_y = y[:, :, :num_cls], y[:, :, num_cls:]                                

                with torch.no_grad():
                    sed_output, doa_output = model(X)
                sed_loss = sed_criterion(sed_output, sed_y)
                doa_loss = doa_criterion(doa_output, doa_y)
                loss = sed_loss + doa_loss

                acc = ((sed_output > 0.5) == sed_y).to('cpu', dtype=torch.float64).sum().item()
                acc /= 3000*11
                mean_loss += loss.item()
                log = f'loss: {mean_loss/idx:.3f} sed accuracy: {acc*100:.2f}%' 
                iterator.set_postfix_str(log)
            
            # if validation loss is not decreased despite of learning,
            # stop train loop
            mean_loss /= len(val_loader)
            if best_loss > mean_loss:
                best_loss = mean_loss
                bad_learning_cnt = 0
            else:
                bad_learning_cnt += 1
                if bad_learning_cnt > patience:
                    print()
                    print(f'Train loop is stoped in epochs, [{epoch}/{epochs}], because validation loss is not decreased')
                    break
        print()


if __name__ == "__main__":
    main()
