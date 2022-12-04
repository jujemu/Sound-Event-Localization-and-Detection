import click
import os
import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import utils
from models import Network

@click.command()
def main(   
    weight_output,

):  
    os.makedirs(weight_output, exist_ok=True)
    device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

    model = Network()
    model = model.to(device)



if __name__ == "__main__":
    main()
