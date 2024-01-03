from datetime import datetime
import os 

import click
import joblib

import torch
from torch.utils.data import DataLoader

from models import Network
from utils import Audio2Vector

@click.command()
@click.option('--aud_dir', 'aud_dir', help='Directory of audio datasets')
@click.option('--aud_path', 'aud_path', help='File path of audio')
@click.option('--model_weight', 'model_weight', help='trained model weight path')
@click.option('--is_sample', 'is_sample', type=bool, help='if you want to put in one audio sample, pass True in')
def infer(
    aud_dir,
    aud_path,
    model_weight,
    is_sample=True
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device is ready on {device}')
    model = Network().to(device)
    model.load_state_dict(torch.load(model_weight))

    if is_sample:
        """
            Ex> 
            python3 Code/pytorch/infer.py --aud_path Code/pytorch/sample/split1_ir0_ov1_1.wav --model_weight checkpoint/2022-12-11_0335/02_model_weight.pkl --is_sample True
        """
        assert aud_path, "when 'is_sample' is True, put in aud_path"
        assert os.path.isfile(aud_path), "aud_path is not a path of directory, but wave file"

        sample_ds = Audio2Vector(sample_path=aud_path, is_sample=True)
        sample_loader = DataLoader(sample_ds, 1)
        for sample in sample_loader:
            sample = sample.to(device)
            result, _ = model(sample)
            result = result.squeeze().cpu().detach().numpy()
        output_path = './sample_result.pkl'
        print(f'inference of sample is dumped in {output_path}')
        joblib.dump(result, output_path)


if __name__=="__main__":
    infer()