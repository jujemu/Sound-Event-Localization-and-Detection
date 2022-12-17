from glob import glob
import os

import click
import joblib
import numpy as np
from sklearn import preprocessing
from tqdm import tqdm

import utils


@click.command()
def standard_scaler(
    train_ds_path,
    output_path
):
    scaler = preprocessing.StandardScaler()
    train_ds = glob(os.path.join(train_ds_path, '*'))
    with tqdm(enumerate(train_ds), total=len(train_ds), desc='Standard Scaling ') as iterator:
        for file_cnt, file_name in iterator:
            log = '{}: {}'.format(file_cnt, file_name)
            iterator.set_postfix_str(log)

            feat_file = np.load(os.path.join(self._feat_dir, file_name))
            scaler.partial_fit(np.concatenate((np.abs(feat_file), np.angle(feat_file)), axis=1))
            del feat_file
    joblib.dump(scaler, output_path)
    print('Normalized_features_wts_file: {}. Saved.'.format(normalized_features_wts_file))


if __name__ == "__main__":
    standard_scaler()