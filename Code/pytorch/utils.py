from glob import glob
import os

import joblib
import librosa
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
from tqdm import tqdm

from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader


class Audio2Vector(Dataset):
    def __init__(
        self,
        dataset_dir,
        scaler_path='./feat_label/foa_wts.pkl',
        is_eval=False
    ):
        super().__init__()
        self.aud_dir = os.path.join(dataset_dir, 'foa_dev')
        self.aud_arr = sorted(os.listdir(self.aud_dir))
        self.desc_dir = None if is_eval else os.path.join(dataset_dir, 'metadata_dev')
        self.desc_arr = sorted(os.listdir(self.desc_dir)) if self.desc_dir else None
        self.scaler = joblib.load(scaler_path) if scaler_path else None

        self._fs = 48000
        self._hop_len_s = 0.02
        self._hop_len = int(self._fs * self._hop_len_s)
        self._frame_res = self._fs / float(self._hop_len)
        self._nb_frames_1s = int(self._frame_res)
        self._audio_max_len_samples = 60 * self._fs
        self._max_frames = int(np.ceil(self._audio_max_len_samples / float(self._hop_len)))

        self._win_len = 2 * self._hop_len
        self._nfft = self._next_greater_power_of_2(self._win_len)

        self._eps = np.spacing(np.float64(1e-16))
        self._nb_channels = 4

        self.cls2idx = {
            'clearthroat': 2,
            'cough': 8,
            'doorslam': 9,
            'drawer': 1,
            'keyboard': 6,
            'keysDrop': 4,
            'knock': 0,
            'laughter': 10,
            'pageturn': 7,
            'phone': 3,
            'speech': 5
        }
        self.idx_to_classes = {str(val): key for key, val in self._cls2idx.items()}

    def __len__(self):
        return len(self.aud_arr)
    
    def __getitem__(self, index):
        # Audio to Feature vector
        aud_path = os.path.join(self.aud_dir, self.aud_arr[index])
        aud = self._load_audio(aud_path)
        spectrogram = self._spectrogram(aud).reshape(self._max_frames, -1)

        if self.scaler:
            spectrogram = self.scaler.transform(np.abs(spectrogram))        
        spectrogram = spectrogram.reshape(-1, 1024, 4)

        if self.is_eval:
            return spectrogram

        # label to (frame x classes) array
        desc_path = os.path.join(self.desc_dir, self.desc_arr[index])
        desc = pd.read_csv(desc_path)

        # if audio has longer than 60s, cut its end to fit 60s
        desc = desc[desc.start_time <= 60.]
        desc.loc[:, 'end_time'] = desc.end_time.apply(lambda x: 60. if x>60. else x)
    
        # convert second into frames
        desc.start_time = np.round(desc.start_time * 50).astype(np.uint32)
        desc.end_time = np.round(desc.end_time * 50).astype(np.uint32)

        sed_label = np.zeros((3000, 11))
        azi_label = np.zeros((3000, 11))
        ele_label = np.ones((3000, 11)) * 50
        for _, row in desc.iterrows():
            cls, start, end, ele, azi, _ = row
            sed_label[start: end, self.cls2idx[cls]] = 1
            azi_label[start: end, self.cls2idx[cls]] = azi
            ele_label[start: end, self.cls2idx[cls]] = ele

        label = np.concatenate((sed_label, azi_label, ele_label), axis=1)
    
        return spectrogram, label
    
    @staticmethod
    def _next_greater_power_of_2(x):
        return 2 ** (x - 1).bit_length()

    ################### INPUT AUDIO FEATURES ###################
    # load audio and length of audio fits in 60second
    def _load_audio(self, audio_path):
        _, audio = wav.read(audio_path)
        audio = audio[:, :self._nb_channels] / 32768.0 + self._eps
        if audio.shape[0] < self._audio_max_len_samples:
            zero_pad = np.zeros((self._audio_max_len_samples - audio.shape[0], audio.shape[1]))
            audio = np.vstack((audio, zero_pad))
        elif audio.shape[0] > self._audio_max_len_samples:
            audio = audio[:self._audio_max_len_samples, :]
        return audio
    
    # short time fourier transform,
    # apply for stft to audio at each channels
    # output shape is (3000, 1024, 4)
    # frame, (n_fft/2), channels
    def _spectrogram(self, audio_input):
        _nb_ch = audio_input.shape[1]
        nb_bins = self._nfft // 2
        spectra = np.zeros((self._max_frames, nb_bins, _nb_ch), dtype=complex)
        for ch_cnt in range(_nb_ch):
            stft_ch = librosa.core.stft(audio_input[:, ch_cnt], n_fft=self._nfft, hop_length=self._hop_len,
                                        win_length=self._win_len, window='hann')
            spectra[:, :, ch_cnt] = stft_ch[1:, :self._max_frames].T
        return spectra
    
    def _extract_spectrogram_for_file(self, audio_filename):
        audio_in = self._load_audio(os.path.join(self._aud_dir, audio_filename))
        return self._spectrogram(audio_in)


def get_scaler(scaler_path='./feat_label/foa_wts.pkl'):
    ds_path = './Development Datasets'
    unnormalized_ds = Audio2Vector(ds_path, scaler_path=None, is_eval=True)

    scaler = preprocessing.StandardScaler()
    with tqdm(unnormalized_ds, desc='Get scaler ') as iterator:
        for X in iterator:
            X = X[0] if isinstance(X, tuple) else X
            scaler.partial_fit(np.abs(X))
    joblib.dump(scaler, scaler_path)
