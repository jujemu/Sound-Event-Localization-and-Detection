import os

import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
from tqdm import tqdm

from sklearn import preprocessing
from torch.utils.data import Dataset


cls2idx = {
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


class Audio2Vector(Dataset):
    def __init__(
        self,
        dataset_dir=None,
        sample_path=None,
        is_sample=False,
        scaler_path='./foa_wts.pkl',
        is_eval=False
    ):
        super().__init__()
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

        self.cls2idx = cls2idx
        self.idx_to_classes = {str(val): key for key, val in self.cls2idx.items()}

        self.is_eval = is_eval
        self.scaler = joblib.load(scaler_path) if scaler_path else None

        # for cross-validation,
        # save train_idx, validation_idx in pickle file
        if not os.path.isfile('./train_idx.pkl'):
            indices = np.random.permutation(len(self.aud_arr))
            val_size = indices.shape[0] // 5
            train_idx = indices[val_size:]
            val_idx = indices[:val_size]

            joblib.dump(train_idx, './train_idx.pkl')
            joblib.dump(val_idx, './val_idx.pkl')
            
        # Only one audio file is put in
        if is_sample:
            self.aud_path = sample_path
            self.is_sample = is_sample
            return
        self.aud_dir = os.path.join(dataset_dir, 'foa_dev')
        self.aud_arr = sorted(os.listdir(self.aud_dir))
        self.desc_dir = None if is_eval else os.path.join(dataset_dir, 'metadata_dev')
        self.desc_arr = sorted(os.listdir(self.desc_dir)) if self.desc_dir else None

    def __len__(self):
        return 1 if self.is_sample else len(self.aud_arr)

    def __getitem__(self, index):
        # Audio to Feature vector
        aud_path = self.aud_path if self.is_sample else os.path.join(self.aud_dir, self.aud_arr[index])
        aud = self._load_audio(aud_path)
        spectrogram = self._spectrogram(aud).reshape(self._max_frames, -1)

        if self.scaler:
            spectrogram = self.scaler.transform(np.abs(spectrogram))        
        spectrogram = spectrogram.reshape(-1, 1024, 4).transpose(2, 0, 1)

        if self.is_eval or self.is_sample:
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

        azi_label *= np.pi / 180
        ele_label *= np.pi / 50
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


def get_cls2idx():
    return cls2idx


def plot_aud_cls(df_path, infer):
    plt.style.use('seaborn-v0_8-whitegrid')
    cls2idx = get_cls2idx()
    # list sorted by index
    cls_arr = sorted([(key, val) for key, val in cls2idx.items()], key=lambda x: x[1])
    cls_arr = [item[0] for item in cls_arr]

    # csv파일(label)을 시간대 별 클래스를 넘파이 배열로 만들어
    # 이미지로 그립니다.
    SIZE = (3000, 11*100)
    df = preprocess_df(pd.read_csv(df_path))
    aud_arr = np.zeros(SIZE)
    for _, row in df.iterrows():
        cls, start, end, ele, azi, _ = row
        cls_index = cls2idx[cls]
        aud_arr[start: end, cls_index*100:(cls_index+1)*100] = 1
    
    if infer.ndim >= 3:
        infer = infer.squeeze()

    infer_arr = np.zeros(SIZE)
    for idx, _ in enumerate(cls2idx):
        infer_arr[:, idx*100:(idx+1)*100] = np.expand_dims(infer[:, idx], 1)

    _, ax = plt.subplots(2, 1, figsize=(10, 7))
    ax[0].set_title('Label')
    ax[0].imshow(aud_arr.T, cmap='gray')
    ax[0].set_yticks(range(50, 1150, 100), cls_arr)
    ax[1].set_title('Inference')
    ax[1].imshow(infer_arr.T, cmap='gray')
    ax[1].set_yticks(range(50, 1150, 100), cls_arr)
    plt.show()


def preprocess_df(df):
    df = df[df.start_time <= 60.]
    df.loc[:, 'end_time'] = df.end_time.apply(lambda x: 60. if x>60. else x)
    df.start_time = np.round(df.start_time * 50).astype(np.uint32)
    df.end_time = np.round(df.end_time * 50).astype(np.uint32)
    return df
