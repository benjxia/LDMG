from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchaudio
import torchaudio_augmentations as ta

import lightning as L

import pandas as pd
import os

from ldm.modules.vae_gan import DEFAULT_AUDIO_DUR, DEFAULT_INPUT_SR


# This is unused
class GTZANAudioDataset(Dataset):
    def __init__(self, path: str, sample_rate=16000, duration=DEFAULT_AUDIO_DUR):
        """
        Expects a path to .../Data - i.e. the path should end with "Data"
        """
        super().__init__()
        self.path = path
        if not os.path.exists(path):
            import requests
            import zipfile
            # Stream the download to avoid loading the whole file into memory
            with requests.get('https://www.kaggle.com/api/v1/datasets/download/andradaolteanu/gtzan-dataset-music-genre-classification', stream=True) as r:
                r.raise_for_status()  # Raise an error on bad status
                with open('gtzan.zip', 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            # TODO: theres a path bug here, im too lazy to fix it, but it unzips the Data one level too deep than it should
            # Ensure the extract_to directory exists
            os.makedirs(path, exist_ok=True)

            # Extract the ZIP file
            with zipfile.ZipFile('gtzan.zip', 'r') as zip_ref:
                zip_ref.extractall(path)

        self.df = pd.read_csv(os.path.join(path, 'features_30_sec.csv'))
        self.min_audio_len = self.df['length'].min()
        self.sample_rate = sample_rate

        # Filter malformed shit
        self.df = self.df[self.df['filename'] != 'jazz.00054.wav']
        self.resampler = torchaudio.transforms.Resample(22050, sample_rate)

        self.sr = sample_rate
        self.target_frames = self.sr * duration

        self.crop = ta.RandomResizedCrop(self.target_frames)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        genre = self.df.iloc[index]['label']
        filename = self.df.iloc[index]['filename']
        path = os.path.join(self.path, 'genres_original', genre, filename)
        waveform, sr = torchaudio.load(path, normalize=True)

        waveform = waveform.mean(dim=0, keepdim=True)  # Result: shape [1, time]

        if sr != self.sr:
            waveform = self.resampler(waveform)

        if waveform.size(1) < self.target_frames:
            pad_len = self.target_frames - waveform.size(1)
            waveform = F.pad(waveform, (0, pad_len))
        else:
            waveform = self.crop(waveform)

        return waveform
