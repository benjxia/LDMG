from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader

import torchaudio
import torchaudio_augmentations as ta

import lightning as L

import os

from ldm.modules import DEFAULT_INPUT_SR, DEFAULT_AUDIO_DUR

class FmaMediumDataset(Dataset):
    def __init__(self, root_dir, sample_rate=DEFAULT_INPUT_SR, duration=DEFAULT_AUDIO_DUR):
        """
        Args:
            root_dir (str): Path to the fma_medium directory.
            sample_rate (int): Target sample rate.
            duration (int): Duration of each sample in seconds.
        """
        self.sample_rate = sample_rate
        self.num_samples = sample_rate * duration
        self.audio_paths = []

        # Collect all MP3 files recursively
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".mp3"):
                    self.audio_paths.append(os.path.join(root, file))

        self.target_frames = sample_rate * duration
        self.crop = ta.RandomResizedCrop(self.target_frames)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        try:
            waveform, sr = torchaudio.load(path)
        except RuntimeError:
            # Cus theres a few malformed waveforms in fma medium, pray the collate function can handle this case
            return None
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to target sample rate
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Pad or crop to fixed number of samples
        if waveform.shape[1] < self.num_samples:
            pad_amt = self.num_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amt))
        else:
            waveform = self.crop(waveform)

        return waveform

class MusicDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        batch_size: int = 1,
        num_workers: int = 1,
        target_sr: int = 16000,
        clip_duration: float = DEFAULT_AUDIO_DUR,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_sr = target_sr
        self.clip_duration = clip_duration

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = FmaMediumDataset(
            self.data_dir,
            self.target_sr,
            self.clip_duration
        )

    def _collate_fn(self, batch):
        clips = []
        for clip in batch:
            if clip is None:
                continue
            clips.append(clip)
        return torch.stack(clips)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self._collate_fn
        )
