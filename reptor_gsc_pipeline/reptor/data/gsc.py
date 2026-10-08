from __future__ import annotations
import csv, random
from pathlib import Path
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

LABELS = ("yes","no","up","down","left","right","on","off","stop","go","_unknown_","_silence_")
LABEL_TO_INDEX = {x:i for i,x in enumerate(LABELS)}

class GSCDataset(Dataset):
    """Manifest-driven GSC v2 dataset for RepTor.

    Paper-specified: 16 kHz, 40 MFCC, 30 ms window, 10 ms stride.
    Reproduction choices (configurable): n_fft=512, n_mels=40,
    max time shift=100 ms, background-noise prob=0.8, max scale=0.1,
    SpecAugment frequency/time mask widths below.
    """
    def __init__(self, data_root, manifest_path, training=False, sample_rate=16000,
                 time_shift_ms=100.0, noise_prob=0.8, noise_max_scale=0.1,
                 specaugment=True, freq_mask_param=6, time_mask_param=10):
        self.data_root = Path(data_root)
        self.training = training
        self.sample_rate = sample_rate
        self.time_shift_samples = int(sample_rate*time_shift_ms/1000)
        self.noise_prob = noise_prob
        self.noise_max_scale = noise_max_scale
        self.specaugment = specaugment
        with Path(manifest_path).open("r", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        self.noise_files = sorted((self.data_root/"_background_noise_").glob("*.wav"))
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate, n_mfcc=40,
            melkwargs={"n_fft":512,"win_length":480,"hop_length":160,"n_mels":40,"center":True})
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)

    def __len__(self): return len(self.rows)

    def _load(self, row):
        if row["kind"] == "silence": return torch.zeros(1, self.sample_rate)
        x, sr = torchaudio.load(self.data_root/row["path"])
        if x.size(0)>1: x=x.mean(0, keepdim=True)
        if sr != self.sample_rate: x=torchaudio.functional.resample(x, sr, self.sample_rate)
        n=x.size(-1)
        if n<self.sample_rate: x=F.pad(x,(0,self.sample_rate-n))
        elif n>self.sample_rate: x=x[...,:self.sample_rate]
        return x

    def _shift(self, x):
        if self.time_shift_samples<=0: return x
        s=random.randint(-self.time_shift_samples,self.time_shift_samples)
        if s==0: return x
        if s>0: return F.pad(x,(s,0))[...,:self.sample_rate]
        s=abs(s); return F.pad(x,(0,s))[...,s:s+self.sample_rate]

    def _noise(self, x):
        if not self.noise_files or random.random()>=self.noise_prob: return x
        n, sr = torchaudio.load(random.choice(self.noise_files))
        if n.size(0)>1: n=n.mean(0, keepdim=True)
        if sr!=self.sample_rate: n=torchaudio.functional.resample(n,sr,self.sample_rate)
        if n.size(-1)<self.sample_rate:
            reps=(self.sample_rate+n.size(-1)-1)//n.size(-1); n=n.repeat(1,reps)
        m=n.size(-1)-self.sample_rate; start=random.randint(0,m) if m>0 else 0
        n=n[...,start:start+self.sample_rate]
        return (x+random.uniform(0.0,self.noise_max_scale)*n).clamp(-1,1)

    def __getitem__(self, idx):
        row=self.rows[idx]; x=self._load(row)
        if self.training: x=self._noise(self._shift(x))
        feat=self.mfcc(x).squeeze(0)
        if self.training and self.specaugment:
            feat=self.time_mask(self.freq_mask(feat))
        return feat, LABEL_TO_INDEX[row["label"]]
