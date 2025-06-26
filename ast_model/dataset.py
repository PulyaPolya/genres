from torch.utils.data import Dataset
from transformers import PretrainedConfig
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import torch
from pedalboard import Pedalboard, PitchShift, time_stretch
import soxr
import torchaudio
import random
from beat_this.preprocessing import LogMelSpect #load_audio
class SpectrogramDataset(Dataset):
    def __init__(self, path, data, labels, transform = None, augment = False):
        self.paths = path
        self.labels = labels
        self.max_time = 1020  # in order to match the input dimension
        self.data = data
        self.transform = transform
        self.augment = augment
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        #spec = self.spectrograms[idx]  # shape: (128, time)
        if self.transform:
            spec =self.transform(self.paths[idx], self.augment)
            if spec is  None:
                 return self.__getitem__((idx + 1) % len(self))
        else:
            spec = self.data[self.paths[idx]]
        if spec.ndim == 3 and spec.shape[0] == 1:
            spec = spec.squeeze(0)
        if spec.shape[0] > self.max_time:
            start = np.random.randint(0, spec.shape[0] - self.max_time)
            spec = spec[start: start + self.max_time, :]
        else:
            pad_len = self.max_time - spec.shape[0]
            spec = np.pad(spec, ((0, pad_len), (0,0)), mode = 'constant')
        spec = spec[:self.max_time, :]
        #spec = spec.float()
        spec = torch.tensor(spec, dtype=torch.float32)

        #spec = spec.unsqueeze(0)
        label = self.labels[idx]
        return {"input_values": spec, "labels": int(label)}
    

class Augment:
    def __init__(self, out_sr = 22050, aug_sr = 44100, mel_params = None, augm_params = None):
        self.out_sr= out_sr
        self.aug_sr=aug_sr            
        self.max_len = 30
        default_mel_params = dict(
                        n_fft=1024,
                        hop_length=441,
                        f_min=30,
                        f_max=11000,
                        n_mels=128,
                        mel_scale="slaney",
                        normalized="frame_length",
                        power=1
                    )
        default_augm_params = dict(
                time_stretch= (20,4),
                pitch_shift = (-5,6),
                noise = 5
            )
        self.mel_params = mel_params if mel_params is not None else default_mel_params
        self.augm_params = augm_params if augm_params is not None else default_augm_params
        self.logspect_class = LogMelSpect(self.out_sr, **self.mel_params)
        
    def load_audio(self,path, dtype="float64"):
        try:
            waveform, samplerate = torchaudio.load(path, channels_first=False)
            waveform = np.asanyarray(waveform.squeeze().numpy(), dtype=dtype)
            return waveform, samplerate
        except Exception:
            # in case torchaudio fails, try soundfile
            try:
                import soundfile as sf

                return sf.read(path, dtype=dtype)
            except Exception:
                raise RuntimeError(f'Could not load audio from "{path}".')
    def __call__(self, audio_path, augment = False, cut = False):
        try:
            waveform, sr = self.load_audio(audio_path)
            #waveform, sr = torchaudio.load(audio_path, channels_first=False)
        except Exception as e:
            print(f"[WARN] Skipping unreadable file: {audio_path}. Reason: {e}")
            return None
        assert (
                    sr == self.out_sr
                ), f"Sample rate mismatch: {sr} != {self.out_sr}"
        if augment and random.random() < 0.3:
            waveform = np.asarray(waveform, dtype=np.float32)
            transformation = random.choice(["noise", "stretch", "pitch"])
            if transformation == "noise":
                noise_random = np.random.normal(0, 1, size = waveform.shape)
                augmented = waveform + noise_random*self.augm_params["noise"] / 100
                augmented = np.clip(augmented, -1.0, 1.0)
            elif transformation == "stretch":
                time_stretch_params = self.augm_params["time_stretch"]
                stretch_range = (
                    -time_stretch_params [0],
                    time_stretch_params [0] + 1,
                    time_stretch_params [1] if len(time_stretch_params ) > 1 else 1,
                )
                stretch = random.randrange(stretch_range[0],stretch_range[1], stretch_range[2])
                augmented = time_stretch(
                input_audio=waveform,
                samplerate=self.aug_sr,
                stretch_factor=1 + stretch / 100,
                pitch_shift_in_semitones=0.0,
            ).squeeze()
            elif transformation == "pitch":
                pitch_shift = self.augm_params["pitch_shift"]
                shifts = (
                (pitch_shift[0], pitch_shift[1] + 1)
                if pitch_shift
                else [0]
                )
                shift = random.randrange(shifts[0], shifts[1])
                board = Pedalboard(
                [
                    PitchShift(semitones=shift),
                ]
                )
            # apply pedalboard
                augmented = board(waveform, self.aug_sr)
        else:
            augmented = waveform.copy()
        # if self.out_sr != sr:
        #     #print(f"preprocessing {audio_path}")
        #     augmented = soxr.resample(augmented, in_rate = sr, out_rate = self.out_sr)
        spec = self.logspect_class(torch.tensor(augmented, dtype=torch.float32))
        return spec