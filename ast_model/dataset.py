from torch.utils.data import Dataset
from transformers import PretrainedConfig
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import torch
from sklearn.model_selection import StratifiedGroupKFold
from pedalboard import Pedalboard, PitchShift, time_stretch
import soxr
import pandas as pd
import soundfile as sf
import os
from os import listdir
from os.path import isfile, join
import torchaudio
import random
from sklearn.model_selection import GroupShuffleSplit
from beat_this.preprocessing import LogMelSpect #load_audio

class SpectrogramDataset(Dataset):
    def __init__(self, path, labels, label_names_set, transform = None, augment = False, state= "train", overlap = 20):
        self.paths = path
        self.labels = labels
        self.max_time = 1020  # in order to match the input dimension
        self.transform = transform
        self.step  = self.max_time - overlap
        self.augment = augment
        self.state = state
        self.labels_names_set = label_names_set # storing the genre names 
        self.num_crops  = 2
        self.overlap    = overlap
        
    def __len__(self):
        if self.state == "train":
            return len(self.paths)*self.num_crops
        else:
            return len(self.paths)

    def __getitem__(self, idx):
        #spec = self.spectrograms[idx]  # shape: (128, time)
        audio_idx = idx // self.num_crops if self.state == "train" else idx
        crop_idx  = idx % self.num_crops

        if self.transform:
            spec =self.transform(self.paths[audio_idx], self.augment)
            if spec is  None:
                 return self.__getitem__((idx + 1) % len(self))
        else:
            spec = self.data[self.paths[audio_idx]]
        if spec.ndim == 3 and spec.shape[0] == 1:
            spec = spec.squeeze(0)
        T = spec.shape[0]
        if T < self.max_time:
            pad = self.max_time - T
            spec = np.pad(spec, ((0, pad), (0, 0)), mode="constant")
        if self.state == "train":
            # randomly choose where to crop the fixed length during training 
            max_start = spec.shape[0] - self.max_time
            desired   = crop_idx * self.step
            start     = min(desired, max_start)
            spec = spec[start : start + self.max_time]

            
        else:
            # always take the center piece during validation and testing
            start = (spec.shape[0] - self.max_time) // 2
        #spec = spec[start: start + self.max_time, :]

        spec = spec[:self.max_time, :]
        #spec = spec.float()
        #spec = torch.tensor(spec, dtype=torch.float32)
        if isinstance(spec, np.ndarray):
            spec = torch.from_numpy(spec).float()
        else:
            spec = spec.detach().clone().float()
        #spec = spec.unsqueeze(0)
        label = self.labels[audio_idx]
        return {"input_values": spec, "labels": int(label)}
    

class Augment:
    def __init__(self, out_sr = 22050, aug_sr = 44100, mel_params = None, augm_params = None, augment_prob = 0.3):
        self.out_sr= out_sr
        self.aug_sr=aug_sr            
        self.augment_prob = augment_prob
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
                noise = 3
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
        if augment and random.random() < self.augment_prob:
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
    
class ArtistSplit:
    def __init__(self, root, dataset_csv):
        self.root = root
        if not  os.path.isfile(dataset_csv):
            df = self.create_dataset_df(save_path=dataset_csv)
        self.dataset_csv = dataset_csv
    
    def get_artist(self,filename):
        file_path = os.path.join(self.root, filename)
        if "___" in filename:
            artist = filename.split("___")[0]
        elif "-" in filename:
            artist = filename.split("-")[0]
        else:
            raise ValueError(f"Can not extract artist from the file {file_path}")
        return artist
    
    def create_dataset_df(self, save_path = None):
        header = ["title",  "genre", "artist", "filepath"]
        dataset_table =dict((el,[]) for el in header)
        subfolders = [ f.path for f in os.scandir(self.root) if f.is_dir() ]  
        for folder in subfolders:
            genre = os.path.basename(folder)
            for f in listdir(folder) :
                file = join(folder, f)
                if isfile(file):
                    dataset_table["title"].append(f)
                    dataset_table["filepath"].append(file)
                    dataset_table["genre"].append(genre)
                    artist = self.get_artist(f)
                    dataset_table["artist"].append(artist)

        dataset_df = pd.DataFrame.from_dict(dataset_table)
        if save_path:
            dataset_df.to_csv(save_path)
        return dataset_df
    
    def get_labels(self):
        df = pd.read_csv(self.dataset_csv)
        labels = list(df.genre.values)
        return labels
    def balanced_group_split(self,df, test_size, val_size,
                         group_col='artist',
                         class_col='genre',
                         tol=0.01,
                         seed=42,
                         max_tries=1000):
        """
        Returns train, val, test indices so that:
        - no artist overlaps splits
        - each split’s genre distribution is within tol of overall
        """
        rng = np.random.RandomState(seed)
        # always have the same seed for train-test split
        gss   = GroupShuffleSplit(n_splits=max_tries,
                                test_size=test_size,
                                random_state=42)

        genres = df[class_col].value_counts(normalize=True)
        for trainval_idx, test_idx in gss.split(df, groups=df[group_col]):
            # check test balance
            test_dist = df.iloc[test_idx][class_col].value_counts(normalize=True)
            if np.max(np.abs(test_dist.reindex(genres.index, fill_value=0) - genres)) > tol:
                continue

            # now val split within trainval
            gss_val = GroupShuffleSplit(n_splits=max_tries,
                                        test_size=val_size/(1-test_size),
                                        random_state=seed)
                                                    # multiple splits
            df_test = df.iloc[test_idx]
            df_trainval = df.iloc[trainval_idx]
            for train_idx, val_idx in gss_val.split(df_trainval, 
                                                    groups=df_trainval[group_col]):
                val_dist = df_trainval.iloc[val_idx][class_col] \
                            .value_counts(normalize=True)
                if np.max(np.abs(val_dist.reindex(genres.index, fill_value=0) - genres)) <= tol:
                    # success!
                    # map back to original indices
                    train_idx = trainval_idx[np.array(train_idx)]
                    val_idx   = trainval_idx[np.array(val_idx)]
                    df_train = df.iloc[train_idx]
                    df_val= df.iloc[val_idx]
                    return df_train, df_val, df_test

        raise RuntimeError("Could not find a balanced grouping within tol")
    def get_distributions_for_split(self, df_split, genres, split):
        genre_counts = {}
        for genre in genres:
            size = len(df_split[df_split["genre"] ==genre])
            genre_counts[genre] = size
        print(f"genre distributions for {split} split: {genre_counts}")
    def create_splits(self, val_size = 0.2, test_size = 0.1, seed = 42):        # the main function here that does the job
        df = pd.read_csv(self.dataset_csv)
        #getting splits with non-intersecting artists balanced as possible
        df_train, df_val, df_test = self.balanced_group_split(df = df, test_size = test_size, val_size = val_size, seed = seed,
                         group_col='artist',
                         class_col='genre',
                         tol=0.018,
                        max_tries=100000)
        genres = set(df["genre"].unique()) 
        self.get_distributions_for_split(df_val, genres, "validation")
        self.get_distributions_for_split(df_test, genres, "test")
        artist_train =set(df_train["artist"])
        artist_val = set(df_val["artist"])
        artist_test = set(df_test["artist"])
        intersection =bool (artist_test & artist_train or artist_train & artist_val or artist_val & artist_test) 
        assert not intersection, "Artists shouldn't intersect"
        # now we only return arrays that contain file paths and corresponding labels
        result =  []
        for df_split in [df_train, df_val, df_test]:
            result.extend([df_split["filepath"].tolist(), df_split["genre"].tolist()])
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = result
        return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels
    
    
        