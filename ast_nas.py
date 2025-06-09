from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
import torch.nn.functional as F
import random
from os import listdir
from os.path import isfile, join
from transformers import PretrainedConfig
from torch.utils.data import DataLoader
from transformers import ASTConfig
from sklearn.preprocessing import LabelEncoder
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import ASTModel
from transformers import DefaultDataCollator
import evaluate
from transformers import Trainer, TrainingArguments
import os
from transformers import EarlyStoppingCallback
from torchinfo import summary
import optuna
import gc
from pedalboard import Pedalboard, PitchShift, time_stretch
from beat_this.preprocessing import LogMelSpect, load_audio
#from ray.train import Sca
import pandas as pd
#from ray.tune.integration.wandb import WandbLogger
from typing import Dict, List, Any
import wandb
import warnings
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="`resume_download` is deprecated")
# import os
# os.environ["WANDB_MODE"] = "offline"
#data_path = Path(r"P:\datasets\beat-this\data\audio\spectograms_npz\gtzan.npz")
W_PC = False

metric = evaluate.load("accuracy")
data_collator = DefaultDataCollator()
ast_base = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
def get_spectrograms (data_path = None):
    data_path = data_path if data_path is not None else Path(r"C:\Users\Kochana\projects\genres\data\gtzan\gtzan.npz")
    data = np.load(data_path)
    lst = data.files
    tracks_path = []
    labels = []
    for path in lst:
        #path_file = os.path.join(data_path, file, "track.npy")
        tracks_path.append(path)
        labels.append(path[6:][:-12])
    return data, tracks_path, labels

def get_audio(data_path = None):
    labels = []
    track_paths = []
    data_path = data_path if data_path is not None else r"C:\Users\Kochana\projects\genres\data\gtzan_old\gtzan_old"
    subfolders = [ f.path for f in  os.scandir(data_path) if f.is_dir() ]
    for dir in subfolders:
        onlyfiles = [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]
        labels.extend([dir.split("\\")[-1]]*len(onlyfiles))
        track_paths += onlyfiles
    return data_path, track_paths, labels



class ASTGenreConfig(ASTConfig):
    model_type= "ast-genre_classification"
    def __init__(self, **kwargs):   # **kwargs: arbitrary number of key words arguments
        super().__init__(**kwargs)
        self.num_labels =10
        self.num_layers_top = kwargs.get("num_layers_top", 2)
        self.dropouts = kwargs.get("dropouts", [0.2]*self.num_layers_top)
        self.conv_dim = kwargs.get("conv_dim", [64]*self.num_layers_top)
        self.gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", 8)
        self.learning_rate = kwargs.get("learning_rate",3e-5 )
        self.freeze_layers = kwargs.get("freeze_layers", None)
        self.dropout = kwargs.get("dropout", 0)

class GTZANSpectrogramDataset(Dataset):
    def __init__(self, path, labels, transform = None, augment = False):
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
        else:
            spec = data[self.paths[idx]]
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
    def __init__(self, audio_sr = 22050, aug_sr = 44100, mel_params = None, augm_params = None):
        self.audio_sr = audio_sr
        self.aug_sr=aug_sr            
        
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
        self.logspect_class = LogMelSpect(audio_sr, **self.mel_params)
        
    
    def __call__(self, audio_path, augment = False, cut = False):
        waveform, sr = load_audio(audio_path)
        assert (
                    sr == self.audio_sr
                ), f"Sample rate mismatch: {sr} != {self.audio_sr}"
        
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
        spec = self.logspect_class(torch.tensor(augmented, dtype=torch.float32))
        return spec



class ASTForGenreClassification(PreTrainedModel):
    config_class = ASTGenreConfig

    def __init__(self, config, ast_model=ast_base):
        super().__init__(config)
        self.ast = ast_model
        self.dropout = nn.Dropout(config.dropout)
        self.activation_fn = self.get_activation(config.activation_fn)
        self.freeze_layers = config.freeze_layers
        self.classifier = nn.Linear(768, config.num_labels)
        if self.freeze_layers:
            print(f"freezing {self.freeze_layers} layers")
            for i, layer in enumerate(self.ast.encoder.layer):
                if i < self.freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        

    def get_activation(self, activation_fn) -> nn.Module:
        acrivation_mapping = {
        "relu":  nn.ReLU(),
        "tanh":  nn.Tanh(),
        "gelu":  nn.GELU(),
        "none":  nn.Identity(),
    }
        return acrivation_mapping.get(activation_fn)
    def get_normalisation(self, norm_type, dim) -> nn.Module:
        norm_mapping = {
        "batch": lambda d: nn.BatchNorm1d(d),
        "layer": lambda d: nn.LayerNorm(d),
        "none": lambda d: nn.Identity(),
    }
        return norm_mapping.get(norm_type, lambda d: nn.Identity())(dim)
    def forward(self, input_values, labels=None):
        x = self.ast.embeddings(input_values)
        x = self.ast.encoder(x).last_hidden_state
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.activation_fn(x)
        logits = self.classifier(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
        return SequenceClassifierOutput(loss=loss, logits=logits)
    
def objective(trial):
    if trial is not None:
        config = ASTGenreConfig(
                                activation_fn = trial.suggest_categorical("nonlinearity", ["relu", "gelu", "tanh", "none"]),
                                dropout = trial.suggest_int(f"drop_out", 0, 4)/ 10,
                                gradient_accumulation_steps = trial.suggest_int(f"gradient_accumulation_steps", 2, 16, step = 2),
                                learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log = True) ,
                                freeze_layers =trial.suggest_int("freeze_layers", 0, 8)
                                )
        wandb.init(project="ast_model", name=f"w_PC_freeze{trial.number}", config=
                    {
                        "activation_fn" : config.activation_fn,
                        "dropout" : config.dropout, 
                        "freeze_layers" :  config.freeze_layers

                    })
    else:   
        config = ASTGenreConfig()
    model = ASTForGenreClassification(config=config, ast_model=ast_base)
    if trial:
        #print(f"hyperparameters chosen: num_layers = {num_layers_top}")
        summary(model)
    training_args = TrainingArguments(
    output_dir="./ast-gtzan_w_pc",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=3e-5,
    dataloader_num_workers=8,
    num_train_epochs=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=W_PC,
    gradient_accumulation_steps=8,
    greater_is_better=True,
    report_to=["wandb"],
    push_to_hub=False,
    #hub_model_id="polinaZaroko/ast_try_again",
    hub_strategy="checkpoint",
    save_total_limit=2,
    warmup_ratio=0.1  #proportion of training to be dedicated to a linear warmup where learning rate gradually increases.   
)    
    trainer = Trainer(
    model = model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=None,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=8)],
)
    trainer.train()
    eval_result = trainer.evaluate()
    wandb.log({"eval_accuracy": eval_result["eval_accuracy"]})
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()
    wandb.finish()
    return eval_result["eval_accuracy"]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return metric.compute(predictions=predictions, references=labels)

if __name__ == "__main__":    
    data, tracks_path, labels  = get_audio()
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    train, test, train_labels, test_labels = train_test_split(
            tracks_path, encoded_labels, test_size=0.1, stratify=encoded_labels, random_state=42)
    train, validation, train_labels, validation_labels = train_test_split(
            train, train_labels, test_size=0.2, stratify=train_labels, random_state=42)
    transform = Augment()
    train_dataset = GTZANSpectrogramDataset(train, train_labels,transform = transform, augment = True)
    val_dataset = GTZANSpectrogramDataset(validation, validation_labels, transform = transform, augment = False)

    study = optuna.create_study(direction= "maximize")
    study.optimize(objective, n_trials = 10)
    best_trial = study.best_trial
    print("Best hyperparameters:", study.best_params)
    df = pd.DataFrame(study.best_params, index = ['i',])
    df.to_csv("best_hyp.csv")