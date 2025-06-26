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
from transformers import EarlyStoppingCallback, TrainerCallback, TrainerState, TrainerControl
from torchinfo import summary
import optuna
import gc
from pedalboard import Pedalboard, PitchShift, time_stretch
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from beat_this.preprocessing import LogMelSpect #load_audio
#from ray.train import Sca
import pandas as pd
#from ray.tune.integration.wandb import WandbLogger
from typing import Dict, List, Any
import wandb
import warnings
import logging
import json
import soxr
import torchaudio
from dataclasses import dataclass
from types import SimpleNamespace
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from dataset import Augment, SpectrogramDataset

logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="`resume_download` is deprecated")
import os
#os.environ["WANDB_MODE"] = "offline"
#W_PC = True
gpu = torch.cuda.is_available()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For full reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class Config:
    dataset_name : str
    RUN_NAS : bool = False              # run NAS or not
    audio_path: str | None = None
    spectrogram_path : str | None = None
    wandb_name : str | None = None
    optuna_name : str | None = None
    num_workers : int = 0               # num workers for data loader
    num_trials : int = 1                # num optuna trials
    num_epochs : int = 10




metric = evaluate.load("accuracy")
data_collator = DefaultDataCollator()
ast_base = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

def get_spectrograms (data_path):
    data_path = data_path 
    data = np.load(data_path)
    lst = data.files
    tracks_path = []
    labels = []
    for path in lst:
        #path_file = os.path.join(data_path, file, "track.npy")
        tracks_path.append(path)
        labels.append(path[6:][:-12])
    return data, tracks_path, labels

def get_audio(dataset_name, data_path = None):
    labels = []
    track_paths = []
    num_labels = 0
    #folders_to_ignore =['Childrenss','Religious', 'Comedy_and_Spoken_Word' ] if dataset_name == "1517" else []
    #ata_path = data_path if data_path is not None else r"C:\Users\Kochana\projects\genres\data\gtzan_old\gtzan_old"
    subfolders = [ f.path for f in  os.scandir(data_path) if f.is_dir() ]
    for dir in subfolders:
        label = dir.split("\\")[-1]
        #if label not in folders_to_ignore:
        onlyfiles = [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]
        labels.extend([label]*len(onlyfiles))
        track_paths += onlyfiles
        num_labels += 1
        #else:
            #print(f"skipping {label} genre")
    return data_path, track_paths, labels, num_labels



class ASTGenreConfig(ASTConfig):
    model_type= "ast-genre_classification"
    def __init__(self, **kwargs):   # **kwargs: arbitrary number of key words arguments
        super().__init__(**kwargs)
        self.num_labels =kwargs.get("num_labels", 10)
        self.dropouts = kwargs.get("dropouts", 0.2)
        self.learning_rate = kwargs.get("learning_rate",3e-5 )
        self.freeze_layers = kwargs.get("freeze_layers", None)
        self.dropout_top = kwargs.get("dropout_top", 0)


    




class ASTForGenreClassification(PreTrainedModel):
    config_class = ASTGenreConfig

    def __init__(self, config, ast_model=ast_base):
        super().__init__(config)
        self.ast = ast_model
        self.dropout = nn.Dropout(config.dropout_top)
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
       # "tanh":  nn.Tanh(),
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
        assert labels.max() < logits.shape[1], f"Invalid label {labels.max()} for {logits.shape[1]} classes"
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
        return SequenceClassifierOutput(loss=loss, logits=logits)

class EarlyStoppingBelowThresholdCallback (TrainerCallback):
    def __init__(self, threshold = 0.2, patience = 3):
        self.threshold = threshold
        self.patience = patience
        self.counter = 0
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs ):
        acc = metrics.get("eval_accuracy", None)
        if acc is not None:
            if acc < self.threshold:
                self.counter += 1
                print(f"acc {acc} is  below threshold {self.threshold} with counter {self.counter}")
                if self.counter >= self.patience:
                    control.should_epoch_stop = True
                    print(f"early stopping ")
        else:
            self.counter = 0
        return control

        
def objective(trial, train_dataset, val_dataset, params, label_encoder):
    if trial is not None:
        config = ASTGenreConfig(
                                num_labels = params.num_labels, 
                                activation_fn = trial.suggest_categorical("nonlinearity", ["relu", "gelu", "none"]),
                                dropout_top = trial.suggest_float(f"dropout_top", 0.0, 0.4),
                                #gradient_accumulation_steps = trial.suggest_int(f"gradient_accumulation_steps", 2, 16, step = 2),
                                learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log = True) ,
                                freeze_layers =trial.suggest_int("freeze_layers", 0, 8)
                                )
        if params.wandb_name:
            if params.RUN_NAS:
                name = f"trial_{trial.number}"
                group= params.wandb_name
            else:
                name = params.wandb_name
                group = None
            wandb.init(project="ast_model", name=name, group = group, config=
                        {
                            "activation_fn" : config.activation_fn,
                            "dropout" : config.dropout_top, 
                            "freeze_layers" :  config.freeze_layers,
                            "learning_rate" : config.learning_rate,
                            #"gradient_accumulation": config.gradient_accumulation_steps

                        })
    else:   
        config = ASTGenreConfig()
    model = ASTForGenreClassification(config=config, ast_model=ast_base)
    if trial:
        #print(f"hyperparameters chosen: num_layers = {num_layers_top}")
        hyperparams = {key : getattr(config, key) for key in ["num_labels", "activation_fn", "dropout_top", "learning_rate", "learning_rate", "freeze_layers"]}
        hyperparams_df = pd.DataFrame.from_dict([hyperparams])
        print("Chosen hyperparameters")
        print(hyperparams_df)
        summary(model)
    training_args = TrainingArguments(
    output_dir="./ast-gtzan_cluster",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=config.learning_rate,
    dataloader_num_workers=params.num_workers,
    num_train_epochs=params.num_epochs,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=gpu,
    gradient_accumulation_steps=8,
    greater_is_better=True,
    report_to = ["wandb"],
    push_to_hub=False,
    #hub_model_id="polinaZaroko/ast_try_again",
    hub_strategy="checkpoint",
    save_total_limit=2,
    seed = 42,
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
    callbacks=[EarlyStoppingCallback(early_stopping_patience=8),
               EarlyStoppingBelowThresholdCallback(threshold=0.3, patience=3)],
)
    trainer.train()
    eval_result = trainer.evaluate()
    
    predictions = trainer.predict(val_dataset)
    y_pred = predictions.predictions.argmax(axis = 1)
    y_true = predictions.label_ids
    labels = list(label_encoder.classes_)
    genre_names = [os.path.basename(label) for label in labels]
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=genre_names)
    disp.plot(ax=ax, xticks_rotation=90, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.pdf", bbox_inches="tight")
    plt.close()
    genre_names = list(label_encoder.classes_)
    if params.wandb_name:
        wandb.log({"eval_accuracy": eval_result["eval_accuracy"],
                    "confusion_matrix": wandb.Image("confusion_matrix.pdf")})
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()
    wandb.finish()
    return eval_result["eval_accuracy"]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return metric.compute(predictions=predictions, references=labels)

def main():
    print(torch.cuda.is_available())
    with open ("ast_training_params.json") as f:
        #config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
        config_dict = json.load(f) 
    config = Config(**config_dict)
    if config.audio_path:   
        data, tracks_path, labels, num_labels  = get_audio(config.dataset_name, config.audio_path)
    elif config.spectrogram_path:
        data, tracks_path, labels  = get_spectrograms(config.spectrogram_path)
    config.num_labels = num_labels 
    print(num_labels)
    #global W_PC
    #W_PC = config.W_PC
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    #amount = 1000
    #L = range(14)
    #y_pred = [random.choice(L) for _ in range(amount)]
    #y_true = [random.choice(L) for _ in range(amount)
    """
    labels = list(le.classes_)
    genre_names = [os.path.basename(label) for label in labels]
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=genre_names)
    disp.plot(ax=ax, xticks_rotation=90, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.pdf", bbox_inches="tight")
    plt.close()
    """
    #raise Exception("aaaa")
    train, test, train_labels, test_labels = train_test_split(
            tracks_path, encoded_labels, test_size=0.1, stratify=encoded_labels, random_state=42)
    train, validation, train_labels, validation_labels = train_test_split(
            train, train_labels, test_size=0.2, stratify=train_labels, random_state=42)
    
    transform = Augment()
    train_dataset = SpectrogramDataset(train,data, train_labels,transform = transform, augment = True)
    val_dataset = SpectrogramDataset(validation,data, validation_labels, transform = transform, augment = False)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=0)
    sampler = optuna.samplers.TPESampler(seed=42, 
                                         multivariate=True,
                                         warn_independent_sampling=False)
    study = optuna.create_study(study_name=config.optuna_name,
                                direction= "maximize",
                                sampler = sampler,
                                pruner = pruner,
                                storage = "sqlite:///optuna.db",
                                load_if_exists=True )
    #study.optimize(make_objective(config_params, train_dataset, valid_dataset, test_dataset), n_trials=config_params.num_trials)
    #study = optuna.create_study(direction= "maximize")
    study.optimize(lambda trial: objective(trial, train_dataset= train_dataset,  val_dataset = val_dataset, params = config, label_encoder = le), n_trials = config.num_trials)
    best_trial = study.best_trial
    print("Best hyperparameters:", study.best_params)
    df = pd.DataFrame(study.best_params, index = ['i',])
    df.to_csv("best_hyp.csv")
    
if __name__ == "__main__":
    main()