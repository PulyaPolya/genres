from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
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
from ray import tune
import ray
from ray.train.torch import TorchTrainer
#from ray.train import Sca
import pandas as pd
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.logger import DEFAULT_LOGGERS
#from ray.tune.integration.wandb import WandbLogger
from typing import Dict, List, Any


#os.environ["RAY_LOG_TO_STDERR"] = "1"
#os.environ["RAY_DISABLE_DASHBOARD"] = "1"
#ray.init(include_dashboard = False, num_gpus =0)
data_path = Path(r"P:\datasets\beat-this\data\audio\spectograms_npz\gtzan.npz")
data = np.load(data_path)
lst = data.files
tracks_path = []
labels = []
#data_path = Path(r"/content/drive/MyDrive/data/gtzan_old")
for path in lst:
    #path_file = os.path.join(data_path, file, "track.npy")
    tracks_path.append(path)
    labels.append(path[6:][:-12])
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)
train, test, train_labels, test_labels = train_test_split(
        tracks_path, encoded_labels, test_size=0.1, stratify=encoded_labels, random_state=42)
train, validation, train_labels, validation_labels = train_test_split(
        train, train_labels, test_size=0.2, stratify=train_labels, random_state=42)
class CustomTrainer(Trainer):

        def __init__(self, *args, **kwargs):
            super(CustomTrainer, self).__init__(*args, **kwargs)

        def _hp_search_setup(self, trial: Any):
            try:
                trial.pop('wandb', None)
            except AttributeError:
                pass
            super(CustomTrainer, self)._hp_search_setup(trial)

class ASTGenreConfig(ASTConfig):
    model_type= "ast-genre_classification"
    def __init__(self, num_labels = 10, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels

class GTZANSpectrogramDataset(Dataset):
    def __init__(self, path, labels):
        self.paths = path
        self.labels = labels
        self.max_time = 1020
        self.data = data
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        #spec = self.spectrograms[idx]  # shape: (128, time)
        spec = data[self.paths[idx]]
        if spec.ndim == 3 and spec.shape[0] == 1:
            spec = spec.squeeze(0)
        spec = spec[:self.max_time, :]
        spec = torch.tensor(spec, dtype=torch.float32)

        #spec = spec.unsqueeze(0)
        label = self.labels[idx]
        return {"input_values": spec, "labels": int(label)}
data_collator = DefaultDataCollator()
#metric = load_metric("accuracy")
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./ast-gtzan_w_pc",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=3e-5,
    num_train_epochs=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    #fp16=True,
    gradient_accumulation_steps=8,
    greater_is_better=True,
    report_to="wandb",
    push_to_hub=False,
    hub_model_id="polinaZaroko/ast_try_again",
    hub_strategy="checkpoint",
    save_total_limit=2,
    warmup_ratio=0.1  #proportion of training to be dedicated to a linear warmup where learning rate gradually increases.
     
)
train_dataset = GTZANSpectrogramDataset(train, train_labels)
val_dataset = GTZANSpectrogramDataset(validation, validation_labels)
ast_base = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

class ASTForGenreClassification(PreTrainedModel):
    config_class = ASTGenreConfig

    def __init__(self, config, ast_model=ast_base):
        super().__init__(config)
        self.ast = ast_model
        self.classifier = nn.Linear(768, config.num_labels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_values, labels=None):
        x = self.ast.embeddings(input_values)
        x = self.ast.encoder(x).last_hidden_state
        x = x.mean(dim=1)  # or use x[:, 0, :]
        x = self.dropout(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)

        return SequenceClassifierOutput(loss=loss, logits=logits)
    
def model_init():
    config = ASTGenreConfig(num_labels=10)
    ast_base = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    model = ASTForGenreClassification(config=config, ast_model=ast_base)
    return model

def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 3e-5, 3e-4, log=True),
        "num_train_epochs": trial.suggest_categorical( "num_train_epochs",[2, 3, 4])
    }
trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=None,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=8)],
)

best_run = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    n_trials=10,
    hp_space=hp_space,
)