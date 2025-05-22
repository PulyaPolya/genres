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
from torchinfo import summary
#from ray.train import Sca
import pandas as pd
#from ray.tune.integration.wandb import WandbLogger
from typing import Dict, List, Any
import wandb

#data_path = Path(r"P:\datasets\beat-this\data\audio\spectograms_npz\gtzan.npz")
data_path = Path(r"C:\Users\Kochana\projects\genres\data\gtzan\gtzan.npz")
data = np.load(data_path)
lst = data.files
tracks_path = []
labels = []
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

class ASTGenreConfig(ASTConfig):
    model_type= "ast-genre_classification"
    def __init__(self, **kwargs):   # **kwargs: arbitrary number of key words arguments
        super().__init__(**kwargs)
        self.num_labels =10
        self.num_layers_top = kwargs.get("num_layers_top", 2)
        self.dropouts = kwargs.get("dropouts", [0.2]*self.num_layers_top)
        self.conv_dim = kwargs.get("conv_dim", [64]*self.num_layers_top)

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
ast_base = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

class ASTForGenreClassification(PreTrainedModel):
    config_class = ASTGenreConfig

    def __init__(self, config, ast_model=ast_base):
        super().__init__(config)
        self.ast = ast_model
        
        self.num_layers_top = config.num_layers_top
        self.dropout = nn.Dropout(0.2)
        self.dropouts = config.dropouts
        self.conv_dim = config.conv_dim
        self.activation_fn = nn.ReLU()
        layers = [nn.Conv1d(768, self.conv_dim[0], kernel_size=3, padding="same")]
        for i in range(self.num_layers_top-1):
            layers.append( nn.Conv1d(self.conv_dim[i], self.conv_dim[i+1], kernel_size=3, padding=1))
        self.conv_layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(self.conv_dim[-1], config.num_labels)

    def forward(self, input_values, labels=None):
        x = self.ast.embeddings(input_values)
        x = self.ast.encoder(x).last_hidden_state
         # or use x[:, 0, :]
        x = x.transpose(1,2)
        for layer in self.conv_layers:
            x = self.activation_fn(layer(x))
        x = x.mean(dim=2) 
        x = self.dropout(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)

        return SequenceClassifierOutput(loss=loss, logits=logits)
    
def model_init(trial= None):
    if trial is not None:
        MAX_LAYERS = 4
        num_layers = trial.suggest_int("num_layers", 1,MAX_LAYERS)
        conv_dim = []
        dropouts = []
        for i in range (MAX_LAYERS):
            dropouts.append((trial.suggest_int(f"drop_out{i}", 0, 3))/ 10)
            conv_dim.append(trial.suggest_categorical(f"dim_{i}", [64, 128, 256]) )
        conv_dim = conv_dim[:num_layers -1]
        conv_dim.append(trial.suggest_categorical(f"dim_last", [32, 16]) )
        print("chose number of layers")
        print(f"num_layers: {num_layers}, conv_dim: {conv_dim[:num_layers]}")
        config = ASTGenreConfig(num_layers_top = num_layers,
                                dropouts= sorted(dropouts[:num_layers], reverse=True),
                                conv_dim =sorted(conv_dim, reverse=True)
                                )
        wandb.init(project="ast_model", name="0wpc_try", config=config
        )
    else:   
        config = ASTGenreConfig()
    model = ASTForGenreClassification(config=config, ast_model=ast_base)
    if trial:
        print(f"hyperparameters chosen: num_layers = {num_layers}")
        summary(model)
    return model

# def hp_space(trial):
#     return {
#         "learning_rate": trial.suggest_float("learning_rate", 3e-5, 3e-4, log=True),
#         #"num_train_epochs": trial.suggest_categorical( "num_train_epochs",[2, 3, 4])
#     }
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
    num_train_epochs=3,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
    gradient_accumulation_steps=8,
    greater_is_better=True,
    report_to=None,
    push_to_hub=False,
    #hub_model_id="polinaZaroko/ast_try_again",
    hub_strategy="checkpoint",
    save_total_limit=2,
    warmup_ratio=0.1  #proportion of training to be dedicated to a linear warmup where learning rate gradually increases.   
)
train_dataset = GTZANSpectrogramDataset(train, train_labels)
val_dataset = GTZANSpectrogramDataset(validation, validation_labels)

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
    n_trials=2,
   #hp_space=hp_space,
)
print(f"best hyperparameters: {best_run.hyperparameters}")