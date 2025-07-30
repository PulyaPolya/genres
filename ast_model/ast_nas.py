from transformers import PreTrainedModel, DefaultDataCollator, ASTModel, ASTConfig,  Trainer, TrainingArguments
from transformers import EarlyStoppingCallback, TrainerCallback, TrainerState, TrainerControl
import random
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import torch
from collections import Counter
import evaluate
import os
from torchinfo import summary
import optuna
import gc
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import warnings
import logging
import json
from dataclasses import dataclass
from transformers import AutoModelForSequenceClassification
from dataset import Augment, SpectrogramDataset, ArtistSplit
from model import ASTGenreConfig, ASTForGenreClassification
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="`resume_download` is deprecated")
#os.environ["WANDB_MODE"] = "offline"
gpu = torch.cuda.is_available()
best_eval_acc = 0.61
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
    dataset_table: str = "dataset_table.csv"   # path to store the dataset table
    data_path : str | None = None
    data_type : str | None = None
    wandb_name : str | None = None
    optuna_name : str | None = None
    num_workers : int = 0               # num workers for data loader
    num_trials : int = 1                # num optuna trials
    num_epochs : int = 10
    batch_size : int = 2
    hf_token : str | None = None
    hf_model_id : str | None = None


metric = evaluate.load("accuracy")
data_collator = DefaultDataCollator()


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

def get_audio(data_path = None):
    labels = []
    track_paths = []
    num_labels = 0
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



class EarlyStoppingBelowThresholdCallback (TrainerCallback):      # quickly discard models that result in very low accuracy, potentially due to bad 
    def __init__(self, threshold = 0.2, patience = 3):            # learning rate choice
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

def get_confusion_matrix(predictions, label_encoder,name):
    y_pred = predictions.predictions.argmax(axis = 1)
    y_true = predictions.label_ids
    labels = list(label_encoder.classes_)
    genre_names = [os.path.basename(label) for label in labels]
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=genre_names)
    disp.plot(ax=ax, xticks_rotation=90, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.savefig(f"cm_{name}.pdf", bbox_inches="tight")
    plt.close()
    genre_names = list(label_encoder.classes_)


def objective(trial, train_dataset, val_dataset,test_dataset,  params, label_encoder):
    if trial is not None:
        id2label = {i: label for i, label in enumerate(train_dataset.labels_names_set)}
        label2id = {label: i for i, label in enumerate(train_dataset.labels_names_set)}
       
        
        config = ASTGenreConfig(
                                num_labels = params.num_labels, 
                                activation_fn =trial.suggest_categorical("nonlinearity", ["relu", "gelu", "none"]),
                                normalisation =  trial.suggest_categorical("normalisation", [ "layer", "none"]),
                                batch_size =params.batch_size, 
                                dropout_top = trial.suggest_float(f"dropout_top", 0.0, 0.4),
                                learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log = True) ,
                                freeze_layers =trial.suggest_int("freeze_layers", 0, 8),
                                id2label=id2label,
                                label2id=label2id,
                                )
        if params.RUN_NAS:
                name = f"trial_{trial.number}"
                group= params.wandb_name
                seed = 42
        else:
                name = params.wandb_name
                group = None
                seed = random.randint(0, 2**31-1)
        set_seed(seed)
        if params.wandb_name:
            
            wandb.init(project="ast_model", name=name, group = group, config=
                        {
                            "activation_fn" : config.activation_fn,
                            "normalisation": config.normalisation, 
                            "dropout" : config.dropout_top, 
                            "freeze_layers" :  config.freeze_layers,
                            "learning_rate" : config.learning_rate,
                            "batch_size" : config.batch_size
                            #"gradient_accumulation": config.gradient_accumulation_steps

                        })
    else:   
        config = ASTGenreConfig()
    model = ASTForGenreClassification(config=config)
    if trial:
        #print(f"hyperparameters chosen: num_layers = {num_layers_top}")
        hyperparams = {key : getattr(config, key) for key in ["num_labels", "activation_fn", "dropout_top", "learning_rate", "learning_rate", "freeze_layers", "normalisation", "batch_size"]} #"normalisation"
        hyperparams_df = pd.DataFrame.from_dict([hyperparams])
        print("Chosen hyperparameters")
        print(hyperparams_df)
        summary(model)
    training_args = TrainingArguments(
    output_dir="./ast-merge",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=config.batch_size,
    per_device_eval_batch_size=config.batch_size,
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
    #hub_model_id=params.hf_model_id,
    #hub_strategy="end",  
    #hub_strategy="checkpoint", # pushes all models regardless of eval acc
    save_total_limit=1,
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
         #      EarlyStoppingBelowThresholdCallback(threshold=0.3, patience=3)
         ],
)
    print(f"trial{trial.number} before")
    print("Model hash before training:", hash(tuple(p.data_ptr() for p in model.parameters())))

    trainer.train()
    print("evaluating val")
    eval_result = trainer.evaluate()
    eval_accuracy = eval_result["eval_accuracy"]
    # print("evaluating test")
    # test_result = trainer.evaluate(eval_dataset=test_dataset)
    # print(test_result)
    # global best_eval_acc
    # if eval_accuracy > best_eval_acc:
    #     best_eval_acc = eval_accuracy
    #     trainer.save_model()       # writes best weights to ./ast-gtzan_cluster
    #     trainer.push_to_hub(       # pushes that directory
    #         commit_message="Upload best model at end of HPO",
    #         blocking=True         # wait until upload finishes
    #     )
    # getting confusion matrix 
    predictions = trainer.predict(val_dataset)
    get_confusion_matrix(predictions, label_encoder, name)
    if params.wandb_name:
        wandb.log({"eval_accuracy": eval_result["eval_accuracy"],
                  # "test_accuracy": test_result["eval_accuracy"],
                   "seed":seed
                    #"confusion_matrix": wandb.Image("confusion_matrix.pdf")
                    })
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()
    wandb.finish()

    return eval_accuracy
    


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
    #os.environ["HF_TOKEN"] = config.hf_token
    split_artists = ArtistSplit(config.data_path, config.dataset_table)
    labels = split_artists.get_labels()
    le = LabelEncoder()
    le.fit(labels)
    config.num_labels = len(le.classes_) 
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = split_artists.create_splits()
    train_labels_enc = le.transform(train_labels)
    validation_labels_enc = le.transform(val_labels) 
    test_labels_enc = le.transform(test_labels)   
    le = LabelEncoder()
    le.fit(labels)
    config.num_labels = len(le.classes_) 
    label_names_set = set(labels)
    # adding augmentation class applied to the training data
    transform = Augment( augment_prob = 0.5)
    train_dataset = SpectrogramDataset(train_paths, train_labels_enc, label_names_set = label_names_set,transform = transform, augment = True, state = "train")
    val_dataset = SpectrogramDataset(val_paths, validation_labels_enc, label_names_set= label_names_set, transform = transform, augment = False, state = "valid")
    test_dataset = SpectrogramDataset(test_paths, test_labels_enc, label_names_set= label_names_set, transform = transform, augment = False, state = "test")
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
    study.optimize(lambda trial: objective(trial, train_dataset= train_dataset,  val_dataset = val_dataset, test_dataset = test_dataset,
                                            params = config, label_encoder = le), n_trials = config.num_trials)
    print("Best hyperparameters:", study.best_params)
    df = pd.DataFrame(study.best_params, index = ['i',])
    df.to_csv("best_hyp.csv")
    
if __name__ == "__main__":
    # model =   ASTForGenreClassification.from_pretrained("./ast-gtzan_cluster/checkpoint-1575")
    # model.push_to_hub("PolinaKozarovytska/ast")
    main()