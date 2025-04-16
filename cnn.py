import os
import random
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ------------------ Model Definition ------------------

class EarlyStopping:
    def __init__(self, patience = 5,min_delta = 0, mode = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode  # 'min' for loss, 'max' for accuracy
        self.best_model_state = None
    
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif((self.mode == 'min' and score < self.best_score - self.min_delta)
            or (self.mode == 'max' and score > self.best_score + self.min_delta)):
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            print(f" No improvement. EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


class MusicGenreCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(MusicGenreCNN, self).__init__()
        
        self.bn_0_freq = nn.BatchNorm2d(1)  # input: [B, 1, 300, 300]

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout(0.25)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(1024, num_classes)
        self.pool4 = nn.MaxPool2d(3)  # from 37x37 → 12x12
        self.drop4 = nn.Dropout(0.25)

        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.pool5 = nn.MaxPool2d(3)  # from 12x12 → 4x4
        self.drop5 = nn.Dropout(0.25)


    def forward(self, x):
    
        x = self.bn_0_freq(x)
        x = F.elu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        x = F.elu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        x = F.elu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.drop3(x)

        x = F.elu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.drop4(x)

        x = F.elu(self.bn5(self.conv5(x)))
        x = self.pool5(x)
        x = self.drop5(x)
        #print(x.shape) 
        x = x.view(x.size(0), -1)  
        #print(x.shape)     # Flatten to [B, 51200]
        #x = F.relu(self.fc1(x)) 
        #print(x.shape)      # [B, 256]
        x = self.fc2(x) 
        return x

# ------------------ Data Augmentation ------------------
def augment_audio(y, sr):
    #y = librosa.effects.pitch_shift(y, sr, n_steps=random.uniform(-1, 1))
    y = librosa.effects.pitch_shift(y, sr=sr, n_steps=random.choice([1,-1, 0]))
    #y = librosa.effects.time_stretch(y, rate=random.uniform(0.9, 1.1))
    noise = np.random.randn(len(y))
    #y = y + 0.03 * noise
    return y


# ------------------ Dataset ------------------
class GenreDataset(Dataset):
    def __init__(self, file_paths, labels, segment_starts, sr=22050, n_mels=300, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.segment_starts = segment_starts
        self.sr = sr
        self.n_mels = n_mels
        self.augment = augment

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        offset = self.segment_starts[idx]
        try:
            y, sr = librosa.load(path, sr=self.sr,offset=offset, duration=15.0)
            if self.augment:
                y = augment_audio(y, sr)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = librosa.util.fix_length(mel_db, size=300, axis=1)
            mel_tensor = torch.tensor(mel_db).unsqueeze(0).float() 
            #print(f"Mel shape before tensor: {mel_db.shape}")  
            return mel_tensor, label # [1, 128, 128]
        except Exception as e:
            # print(f"problems with the path {path}")
            # print(e)
            dummy_tensor = torch.zeros((1, 300, 300), dtype=torch.float32)
            return dummy_tensor, -1 

@torch.no_grad()
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        mask = labels != -1
        if not mask.any():
            continue
        inputs, labels = inputs[mask].to(device), labels[mask].to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item()*inputs.size(0)
        _, preds = torch.max(outputs,1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc



# ------------------ Training Utilities ------------------
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        mask = labels != -1
        if not mask.any():
            continue
        inputs, labels = inputs[mask].to(device), labels[mask].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def expand(file_list, label_list):
    paths, labels, starts = [], [], []
    for path, label in zip(file_list, label_list):
        for start in [0.0, 15.0]:
            paths.append(path)
            labels.append(label)
            starts.append(start)
    return paths, labels, starts

# ------------------ Main Script ------------------
# Example file loading -- replace with your actual dataset structure
AUDIO_DIR = r"C:\Polina\master\thesis\beat_this\data\gtzan_old"
all_files = []
labels = []
for genre in os.listdir(AUDIO_DIR):
    genre_path = os.path.join(AUDIO_DIR, genre)
    for file in os.listdir(genre_path):
        if file.endswith('.wav'):
            all_files.append(os.path.join(genre_path, file))
            labels.append(genre)

# Encode labels
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)

# Train-test split
train_files, val_files, train_labels, val_labels = train_test_split(
    all_files, encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42)

train_paths, train_labels, train_starts = expand(train_files, train_labels)
val_paths, val_labels, val_starts = expand(val_files, val_labels)
# Create datasets and dataloaders
train_dataset = GenreDataset(train_paths, train_labels, train_starts, augment=True)
val_dataset = GenreDataset(val_paths, val_labels,val_starts, augment=False)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Model training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MusicGenreCNN(num_classes=len(le.classes_)).to(device)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
early_stopping = EarlyStopping(patience = 7, min_delta= 0.001, mode = 'max')
num_epochs = 200
for epoch in tqdm(range(num_epochs)):
    train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

    print(f"Epoch {epoch+1}/{num_epochs} "
          f"| Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} "
          f"| Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    early_stopping(val_acc, model)
    if early_stopping.early_stop:
        print("⏹️ Early stopping triggered. Restoring best model.")
        model.load_state_dict(early_stopping.best_model_state)
        break
    #print(f"Epoch {epoch+1}/10 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
