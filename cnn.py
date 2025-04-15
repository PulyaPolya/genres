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

# ------------------ Model Definition ------------------
class MusicGenreCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(MusicGenreCNN, self).__init__()
        self.batch_norm0 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.elu1 = nn.ELU(64)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
       
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ------------------ Data Augmentation ------------------
def augment_audio(y, sr):
    y = librosa.effects.pitch_shift(y, sr, n_steps=random.uniform(-1, 1))
    y = librosa.effects.time_stretch(y, rate=random.uniform(0.9, 1.1))
    noise = np.random.randn(len(y))
    y = y + 0.005 * noise
    return y

# ------------------ Dataset ------------------
class GenreDataset(Dataset):
    def __init__(self, file_paths, labels, sr=22050, n_mels=128, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.sr = sr
        self.n_mels = n_mels
        self.augment = augment

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        try:
            y, sr = librosa.load(path, sr=self.sr, duration=30.0)
            if self.augment:
                y = augment_audio(y, sr)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = librosa.util.fix_length(mel_db, size=128, axis=1)
            mel_tensor = torch.tensor(mel_db).unsqueeze(0).float() 
            #print(f"Mel shape before tensor: {mel_db.shape}")  
            return mel_tensor, label # [1, 128, 128]
        except Exception as e:
            print(f"problems with the path {path}")
            print(e)
            dummy_tensor = torch.zeros((1, 128, 128), dtype=torch.float32)
            return dummy_tensor, -1 

        

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

# Create datasets and dataloaders
train_dataset = GenreDataset(train_files, train_labels, augment=False)
val_dataset = GenreDataset(val_files, val_labels, augment=False)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=16)

# Model training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MusicGenreCNN(num_classes=len(le.classes_)).to(device)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
    print(f"Epoch {epoch+1}/10 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
