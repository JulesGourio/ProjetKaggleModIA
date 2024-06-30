# data/dataset.py

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms

def only_2_classes(csv_path):
    df = pd.read_csv(csv_path)
    df = df.loc[df['label'].isin([1, 2])]
    df.loc[df['label'] == 1, 'label'] = 0
    df.loc[df['label'] == 2, 'label'] = 1
    #Enregistrement du nouveau fichier
    df.to_csv('data/train/2_classes.csv', index=False)
    return df

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, f"{self.data_frame.iloc[idx, 0]}.jpg")
        try:
            image = Image.open(img_name)
        except FileNotFoundError:
            return None, None
        
        label = self.data_frame.iloc[idx, 1] - 1  # Assuming labels are 1-indexed and converting to 0-indexed
        if self.transform:
            image = self.transform(image)
        
        return image, label


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        try:
            image = Image.open(img_name)
        except FileNotFoundError:
            return None, None
        ids = os.path.splitext(self.image_files[idx])[0]
        if self.transform:
            image = self.transform(image)
        return image, ids

class AddBiasChannel:
    def __init__(self, p0, p1):
        self.p0 = p0
        self.p1 = p1

    def __call__(self, img, label):

        # Determine the probability based on the label
        prob = self.p0 if label == 0 else self.p1

        # Convert probability to a float tensor
        prob_tensor = torch.tensor([prob], dtype=torch.float)

        sigma = 0.5

        # Sample S from Bernoulli distribution with probability `prob`
        S = torch.bernoulli(prob_tensor).item()

        # Decide the value of epsilon based on S
        if S == 0:
            epsilon = torch.zeros_like(img)  # 0 if S = 0
        else:
            epsilon = sigma * torch.randn_like(img)  # N(0, I) if S = 1

        # Expand dimensions to add the bias channel

        # Concatenate the bias channel to the image
        img_with_bias = torch.cat((img, epsilon), dim=0)
        
        return img_with_bias

    
class Dataset2classes(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, p0=0.5, p1=0.5):
        self.data_frame_labels = only_2_classes(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        
        # Liste des noms d'images valides à partir du CSV (convertir id en str et ajoute .jpg à chaque id)
        self.valid_image_files = self.data_frame_labels['id'].astype(str).apply(lambda x: x + '.jpg').tolist()
        
        # Vérifier que ces images existent dans le répertoire
        self.valid_image_files = [img for img in self.valid_image_files if os.path.isfile(os.path.join(self.root_dir, img))]
        
        # Filtrer le DataFrame pour inclure seulement les images valides
        valid_ids = [os.path.splitext(img)[0] for img in self.valid_image_files]
        self.data_frame_labels = self.data_frame_labels[self.data_frame_labels['id'].astype(str).isin(valid_ids)]

        # Initialiser AddBiasChannel avec p0 et p1
        self.add_bias_channel = AddBiasChannel(p0, p1)

    def __len__(self):
        return len(self.data_frame_labels)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of bounds")

        img_id = str(self.data_frame_labels.iloc[idx]['id'])
        img_name = os.path.join(self.root_dir, img_id + '.jpg')
        label = self.data_frame_labels.iloc[idx]['label']
        
        try:
            image = Image.open(img_name).convert('RGB')  # Assure que l'image est en RGB
        except FileNotFoundError:
            raise FileNotFoundError(f"File {img_name} not found.")
        
        if self.transform:
            image = self.transform(image)
        
        # Apply AddBiasChannel with the label
        image = self.add_bias_channel(image, label)

        return image, label