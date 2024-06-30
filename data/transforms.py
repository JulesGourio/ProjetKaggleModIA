# data/transforms.py

from torchvision import transforms
import torch

# Data augmentation transforms
data_augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor()
])

# Test transforsms
test_transforms_Alexnet = transforms.Compose([
    transforms.CenterCrop((227, 227)), #La taille d'entrée de l'AlexNet du papier de recherche n'est pas bonne (227, 227) au lieu de (224, 224)
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

#Ajout d'une normalisation

train_transforms_Resnet = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])


test_transforms_Resnet = transforms.Compose([
    #transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])


#Cas de test on changera p0 et p1 plus tard dans le rapport


def get_biased_transforms(p0, p1):
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(227),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0], std=[0.229, 1])
    ])
