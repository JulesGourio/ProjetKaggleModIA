# models/alexnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes=4, name="AlexNet", in_channels=1):
        super(AlexNet, self).__init__()
        self.name = name
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=10e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=10e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        #self.init_weights()

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                # Initialiser les poids avec une distribution Gaussienne à moyenne zéro et écart-type 0.01
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                if layer.bias is not None:
                    if layer in [self.features[4], self.features[9], self.features[11]]:
                        # Initialiser les biais à 1 pour les couches conv2, conv4 et conv5
                        nn.init.constant_(layer.bias, 1)
                    else:
                        # Initialiser les biais à 0 pour les autres couches
                        nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Linear):
                # Initialiser les poids avec une distribution Gaussienne à moyenne zéro et écart-type 0.01
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                if layer.bias is not None:
                    # Initialiser les biais à 1 pour les couches fully connected
                    nn.init.constant_(layer.bias, 1)


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x






    

def Alexnet():
    return AlexNet(num_classes=4, name="AlexNet")



def count_parameters(model):
    """Compter le nombre total de paramètres dans un modèle."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

