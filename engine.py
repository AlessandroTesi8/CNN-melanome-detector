import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os 
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = os.getcwd()

# Carica l'immagine di test
image = Image.open(path+'/melanoma detector/quest/IMG-2513.jpg')

transform = transforms.Compose([transforms.Resize((227, 227)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

image = transform(image)

classes = ("benign", "malignant")

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(179776, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # N, 3, 32, 32
        x = F.relu(self.conv1(x))   # -> N, 32, 30, 30
        x = self.pool(x)            # -> N, 32, 15, 15
        x = F.relu(self.conv2(x))   # -> N, 64, 13, 13
        x = self.pool(x)            # -> N, 64, 6, 6
        x = F.relu(self.conv3(x))   # -> N, 64, 4, 4
        x = torch.flatten(x, 1)     # -> N, 1024
        x = F.relu(self.fc1(x))     # -> N, 64
        x = self.fc2(x)             # -> N, 10
        return x


PATH = 'melanoma detector/cnn.pth'
loaded_model = ConvNet()
loaded_model.load_state_dict(torch.load(PATH)) # it takes the loaded dictionary, not the path file itself
loaded_model.to(device)
loaded_model.eval()

# Fai una previsione sull'immagine
with torch.no_grad():
    image = image.unsqueeze(0)  # Aggiungi una dimensione di batch
    output = loaded_model(image)

# Calcola la classe predetta
_, predicted_class = torch.max(output, 1)

# Ottieni il nome della classe predetta
predicted_class_name = classes[predicted_class.item()]

print(f'La classe predetta per l\'immagine è: {predicted_class_name}')
