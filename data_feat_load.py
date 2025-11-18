import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import os
from pathlib import Path

IMG_SIZE = 512

def get_transforms(img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    return transform

def load_image(path, transform, device):
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)

def imshow(tensor, title=None):
    image = tensor.cpu().clone().squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis("off")

def get_features(image, model):
    layers = {
        '0': 'conv1_1',
        '5': 'conv2_1',
        '10': 'conv3_1',
        '19': 'conv4_1',
        '21': 'conv4_2',
        '28': 'conv5_1'
    }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features


from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class StyleMeDataset(Dataset):
    # this works for both coco and your chosen art style
    def __init__(self, image_folder, transform=None):
        # given path to art style dir: e.g. wiki-art/cubism/
        # put style name as folder: cubism
        self.name = Path(image_folder).parts[-1] 
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('jpg', 'png', 'jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
