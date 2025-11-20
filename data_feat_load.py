import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from pathlib import Path
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Subset

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



# -------------------------------------------------------------------
# Style Classifier Transforms + DataLoader Utilities
# -------------------------------------------------------------------

def load_wiki_subset(root_path, selected_styles, img_size=256, batch_size=32, samples_per_class=500, val_split=0.2, normalize=True, num_workers=2,seed=42):
    """
    Loads a balanced subset of WikiArt using selected style folders
    """
    # Define transforms
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) # Required for ResNet training

    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Build filtered list of files
    all_images = []
    all_labels = []
    class_to_idx = {}
    idx_to_class = {}

    for idx, style in enumerate(selected_styles):
        style_dir = os.path.join(root_path, style)
        if not os.path.isdir(style_dir):
            print(f"Check if style folder exists: {style_dir}")
            continue

        # Map label index
        class_to_idx[style] = idx
        idx_to_class[idx] = style
        files = [os.path.join(style_dir, f) for f in os.listdir(style_dir)]

        # Limit to N samples per style
        files = files[:samples_per_class]
        all_images.extend(files)
        all_labels.extend([idx] * len(files))

    # Create custom dataset structure
    class SubsetWikiArt(torch.utils.data.Dataset):
        def __init__(self, img_paths, labels, transform):
            self.img_paths = img_paths
            self.labels = labels
            self.transform = transform
        def __len__(self):
            return len(self.img_paths)
        def __getitem__(self, idx):
            path = self.img_paths[idx]
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
            label = self.labels[idx]
            return img, label

    # Subset wikiart
    full_train_set = SubsetWikiArt(all_images, all_labels, train_tfms)
    full_val_set   = SubsetWikiArt(all_images, all_labels, val_tfms)

    # Train/Val Split
    total = len(all_images)
    val_size = int(total * val_split)
    train_size = total - val_size
    gen = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(range(total), [train_size, val_size], generator=gen)
    train_dataset = Subset(full_train_set, train_idx.indices)
    val_dataset   = Subset(full_val_set, val_idx.indices)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, len(selected_styles), selected_styles
