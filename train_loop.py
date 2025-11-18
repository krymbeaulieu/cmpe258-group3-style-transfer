import os
from datetime import datetime
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg19, VGG19_Weights
from torch.utils.data import DataLoader
# ours
from losses import ContentLoss, StyleLossBatch, adaptive_instance_normalization
from data_feat_load import get_transforms, get_features, StyleMeDataset
from models import UNet


from torchvision.utils import make_grid
from torchvision import transforms
import torchvision
import os

def save_prediction_triplet(content_tensor, output_tensor, style_tensor, path, epoch, step):
    """
    Saves a side-by-side comparison of content, stylized output, and style image.

    Args:
        content_tensor: [B, 3, H, W] content image batch (usually from COCO)
        output_tensor: [B, 3, H, W] model output batch
        style_tensor: [B, 3, H, W] style image batch (usually batch size = 1)
        path: directory to save the image
        epoch: current epoch number
        step: current training step
    """
    os.makedirs(path, exist_ok=True)
    to_pil = transforms.ToPILImage()

    content = content_tensor[0].detach().cpu().clamp(0, 1)
    output = output_tensor[0].detach().cpu().clamp(0, 1)
    style = style_tensor[0].detach().cpu().clamp(0, 1)

    # Match size if needed
    if style.shape[-2:] != output.shape[-2:]:
        style = torch.nn.functional.interpolate(style.unsqueeze(0), size=output.shape[-2:], mode='bilinear', align_corners=False)[0]

    # Stack as grid
    grid = make_grid([content, output, style], nrow=3)
    image = to_pil(grid)

    save_path = os.path.join(path, f"epoch{epoch+1}_step{step}.png")
    image.save(save_path)
    print(f"Saved prediction triplet to: {save_path}")
    
def train_style_transfer(
    style="Cubism",
    image_size=256,
    batch_size=4,
    num_epochs=2,
    num_workers=4,
    style_mode="gram",  
    data_dir_base="/scratch/cmpe258-sp25/group3_styletransfer/",
    save_every_n=None
):
    """
    Trains a feedforward image transformation network (e.g., UNet) to stylize COCO images
    into the appearance of a given WikiArt style using perceptual loss.

    Args:
        style (str): Name of the subfolder under wikiart/ containing images of a specific art style (e.g., "Cubism").
        image_size (int): The square size to which both content and style images will be resized (default: 256).
        batch_size (int): Number of COCO images per training batch.
        num_epochs (int): Number of passes through the entire COCO dataset.
        num_workers (int): Number of subprocesses for data loading. Set to 0 for debugging or if using torch.set_default_device.
        style_mode (str): Style transfer method. Choose "gram" for classic perceptual style loss using Gram matrices,
                          or "adain" for Adaptive Instance Normalization (default: "gram").
        data_dir_base (str): Base directory containing 'unlabeled2017/unlabeled2017/' and 'wikiart/' folders.
        save_every_n (int or None): If set, saves a model checkpoint every `n` steps in addition to saving the best model.

    Outputs:
        - Logs progress to a file at: logs/{style}_{style_mode}.log
        - Saves best model checkpoint to: checkpoints/{style}_{style_mode}_best.pth
        - (Optional) Periodic checkpoints if `save_every_n` is set
    """
    assert style_mode in ["gram", "adain"], "style_mode must be 'gram' or 'adain'"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    coco_dir = os.path.join(data_dir_base, "unlabeled2017/unlabeled2017/")
    style_dir = os.path.join(data_dir_base, "wikiart", style)

    # Logging setup
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    log_path = f"logs/{style}_{style_mode}.log"
    log_file = open(log_path, "a")
    def log(msg):
        print(msg)
        log_file.write(f"{msg}\n")
        log_file.flush()

    log(f"\n=== Starting run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    log(f"Device: {device} | Style: {style} | Mode: {style_mode}")

    # Transforms
    transform = get_transforms(image_size)

    # Datasets
    coco_dataset = StyleMeDataset(coco_dir, transform=transform)
    style_dataset = StyleMeDataset(style_dir, transform=transform)
    coco_loader = DataLoader(coco_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # VGG
    vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False

    # Model
    model = UNet(n_class=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Layers
    content_layer = 'conv4_2'
    style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1']

    best_loss = float("inf")
    global_step = 0
    total_steps = len(coco_loader)
    for epoch in range(num_epochs):
        for i, content_batch in enumerate(coco_loader):
            content_batch = content_batch.to(device)

            style_img = style_dataset[random.randint(0, len(style_dataset) - 1)].unsqueeze(0).to(device)

            if style_mode == "adain":
                content_features = get_features(content_batch, vgg)
                style_features = get_features(style_img, vgg)
                t = adaptive_instance_normalization(content_features['conv4_1'], style_features['conv4_1'])

                output = model(content_batch)
                output_features = get_features(output, vgg)
                content_loss = nn.functional.mse_loss(output_features['conv4_1'], t)
                style_loss = torch.tensor(0.0, device=device)

            else:  # gram mode
                output = model(content_batch)
                if output.shape[-2:] != style_img.shape[-2:]:
                    output = torch.nn.functional.interpolate(output, size=style_img.shape[-2:], mode='bilinear', align_corners=False)
                content_features = get_features(content_batch, vgg)
                output_features = get_features(output, vgg)
                style_features = get_features(style_img, vgg)

                content_loss_fn = ContentLoss(content_features[content_layer])
                content_loss_fn(output_features[content_layer])
                content_loss = content_loss_fn.loss

                style_loss = 0
                for layer in style_layers:
                    style_loss_fn = StyleLossBatch(style_features[layer])
                    style_loss_fn(output_features[layer])
                    style_loss += style_loss_fn.loss

            total_loss = content_loss + 1e5 * style_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            global_step += 1

            if i % 50 == 0:
                log(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{total_steps}], "
                    f"Content Loss: {content_loss.item():.4f}, Style Loss: {style_loss.item():.4f}, Total: {total_loss.item():.4f}")

            # Save best model
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                torch.save(model.state_dict(), f"checkpoints/{style}_{style_mode}_best.pth")
                log(f"Saved new best model (loss: {best_loss:.4f})")

            # Optionally save every n steps
            if save_every_n and global_step % save_every_n == 0:
                torch.save(model.state_dict(), f"checkpoints/{style}_{style_mode}_step{global_step}.pth")
                log(f"Saved model at step {global_step}")
            if global_step % 500 == 0:
                save_prediction_triplet(
                    content_tensor=content_batch,
                    output_tensor=output,
                    style_tensor=style_img,
                    path="predictions",
                    epoch=epoch,
                    step=global_step
                )

    log_file.close()
