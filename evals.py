import torch
import lpips

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_lpips_model = lpips.LPIPS(net='vgg').to(device).eval()

def get_lpips(img1, img2):
    """
    Computes LPIPS perceptual distance between two images.
    Both img1 and img2 must be [1, 3, H, W] tensors in [0,1].
    """
    with torch.no_grad():
        dist = lpips_model(img1, img2)
    return dist.item()
