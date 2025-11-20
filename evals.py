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


def activation_style_distance(stylized, target, model, layers=["conv4_1", "conv5_1"]):
    """
    Compare stylized vs. target activations using VGG features
    """
    # Extract dictionary of features from VGG
    feats_s = get_features(stylized, model)
    feats_t = get_features(target, model)
    vec_s, vec_t = [], []
    for layer in layers:
        f_s = feats_s[layer]   # [1, C, H, W]
        f_t = feats_t[layer]
        # Global Average Pooling on feature vectors [1, C]
        gap_s = torch.mean(f_s, dim=[2, 3])
        gap_t = torch.mean(f_t, dim=[2, 3])
        vec_s.append(gap_s)
        vec_t.append(gap_t)
    # Concatenate all selected layers to get final activation embeddings
    act_s = torch.cat(vec_s, dim=1)
    act_t = torch.cat(vec_t, dim=1)
    # Calculate Distances
    l2_dist = torch.norm(act_s - act_t, p=2).item()
    cos_dist = F.cosine_similarity(act_s, act_t).item()
    return l2_dist, cos_dist
