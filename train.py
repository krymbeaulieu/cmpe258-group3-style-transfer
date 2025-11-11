import torch
import time
from data_feat_load import get_features
from losses import adaptive_instance_normalization, mean_std

"""
Following https://docs.pytorch.org/tutorials/advanced/neural_style_tutorial.html#loss-functions
and extending for AdaIN. https://github.com/naoto0804/pytorch-AdaIN for initial reference

Used GPT to make docstrings because it's faster.
"""

import torch
import time
from losses import adaptive_instance_normalization, mean_std

def get_features(image, model):
    """
    Extracts intermediate features from specific layers of a CNN model.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor of shape [1, 3, H, W].
    model : torch.nn.Module
        The pre-trained CNN model (e.g. VGG) from which to extract features.

    Returns
    -------
    features : dict
        A dictionary mapping layer names (e.g., 'conv1_1') to feature maps.
    """
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
    
def style_transfer(content, style, model, content_weight=1, style_weight=1e6,
                   steps=5000, stop_threshold=10000, loss_method="gram",
                   layers=None):
    """
    Performs neural style transfer using either Gram matrix or AdaIN-based style loss.

    Parameters
    ----------
    content : torch.Tensor
        The content image tensor of shape [1, 3, H, W].
    style : torch.Tensor
        The style image tensor of shape [1, 3, H, W].
    model : torch.nn.Module
        A pre-trained CNN model (usually VGG19) truncated at appropriate layers.
    content_weight : float, optional
        Weight for content loss. The default is 1.
    style_weight : float, optional
        Weight for style loss. The default is 1e6. (for gram)
        Recommend much smaller for AdaIN. 10 or so. 
    steps : int, optional
        Number of optimization steps. The default is 5000.
    stop_threshold : float, optional
        Early stopping threshold for total loss. The default is 10000.
    loss_method : str, optional
        Loss method to use: 'gram' or 'adain'. The default is 'gram'.
    layers : list of str, optional
        List of layer names to use for style loss. Defaults to commonly used VGG layers.

    Returns
    -------
    target : torch.Tensor
        The stylized image tensor.
    """
    # Validation Checks
    if loss_method not in ("gram", "adain"):
        raise ValueError("loss_method must be 'gram' or 'adain'")
    if layers is None:
        layers = ['conv1_1','conv2_1','conv3_1','conv4_1','conv5_1']
        print(f"layers is none, using {layers} (VGG feature layers)")

    device = content.device
    target = content.clone().requires_grad_(True).to(device)
    optimizer = torch.optim.Adam([target], lr=0.005)

    start_time = time.time()

    for i in range(steps):
        target_features = get_features(target, model)
        content_features = get_features(content, model)
        style_features = get_features(style, model)

        # Content loss
        if loss_method == "gram":
            content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
        elif loss_method == "adain":
            # In AdaIN, match target features to the AdaIN-transformed content features
            # Although AdaIN doesn't use Gram matrices, we still use MSE to align features
            t_feat = adaptive_instance_normalization(content_features['conv4_2'], style_features['conv4_2'])
            content_loss = torch.mean((target_features['conv4_2'] - t_feat)**2)

        # Style loss
        style_loss = 0
        if loss_method == "gram":
            for layer in layers:
                target_f = target_features[layer]
                style_f = style_features[layer]
                _, c, h, w = target_f.shape
                # Reshape feature maps and compute Gram matrices to capture texture and style
                target_gram = torch.mm(target_f.view(c, h * w), target_f.view(c, h * w).t())
                style_gram = torch.mm(style_f.view(c, h * w), style_f.view(c, h * w).t())
                # The style loss is the mean squared error between target and style Gram matrices
                layer_style_loss = torch.mean((target_gram - style_gram)**2) / (c * h * w)
                style_loss += layer_style_loss
        elif loss_method == "adain":
            for layer in layers:
                t_mean, t_std = mean_std(target_features[layer])
                s_mean, s_std = mean_std(style_features[layer])
                # AdaIN style loss compares mean and std statistics across channels
                # Still uses MSE, but on per-channel statistics instead of feature correlations
                style_loss += torch.mean((t_mean - s_mean)**2) + torch.mean((t_std - s_std)**2)

        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            target.clamp_(0, 1)

        if i % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Step [{i}/{steps}]  Total Loss: {total_loss.item():.2f}  Elapsed Time: {elapsed:.2f}s")

        if total_loss.item() < stop_threshold:
            print(f"Stopping early at step {i}, total loss = {total_loss.item():.2f}")
            break

    total_time = time.time() - start_time
    print(f"Style transfer completed in {total_time:.2f} seconds.")

    return target
