import torch
import random
import torch.nn as nn
import torch.nn.functional as F
def feature_map_augment(images, attention_map, theta_range=(0.1, 0.4), padding_ratio=0.1, reduction_prob=0.9):
    """
    Apply feature map augmentation to input images based on attention maps.

    Args:
        images (torch.Tensor): Input images (batch_size, channels, height, width).
        attention_map (torch.Tensor): Attention maps corresponding to images (batch_size, 1, height, width).
        theta_range (tuple or float): Range or fixed value of theta for attention thresholding.
        padding_ratio (float): Unused in the current version but could represent padding ratio for cropping.
        reduction_prob (float): Probability of reducing the attention values outside the threshold.
        
    Returns:
        torch.Tensor: Augmented images after applying feature map dropout.
    """
    batch_size, _, img_height, img_width = images.size()
    drop_masks = []
    
    for batch_index in range(batch_size):
        atten_map = attention_map[batch_index:batch_index + 1]
        
        if isinstance(theta_range, tuple):
            theta_min, theta_max = theta_range
            theta_d = random.uniform(theta_min, theta_max) * atten_map.max()
        else:
            theta_d = theta_range * atten_map.max()
            
        mask = F.interpolate(atten_map, size=(img_height, img_width), mode='bilinear') < theta_d
        reduction_prob_tensor = torch.tensor(reduction_prob, device='cuda')
        mask = torch.where(mask == 0.0, reduction_prob_tensor, mask.float())
        
        drop_masks.append(mask)
    
    drop_masks = torch.cat(drop_masks, dim=0)
    drop_images = images * drop_masks.unsqueeze(1)
    return drop_images