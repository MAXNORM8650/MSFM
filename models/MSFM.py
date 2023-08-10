import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .models import resnet
from .models.FMA import feature_map_augment
from .models.stage1coordinate import calculate_coordinates
from .models.modified_NMS import non_maximum_suppression
from .models.stage2windows import AdaptivePoolingAndNMS
from config import *
class MSFMModel(nn.Module):
    def __init__(self, proposalN=6, num_classes=2, channels=2048, theta=(0.1, 0.4), red_p=0.9, pretrain_path=None, pretrain_path_B=None):
        """
        Initialize the MSFM model.
        
        Args:
            proposalN (int): Number of proposal windows.
            num_classes (int): Number of output classes.
            channels (int): Number of input channels.
            theta (tuple or float): Range or fixed value of theta for attention thresholding.
            red_p (float): Pixel Reduction parameter in Feature Map Augmentation (FMA).
            pretrain_path (str): Path to the pre-trained weights for FMA backbone.
            pretrain_path_B (str): Path to the pre-trained weights for CNN backbone.
        """
        super(MSFMModel, self).__init__()
        self.num_classes = num_classes
        self.proposalN = proposalN
        self.theta = theta
        self.red_p = red_p
        self.FMA_CNN_BackBone = resnet.resnet50(pretrained=True, pth_path = FMA_pretrain_path)
        self.CNN_BackBone = resnet.resnet50(pretrained=True, pth_path = pretrain_path)
        self.BranchModule = AdaptivePoolingAndNMS()  # Define AdaptivePoolingAndNMS class
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x, epoch, batch_idx, status='test', DEVICE='cuda', 
                ratios=None, window_nums_sum=None,
                N_list=None, iou_threshs=None, coordinates_cat=None):
        """
        Forward pass of the MSFM model.
        
        Args:
            x (torch.Tensor): Input tensor.
            epoch (int): Current epoch.
            batch_idx (int): Current batch index.
            status (str): Model status ('test' or 'train').
            DEVICE (str): Device to run the model on.
            ratios (list): List of ratios.
            window_nums_sum (list): List of window numbers.
            N_list (list): List of N values.
            iou_threshs (list): List of IoU thresholds.
            coordinates_cat (list): List of coordinates.
            
        Returns:
            tuple: Tuple containing various output tensors.
        """
        FMA_Output = feature_map_augment(x, torch.sum(self.FMA_CNN_BackBone(x)[-1], 
                                                      dim=1, keepdim=True),
                                         theta=self.theta, red_p=self.red_p)  # FMA

        FM, embedding, CFM = self.CNN_BackBone(att_out)  # Feature Map and Class Feature Map
        batch_size, channel_size, side_size, _ = fm.shape
        assert channel_size == 2048

        raw_logits = self.fc(embedding)

        coordinates = torch.tensor(calculate_coordinates(fm.detach(), CFM.detach()))

        local_imgs = torch.zeros([batch_size, 3, 448, 448]).to(DEVICE)  # [N, 3, 448, 448]
        for i in range(batch_size):
            [x0, y0, x1, y1] = coordinates[i]
            local_imgs[i:i + 1] = F.interpolate(FMA_Output[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)], size=(448, 448),
                                                mode='bilinear', align_corners=True)  # [N, 3, 224, 224]
        local_fm, local_embeddings, local_tscams = self.CNN_BackBone(local_imgs.detach())  # [N, 2048]
        local_logits = self.fc(local_embeddings)  # [N, 2]
        
        proposalN_indices, proposalN_windows_scores, window_scores \
            = self.BranchModule(self.proposalN, local_fm.detach(), ratios, window_nums_sum, N_list, iou_threshs, DEVICE)

        if status == "train":
            # Window images classification
            window_imgs = torch.zeros([batch_size, self.proposalN, 3, 224, 224]).to(DEVICE)  # [N, 4, 3, 224, 224]
            wnds = []
            for i in range(batch_size):
                wnd = []
                for j in range(self.proposalN):
                    [x0, y0, x1, y1] = coordinates_cat[proposalN_indices[i, j]]
                    window_imgs[i:i + 1, j] = F.interpolate(final_local_imgs[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)], size=(224, 224),
                                                            mode='bilinear', align_corners=True)  # [N, 4, 3, 224, 224]
                    wnd.append([x0, y0, x1, y1])
                wnds.append(wnd)
                   
            window_imgs = window_imgs.reshape(batch_size * self.proposalN, 3, 224, 224)  # [N*4, 3, 224, 224]
            _, window_embeddings, _ = self.CNN_BackBone(window_imgs.detach())  # [N*4, 2048]
            proposalN_windows_logits = self.fc(window_embeddings)  # [N*4, 200]
        else:
            proposalN_windows_logits = torch.zeros([batch_size * self.proposalN, self.num_classes]).to(DEVICE)

        return proposalN_windows_scores, proposalN_windows_logits, proposalN_indices, window_scores, raw_logits, local_logits, FMA_Output
