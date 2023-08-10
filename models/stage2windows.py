import torch
import torch.nn as nn
import numpy as np

class AdaptivePoolingAndNMS(nn.Module):
    """
    •	pooling_ratios: List of pooling ratios for the AvgPool2d layers.
    •	num_proposals: Number of proposals to be retained after applying NMS.
    •	input_tensor: Input tensor to the module.
    •	window_nums_sum: List of window number sums.
    •	N_list: List of values for N.
    •	iou_thresholds: List of IoU thresholds.
    •	coordinates_cat: Coordinates for concatenation.
    •	device: Device to be used ('cuda' or 'cpu').
    """
    def __init__(self, pooling_ratios):
        
        super(AdaptivePoolingAndNMS, self).__init__()
        self.avg_pools = [nn.AvgPool2d(ratio, 1) for ratio in pooling_ratios]

    def forward(self, num_proposals, input_tensor, pooling_ratios, window_nums_sum, N_list, iou_thresholds, coordinates_cat, device='cuda'):
        batch, channels, _, _ = input_tensor.size()
        
        pooled_tensors = [avg_pool(input_tensor) for avg_pool in self.avg_pools]

        feature_map_sums = [torch.sum(pooled, dim=1) for pooled in pooled_tensors]

        all_scores = torch.cat([fm.view(batch, -1, 1) for fm in feature_map_sums], dim=1)
        windows_scores_np = all_scores.data.cpu().numpy()
        window_scores = torch.from_numpy(windows_scores_np).to(device).reshape(batch, -1)

        proposal_indices = []
        for i, scores in enumerate(windows_scores_np):
            indices_results = []
            for j in range(len(window_nums_sum) - 1):
                indices_results.append(nms(scores[window_nums_sum[j]:window_nums_sum[j+1]], proposalN = N_list[j], iou_threshs=iou_thresholds[j], coordinates=coordinates_cat[window_nums_sum[j]:window_nums_sum[j+1]]) + window_nums_sum[j])
            proposal_indices.append(np.concatenate(indices_results, axis=1))

        proposal_indices = np.array(proposal_indices).reshape(batch, num_proposals)
        proposal_indices = torch.from_numpy(proposal_indices).to(device)
        
        proposal_scores = torch.cat([torch.index_select(all_score, dim=0, index=proposal_indices[i]) for i, all_score in enumerate(all_scores)], 0).reshape(batch, num_proposals)

        return proposal_indices, proposal_scores, window_scores
    