from utils.localization_utils import *
import numpy as np

# Configuration
CUDA_VISIBLE_DEVICES = '0'
dataset_name = 'MURA'
model_name = ''

batch_size = 6
visualization_num = batch_size
evaluate_trainset = False
save_interval = 1
max_checkpoint_num = 200
end_epoch = 200
initial_learning_rate = 0.001
learning_rate_milestones = [60, 100]
learning_rate_decay_rate = 0.1
weight_decay_coefficient = 1e-4
stride = 32
channels = 2048
input_size = 448
pretrained_model_path_FMA = './models/pretrained/resnet50-19c8e357.pth' # freez it
pretrained_model_path_MSFM = './models/pretrained/resnet50-19c8e357.pth'

# Example of the Window info for MURA datasets
num_windows_per_scale = [3, 2, 1]

total_proposal_windows = sum(num_windows_per_scale)

window_scales = [192, 256, 320]
iou_thresholds = [0.25, 0.25, 0.25]
window_ratios = [
    [6, 6], [5, 7], [7, 5],
    [8, 8], [6, 10], [10, 6], [7, 9], [9, 7],
    [10, 10], [9, 11], [11, 9], [8, 12], [12, 8]
]

if dataset_name == 'MURA':
    checkpoint_path = './checkpoint/MURA'
    dataset_root = './datasets/MURA'
    num_classes = 2

# Compute window information
window_nums_per_scale = compute_window_nums(window_ratios, stride, input_size)
indices_ndarrays = [np.arange(0, num_windows).reshape(-1, 1) for num_windows in window_nums_per_scale]
coordinates = [
    indices_to_coordinates(indices_ndarray, stride, input_size, window_ratios[i])
    for i, indices_ndarray in enumerate(indices_ndarrays)
]
coordinates_concatenated = np.concatenate(coordinates, 0)
window_milestones = [sum(window_nums_per_scale[:i + 1]) for i in range(len(window_nums_per_scale))]
# Example of window sums per scale on MURA dataset
window_sums_per_scale = [0, sum(window_nums_per_scale[:3]), sum(window_nums_per_scale[3:8]), sum(window_nums_per_scale[8:])]
