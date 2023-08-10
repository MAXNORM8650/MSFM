import numpy as np

def non_maximum_suppression(scores, num_proposals, iou_thresholds, bounding_box_coordinates):
    """
    Input Parameters:
      1.	scores: A numpy array containing confidence scores for each bounding box proposal. 
      Each row represents a proposal, and there's a single confidence score associated with it.
      2.	num_proposals: An integer representing the desired number of retained proposals after applying NMS.
      3.	iou_thresholds: A threshold value representing the Intersection over Union (IoU) overlap threshold. If two bounding boxes have an IoU greater than this threshold, one of them will be suppressed.
      4.	bounding_box_coordinates: A numpy array containing the coordinates of bounding box proposals. 
      Each row represents a proposal, and columns 1 to 4 represent the (x1, y1, x2, y2) coordinates of the bounding box, while the last column (index 5) contains the index of the proposal.
    Post-processing:
    If the loop completes without selecting the desired number of proposals, the last selected proposal is added repeatedly until the desired count is reached.
    Output: 
    The function returns a numpy array containing the indices of the selected proposals after NMS.
    
 """    
    if not (isinstance(scores, np.ndarray) and len(scores.shape) == 2 and scores.shape[1] == 1):
        raise TypeError('scores array is not in the correct format')

    num_windows = scores.shape[0]
    indexed_coordinates = np.concatenate((scores, bounding_box_coordinates), axis=1)

    indices = np.argsort(indexed_coordinates[:, 0])
    indexed_coordinates_with_indices = np.concatenate((indexed_coordinates, np.arange(0, num_windows).reshape(num_windows, 1)), axis=1)[indices]
    selected_indices = []

    remaining_windows = indexed_coordinates_with_indices

    while remaining_windows.any():
        current_window = remaining_windows[-1]
        selected_indices.append(current_window[5])

        if len(selected_indices) == num_proposals:
            return np.array(selected_indices).reshape(1, num_proposals).astype(np.int64)
        
        remaining_windows = remaining_windows[:-1]

        start_max = np.maximum(remaining_windows[:, 1:3], current_window[1:3])
        end_min = np.minimum(remaining_windows[:, 3:5], current_window[3:5])
        lengths = end_min - start_max + 1
        intersection_area = lengths[:, 0] * lengths[:, 1]
        intersection_area[np.logical_or(lengths[:, 0] < 0, lengths[:, 1] < 0)] = 0
        iou_map_current = intersection_area / (
            (remaining_windows[:, 3] - remaining_windows[:, 1] + 1) * 
            (remaining_windows[:, 4] - remaining_windows[:, 2] + 1) +
            (current_window[3] - current_window[1] + 1) * 
            (current_window[4] - current_window[2] + 1) - intersection_area)
        
        remaining_windows = remaining_windows[iou_map_current <= iou_thresholds]

    while len(selected_indices) != num_proposals:
        selected_indices.append(current_window[5])

    return np.array(selected_indices).reshape(1, -1).astype(np.int64)
