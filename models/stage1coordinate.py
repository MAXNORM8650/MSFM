import torch
from skimage import measure
def calculate_coordinates(self, feature_maps_set, single_feature_map):

    """ 
    1.	Calculate the total values of feature maps along the first dimension and store the sum in tensor “A”. 
    Calculate the mean of these sums along the last two dimensions and store it in tensor “a”.
    2.	Create a binary mask “M” by comparing each value of “A” with its corresponding value in a and thresholding. 
    This mask is used to identify regions of interest.
    3.	Repeat steps 1 and 2 for the input tensor “fm1” to create a mask “M1”.
    4.	Loop through each index and binary mask in M: 
         4a.	 Convert the binary mask to a numpy array and label connected components. 
         4b.	Calculate properties of these components using skimage.measure.regionprops. 
         4c.	 Find the largest component based on its area. 
         4d.	Create an intersection mask by combining the largest component from step c and the corresponding region from mask M1. 
         4e.	Find properties of the intersection mask. 
         4f.	 Calculate bounding box coordinates in terms of pixel indices.
    5.	Append the calculated bounding box coordinates to a list coordinates.
    """
    total_sum = torch.sum(feature_maps_set, dim=1, keepdim=True)
    mean_total_sum = torch.mean(total_sum, dim=[2, 3], keepdim=True)
    mask_total_sum = (total_sum > mean_total_sum).float()

    single_sum = torch.sum(single_feature_map, dim=1, keepdim=True)
    mean_single_sum = torch.mean(single_sum, dim=[2, 3], keepdim=True)
    mask_single_sum = (single_sum > mean_single_sum).float()

    calculated_coordinates = []
    for i, mask in enumerate(mask_total_sum):
        mask_np = mask.cpu().numpy().reshape(14, 14)
        labeled_components = measure.label(mask_np)

        properties = measure.regionprops(labeled_components)
        areas = [prop.area for prop in properties]
        max_area_index = areas.index(max(areas))

        intersection = ((labeled_components == (max_area_index + 1)).astype(int) + (mask_single_sum[i][0].cpu().numpy() == 1).astype(int)) == 2
        intersection_props = measure.regionprops(intersection.astype(int))

        if len(intersection_props) == 0:
            bounding_box = [0, 0, 14, 14]
        else:
            bounding_box = intersection_props[0].bbox

        x_top_left = bounding_box[0] * 32 - 1
        y_top_left = bounding_box[1] * 32 - 1
        x_bottom_right = bounding_box[2] * 32 - 1
        y_bottom_right = bounding_box[3] * 32 - 1

        if x_top_left < 0:
            x_top_left = 0
        if y_top_left < 0:
            y_top_left = 0

        coordinates = [x_top_left, y_top_left, x_bottom_right, y_bottom_right]
        calculated_coordinates.append(coordinates)

    return calculated_coordinates