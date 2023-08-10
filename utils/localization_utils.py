import numpy as np

def compute_coordinate(image_size, stride, index, ratio):
    size = image_size // stride
    column_window_num = (size - ratio[1]) + 1
    x_index = index // column_window_num
    y_index = index % column_window_num
    x_left_top = x_index * stride - 1
    y_left_top = y_index * stride - 1
    x_right_low = x_left_top + ratio[0] * stride
    y_right_low = y_left_top + ratio[1] * stride

    x_left_top = max(0, x_left_top)  # Ensure non-negative values
    y_left_top = max(0, y_left_top)

    coordinate = np.array((x_left_top, y_left_top, x_right_low, y_right_low)).reshape(1, 4)
    return coordinate

def indices_to_coordinates(indices, stride, image_size, ratio):
    '''
    Example usage
    image_size = 224
    stride = 32
    ratio = (2, 2)
    indices = np.array([[0, 1], [2, 3]])  # Example indices
    coordinates = indices_to_coordinates(indices, stride, image_size, ratio)
    print(coordinates)
    '''
    batch, _ = indices.shape
    coordinates = np.empty((batch, 4), dtype=int)

    for j, index in enumerate(indices):
        coordinates[j] = compute_coordinate(image_size, stride, index, ratio)

    return coordinates
def compute_window_nums(ratios, stride, input_size):
    """
    Calculate the number of windows for each given ratio.

    Parameters:
    ratios (list of tuples): List of width and height ratios for each window.
    stride (int): Stride value.
    input_size (int): Size of the input.

    Returns:
    window_nums (list of int): List of window numbers for each ratio.
    """
    size = input_size // stride
    window_nums = []

    for ratio in ratios:
        width_windows = size - ratio[0] + 1
        height_windows = size - ratio[1] + 1
        window_nums.append(width_windows * height_windows)

    return window_nums
