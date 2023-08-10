from PIL import Image
import numpy as np
import cv2
from config import proposalN
from torchmetrics import CohenKappa
from torchmetrics import ConfusionMatrix
from torchmetrics.functional import auc

def image_with_boxes(image, coordinates=None, box_color=None, num_proposals=6):
    '''
    :param image: image array(CHW) tensor
    :param coordinates: bounding box coordinates, coordinates.shape = [num_proposals, 4], coordinates[0] = (x0, y0, x1, y1)
    :param box_color: color of the bounding boxes
    :param num_proposals: number of proposal windows
    :return: image with bounding boxes (HWC)
    '''

    if isinstance(image, np.ndarray):
        image = image.copy()
    else:
        image = image.clone().detach()

        proposal_colors = [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0)]

        # Anti-normalization
        std = [0.229, 0.224, 0.225]
        mean = [0.485, 0.456, 0.406]
        image[0] = image[0] * std[0] + mean[0]
        image[1] = image[1] * std[1] + mean[1]
        image[2] = image[2].mul(std[2]) + mean[2]
        image = image.mul(255).byte()

        image = image.data.cpu().numpy()
        image = image.astype(np.uint8)
        image = np.transpose(image, (1, 2, 0))  # CHW --> HWC
        image = image.copy()

    if coordinates is not None:
        for i, coordinate in enumerate(coordinates):
            if box_color:
                image = cv2.rectangle(image, (int(coordinate[1]), int(coordinate[0])),
                                      (int(coordinate[3]), int(coordinate[2])),
                                      box_color, 2)
            else:
                if i < num_proposals:
                    # coordinates(x, y) is reversed in numpy
                    image = cv2.rectangle(image, (int(coordinate[1]), int(coordinate[0])),
                                          (int(coordinate[3]), int(coordinate[2])),
                                          proposal_colors[i], 2)
                else:
                    image = cv2.rectangle(image, (int(coordinate[1]), int(coordinate[0])),
                                          (int(coordinate[3]), int(coordinate[2])),
                                          (255, 255, 255), 2)
    return image



def combine_imgs(org, ot, codd):
    size=(codd[3]-codd[1], codd[2]-codd[0])
    image_boxes2 = Image.fromarray(ot)
    image_boxes2 = image_boxes2.resize(size)
    imgo = Image.fromarray(org)
    imgo.paste(image_boxes2, (codd[1], codd[0]))
    return imgo

def classification_results(pre, tar, stat):
    confmat = ConfusionMatrix(num_classes=2)
    cohenkappa = CohenKappa(num_classes=2)
    acc=Accuracy()
    
    print("*****************************")
    print(stat)
    print("Confusion Matrix", confmat(pre, tar).numpy())
    print("cohenkappa score", cohenkappa(pre, tar).numpy())
    print("AUC", auc(pre, tar, reorder=True).numpy())
    print("accuracy", acc(pre, tar).numpy())
    print("*****************************")
    

def tenssor2img(images):
    if type(images) is not np.ndarray:
        image = images.clone().detach()

        rgbN = [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0)]

        # Anti-normalization
        std = [0.229, 0.224, 0.225]
        mean = [0.485, 0.456, 0.406]
        image[0] = image[0] * std[0] + mean[0]
        image[1] = image[1] * std[1] + mean[1]
        image[2] = image[2].mul(std[2]) + mean[2]
        image = image.mul(255).byte()

        image = image.data.cpu().numpy()

        image.astype(np.uint8)

        image = np.transpose(image, (1, 2, 0))  # CHW --> HWC
        
    return Image.fromarray(image)