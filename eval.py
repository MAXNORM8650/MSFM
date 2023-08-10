import torch
from tqdm import tqdm
import os
from tensorboardX import SummaryWriter
import numpy as np
from config import *
# coordinates_cat, proposalN, set, vis_num
from utils.cal_iou import calculate_iou
from utils.vis import image_with_boxes, image_with_box, combine_imgs, classification_results
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np

CFR=True
def eval(model, testloader, criterion, status, save_path, epoch):
    model.eval()
    print('Evaluating')

    raw_loss_sum = 0
    local_loss_sum = 0
    windowscls_loss_sum = 0
    total_loss_sum = 0
    iou_corrects = 0
    raw_correct = 0
    local_correct = 0
    obtain_row=[]
    obtain_local=[]
    desire=[]
    with torch.no_grad():
        for i, data in enumerate(tqdm(testloader)):
            if set == 'CUB':
                images, labels, boxes, scale = data
            else:
                images, labels = data
            desire.append(labels)
            images = images.cuda()
            labels = labels.cuda()
            total_proposal_windows_windows_scores, total_proposal_windows_windows_logits, total_proposal_windows_indices, window_scores, raw_logits, local_logits, FMA_Output
            = model(images, epoch, i, status)

            raw_loss = criterion(raw_logits, labels)
            local_loss = criterion(local_logits, labels)
            windowscls_loss = criterion(total_proposal_windows_windows_logits,
                                        labels.unsqueeze(1).repeat(1, total_proposal_windows).view(-1))

            total_loss = raw_loss + local_loss + windowscls_loss

            raw_loss_sum += raw_loss.item()
            local_loss_sum += local_loss.item()
            windowscls_loss_sum += windowscls_loss.item()

            total_loss_sum += total_loss.item()
            # Row branch

            pred = raw_logits.max(1, keepdim=True)[1]
            raw_correct += pred.eq(labels.view_as(pred)).sum().item()
            obtain_row.append(pred)
            # local branch
            pred = local_logits.max(1, keepdim=True)[1]
            obtain_local.append(pred)
            local_correct += pred.eq(labels.view_as(pred)).sum().item()

            # object branch tensorboard
            indices_ndarray = indices[:visualization_num,:total_proposal_windows].cpu().numpy()
            if i==0 or i==2 or i==4:
                
                with SummaryWriter(log_dir=os.path.join(save_path, 'log'), comment=status + 'Final Results') as writer:
                    cat_imgs = []
                    no_box_imgs=[]
                    local_ims=[]
                    s_box_imgs=[]
                    for j, indice_ndarray in enumerate(indices_ndarray):

                        if labels[j]==0:
                            results=[]
                            att=image_with_boxes(att_out[j])
                            im = image_with_boxes(images[j])
                            results.append(im)
                            results.append(att)
                            local_im = image_with_boxes(local_imgs[j])
                            results.append(local_im)
                                                  img = image_with_boxes(local_imgs[j], coordinates_concatenated[indice_ndarray])
                            results.append(img)
                            s_box_img=image_with_box(local_imgs[j], coordinates_concatenated[indice_ndarray])
                            results.append(s_box_img)
                            results.append(fin_res)
    #                         print(results)
                            results = np.concatenate(results, axis=1)
    #                         show_images(results, cols = 1)

                            writer.add_images(status + '/' + 'Original images' +'/' + 'Local images'+ '/' +'Object image with windows'+ str(i) + str(j), results, epoch, dataformats='HWC')

    raw_loss_avg = raw_loss_sum / (i+1)
    local_loss_avg = local_loss_sum / (i+1)
    windowscls_loss_avg = windowscls_loss_sum / (i+1)
    total_loss_avg = total_loss_sum / (i+1)

    raw_accuracy = raw_correct / len(testloader.dataset)
    local_accuracy = local_correct / len(testloader.dataset)
    
    if CFR==True:
        tar=torch.cat(desire).reshape(-1).cpu()
        o_r=torch.cat(obtain_row).reshape(-1).cpu()
        o_l=torch.cat(obtain_local).reshape(-1).cpu()
        classification_results(o_r, tar, "Row")
        classification_results(o_l, tar, "Local")
    return raw_loss_avg, windowscls_loss_avg, total_loss_avg, raw_accuracy, local_accuracy, local_loss_avg
