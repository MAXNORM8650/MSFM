import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import shutil
import time
from config import num_classes, model_name, model_path, lr_milestones, lr_decay_rate, input_size, \
    root, end_epoch, save_interval, init_lr, batch_size, CUDA_VISIBLE_DEVICES, weight_decay, \
    proposalN, set, channels
from train import train
from utils.dataloader import read_dataset
from utils.halper import auto_load_resume
from utils.loss import FocalLoss
from models.MSFM import MSFMModel

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():

    trainloader, testloader = read_dataset(input_size, batch_size, root, set, subset = "WRIST")
    
    model = MainNet(proposalN=proposalN, num_classes=num_classes, channels=channels, theta=(0.6), red_p=0.15)

    criterion = FocalLoss()

    parameters = model.parameters()

   
    save_path = os.path.join(model_path, model_name)
    if os.path.exists(save_path):
        start_epoch, lr = auto_load_resume(model, save_path, status='train')
        assert start_epoch < end_epoch
    else:
        os.makedirs(save_path)
        start_epoch = 0
        lr = init_lr

    # define optimizers
#     optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(parameters, lr=0.00005)
    model = model.cuda()

    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_decay_rate)

    time_str = time.strftime("%Y%m%d-%H%M%S")
    shutil.copy('./config.py', os.path.join(save_path, "{}config.py".format(time_str)))

    train(model=model,
          trainloader=trainloader,
          testloader=testloader,
          criterion=criterion,
          optimizer=optimizer,
          scheduler=scheduler,
          save_path=save_path,
          start_epoch=start_epoch,
          end_epoch=end_epoch,
          save_interval=save_interval)
    
if __name__ == '__main__':
    main()