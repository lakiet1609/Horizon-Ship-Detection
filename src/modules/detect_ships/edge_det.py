import numpy as np
import os
import cv2
import sys
import torch
import torchvision
from torch.utils.data import DataLoader
from dataset import Custom_Dataset
from model import RCF
from src.logger import logging
from src.exception import CustomException
from src.config.configuration import ship_detection_settings as sds
from src.config.configuration import general_settings as gs



class EdgeDetection:
    def __init__(self, imgs, output_path):
        self.imgs = imgs
        self.output_path = output_path
        logging.info('Initialize edge detection module ...')


    def single_scale_test(self, model, test_loader, save_dir):
        model.eval()
        for idx, image in enumerate(test_loader):
            image = image.cuda()
            _, _, H, W = image.shape
            results = model(image)
            all_res = torch.zeros((len(results), 1, H, W))
            for i in range(len(results)):
                all_res[i, 0, :, :] = results[i]
            filename = f'three_ships_horizon_{idx}'
            torchvision.utils.save_image(1 - all_res, os.path.join(save_dir, '%s.jpg' % filename))
            fuse_res = torch.squeeze(results[-1].detach()).cpu().numpy()
            fuse_res = ((1 - fuse_res) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(save_dir, '%s_ss.png' % filename), fuse_res)
        logging.info('Running single-scale test done')



    def detect_edge(self):
        try:
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = sds.gpu

            if not os.path.isdir(sds.output_path):
                os.makedirs(sds.output_path)
        
            test_dataset  = Custom_Dataset(root=gs.input_path)
            test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1, drop_last=False, shuffle=False)
            
            model = RCF().cuda()

            if os.path.isfile(sds.checkpoint):
                logging.info("=> loading checkpoint from '{}'".format(sds.checkpoint))
                checkpoint = torch.load(sds.checkpoint)
                model.load_state_dict(checkpoint)
                logging.info("=> checkpoint loaded")
            else:
                logging.info("=> no checkpoint found at '{}'".format(sds.checkpoint))

            logging.info('Performing the testing...')
            self.single_scale_test(model, test_loader, sds.output_path)
        
        except Exception as e:
            raise CustomException(e,sys)