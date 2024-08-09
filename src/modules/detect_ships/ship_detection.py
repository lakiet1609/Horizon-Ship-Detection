import numpy as np
import os
import cv2
import sys
import torch
import torchvision
from torch.utils.data import DataLoader
from src.modules.detect_ships.dataset import Custom_Dataset
from src.modules.detect_ships.model import RCF
from src.logger import logging
from src.exception import CustomException
from src.utils import *
from src.config.configuration import ship_detection_settings as sds
from src.config.configuration import general_settings as gs


class ShipDetection:
    def __init__(self, img):
        self.img = img
        logging.info('Initialize edge detection module ...')


    def single_scale_test(self, model, test_loader, save_dir):
        model.eval()
        img_names = []
        for idx, image in enumerate(test_loader):
            image = image.cuda()
            _, _, H, W = image.shape
            results = model(image)
            all_res = torch.zeros((len(results), 1, H, W))
            for i in range(len(results)):
                all_res[i, 0, :, :] = results[i]
            filename = f'three_ships_horizon_{idx}'
            img_names.append(filename)
            torchvision.utils.save_image(1 - all_res, os.path.join(save_dir, '%s.jpg' % filename))
            fuse_res = torch.squeeze(results[-1].detach()).cpu().numpy()
            fuse_res = ((1 - fuse_res) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(save_dir, '%s_ss.png' % filename), fuse_res)
        logging.info('Running single-scale test done')
        


    def detect_edge(self):
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = sds.gpu

        if not os.path.isdir(sds.edge_output_path):
            os.makedirs(sds.edge_output_path)
    
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
        self.single_scale_test(model, test_loader, sds.edge_output_path)
        

    def detecting_ships(self):
        final_bboxes = []
        for img in os.listdir(sds.edge_output_path):
            img_path = os.path.join(sds.edge_output_path, img)
            img_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            erase_horizon_img = draw_white_line(img_data, sds.start_point, sds.end_point)
            erode_image = erode_img(erase_horizon_img, sds.erode_kernel, sds.erode_iterations)
            dilate_image = dilate_img(erode_image, sds.dilate_kernel, sds.dilate_iterations) 
            mask_image, bboxes = check_area_of_mask(dilate_image, sds.min_width, sds.min_height, sds.expansion_size)
            img_output_path = os.path.join(gs.output_path, sds.mask_file_name)
            cv2.imwrite(img_output_path, mask_image)
            final_bboxes.append(bboxes)
        
        return final_bboxes
    
    def save_and_draw_bb_img(self):
        try:
            image = cv2.imread(gs.input_path)
            if image is None:
                raise ValueError("Image not found or unable to load image.")
            
            bboxes = self.detecting_ships()
            color = (0, 255, 0) 
            thickness = 2  
            for bbox in bboxes:
                x, y, w, h = bbox
                top_left = (x, y)
                bottom_right = (x + w, y + h)
                image = cv2.rectangle(image, top_left, bottom_right, color, thickness)
            
            output_name = 'output_image.tiff'
            output_path = os.path.join(gs.output_path, output_name)
            cv2.imwrite(os.path.join(gs.output_path, output_name), image)
            logging.info(f"Image saved to {output_path}")

            return output_path
        except Exception as e:
            raise CustomException(e,sys)
 