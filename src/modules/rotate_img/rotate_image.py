import cv2
import os
import sys
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.config.configuration import rotate_settings as rs
from src.config.configuration import general_settings as gs


class RotateImage:
    def __init__(self, img, start_point, end_point):
        self.img = img
        self.start_point = start_point
        self.end_point = end_point
        logging.info('Initialize rotate image module ...')
    
    def rotate_image_to_level(self):
        try:

            dx = self.end_point[0] - self.start_point[0]
            dy = self.end_point[1] - self.start_point[1]
            angle = np.arctan2(dy, dx) * (180 / np.pi)
            (h, w) = self.img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_image = cv2.warpAffine(self.img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            img_output_path = os.path.join(gs.output_path, rs.file_name)
            cv2.imwrite(img_output_path, rotated_image)
            
            logging.info(f"Rotated image saved to: {img_output_path}")
            return rotated_image 
        
        except Exception as e:
            raise CustomException(e,sys)