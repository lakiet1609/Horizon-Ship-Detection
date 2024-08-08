import cv2
import os
import sys
import numpy as np
from src.logger import logging
from src.exception import CustomException

class RotateImage:
    def __init__(self, imgs, list_points, output_path):
        self.imgs = imgs
        self.list_points = list_points
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)
        
        logging.info('Initialize rotate image module ...')
    
    def rotate_image_to_level(self):
        try:
            rotate_imgs = []
            for img, point in zip(self.imgs, self.list_points):
                start_point, end_point = point[0], point[1]
                dx = end_point[0] - start_point[0]
                dy = end_point[1] - start_point[1]
                angle = np.arctan2(dy, dx) * (180 / np.pi)
                (h, w) = img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_image = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                rotate_imgs.append(rotated_image)
                cv2.imwrite(self.output_path, rotated_image)
                logging.info(f"Rotated image saved to: {self.output_path}")
            return rotate_imgs 
        
        except Exception as e:
            raise CustomException(e,sys)