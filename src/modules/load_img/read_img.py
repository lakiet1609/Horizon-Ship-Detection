import cv2
import sys
import os
from src.logger import logging
from src.exception import CustomException

class ReadImage:
    def __init__(self, root):
        self.root = root
        self.paths = [os.path.join(self.root, file_path) for file_path in os.listdir(self.root)]
        logging.info('Initialize read image module ...')
    
    
    def read_image(self) -> list:
        try:
            list_of_imgs = [cv2.imread(img) for img in self.paths]
            logging.info('Finished reading all images.')
            return list_of_imgs
            
        except Exception as e:
            raise CustomException(e,sys)
    
    
    