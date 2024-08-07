import cv2
from src.logger import logging
from src.exception import CustomException

class ComputerVisionTest:
    def __init__(self, img):
        self.img = img
        logging.info('Initialize computer vision test ...')
    
            
    