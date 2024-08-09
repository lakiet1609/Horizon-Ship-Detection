import cv2
import sys
import os
from src.logger import logging
from src.exception import CustomException
from src.config.configuration import general_settings as gs
from src.config.configuration import crop_settings as cs


class CropImage:
    def __init__(self, img):
        self.img = img
        logging.info('Initialize crop image module ...')
    
    def crop_blank_pixels(self):
        try:

            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            coords = cv2.findNonZero(thresh)

            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                cropped_image = self.img[y:y+h, x:x+w]
                img_output_path = os.path.join(gs.output_path, cs.file_name)
                cv2.imwrite(img_output_path, cropped_image)
                print(f"Cropped image saved to {img_output_path}")

            return cropped_image
        
        except Exception as e:
            raise CustomException(e,sys)