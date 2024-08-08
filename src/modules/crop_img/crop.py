import cv2
import sys
from src.logger import logging
from src.exception import CustomException

class CropImage:
    def __init__(self, imgs, output_path):
        self.imgs = imgs
        self.output_path = output_path
        logging.info('Initialize crop image module ...')
    
    def crop_blank_pixels(self):
        try:
            crop_imgs = []
            for img in self.imgs:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
                coords = cv2.findNonZero(thresh)

                if coords is not None:
                    x, y, w, h = cv2.boundingRect(coords)
                    cropped_image = img[y:y+h, x:x+w]
                    crop_imgs.append(cropped_image)
                    cv2.imwrite(self.output_path, cropped_image)
                    print(f"Cropped image saved to {self.output_path}")

            return crop_imgs
        
        except Exception as e:
            raise CustomException(e,sys)