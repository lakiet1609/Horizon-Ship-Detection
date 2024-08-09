from src.modules.load_img.read_img import ReadImage
from src.modules.detect_horizon.horizon import DetectHorizon
from src.modules.rotate_img.rotate_image import RotateImage
from src.modules.crop_img.crop import CropImage
from src.modules.detect_ships.ship_detection import ShipDetection
from src.config.configuration import general_settings as gs
from src.logger import logging

img_path = gs.input_path

##STAGE 1: READ THE IMAGE
logging.info('INITIALIZE STAGE 1')
image_data = ReadImage().read_image(img_path)
logging.info('FINISHED STAGE 1')


##STAGE 2: DETECT HORIZON
logging.info('INITIALIZE STAGE 2')
start_point, end_point = DetectHorizon(image_data).horizon_detection()
logging.info('FINISHED STAGE 2')


##STAGE 3: ROTATE IMAGE TO ADJUST
logging.info('INITIALIZE STAGE 3')
rotated_image = RotateImage(image_data, start_point, end_point).rotate_image_to_level()
logging.info('FINISHED STAGE 3')


##STAGE 4: CROP THE IMAGE DOWN
logging.info('INITIALIZE STAGE 4')
cropped_image = CropImage(rotated_image).crop_blank_pixels()
logging.info('FINISHED STAGE 4')


##STAGE 5: EDGE DETECTION TO DETECT SHIPS AND DRAW BOUNDING BOX
logging.info('INITIALIZE STAGE 5')
output_path = ShipDetection(cropped_image).save_and_draw_bb_img()
logging.info('FINISHED STAGE 5')